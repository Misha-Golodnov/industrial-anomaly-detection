"""Continuous TEP replay at a fixed normal full-run alarm budget."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_ashrae_benchmark import empirical_threshold

from industrial_novelty.data.experiment import split_run_ids
from industrial_novelty.data.loading import load_tep_file
from industrial_novelty.data.scaling import ChannelScaler
from industrial_novelty.data.schema import (
    ALL_CHANNELS,
    FAULT_ONSET_SAMPLE,
    LAST_NORMAL_SAMPLE,
    SAMPLE_PERIOD_MINUTES,
    SHUTDOWN_PRESSURE_COLUMN,
    SHUTDOWN_PRESSURE_KPA,
)
from industrial_novelty.models.canonical import CanonicalVariateIndex
from industrial_novelty.models.chiller_precision import ReferencePrecision
from industrial_novelty.models.fusion import TailMax
from industrial_novelty.models.innovation import InnovationScan
from industrial_novelty.models.slow import SlowFeatureIndex
from industrial_novelty.models.subspace import WindowMomentIndex, tail_scale

ALPHA = 0.05
WINDOW = 24
STRIDE = 8
HORIZONS = (30, 60, 120)
HARD = (3, 9, 15)


def runs_from_frame(frame, scaler):
    ordered = frame.sort_values(["simulationRun", "sample"])
    ids = ordered.simulationRun.unique()
    sample = ordered["sample"].to_numpy().reshape(len(ids), -1)
    if not np.all(sample == np.arange(1, sample.shape[1] + 1)):
        raise ValueError("Expected equal-length TEP runs with a complete sample grid.")
    pressure = ordered[SHUTDOWN_PRESSURE_COLUMN].to_numpy().reshape(sample.shape)
    trip = pressure >= SHUTDOWN_PRESSURE_KPA
    first_trip = np.where(
        trip.any(axis=1), trip.argmax(axis=1) + 1, sample.shape[1] + 1
    )
    values = scaler.transform_array(ordered.loc[:, list(ALL_CHANNELS)].to_numpy())
    return ids, values.reshape(*sample.shape, len(ALL_CHANNELS)), first_trip


def endpoints(length, stride=STRIDE):
    """One-based decision times, with identical warmup for all methods."""
    return np.arange(WINDOW, length + 1, stride)


def window_array(runs, times):
    shape = (len(runs) * len(times), WINDOW, runs.shape[-1])
    indices = times[:, None] + np.arange(-WINDOW, 0)
    return runs[:, indices].reshape(shape)


class DynamicReplay:
    def fit(self, train):
        self.dynamic = TailMax(
            (CanonicalVariateIndex(), WindowMomentIndex(), SlowFeatureIndex())
        ).fit(train)
        raw = self.dynamic.score(train)
        self.center = raw.mean()
        self.spread = max(raw.std(), 1e-8)
        return self

    def score(self, windows, shape):
        raw = self.dynamic.score(windows).reshape(shape)
        return lfilter([0.2], [1, -0.8], (raw - self.center) / self.spread, axis=1)


class ReplaySuite:
    def fit(self, fit, reference):
        train = window_array(fit, endpoints(fit.shape[1]))
        ref = window_array(reference, endpoints(reference.shape[1]))
        self.dynamic = DynamicReplay().fit(train)
        self.precision = ReferencePrecision("raw").fit(train, ref)
        self.innovation = InnovationScan().fit(fit, reference)
        ref_scores = self._score(reference)
        self.hybrid_scales = {
            name: tail_scale(ref_scores[name].reshape(-1), 0.01)
            for name in ("ashrae_precision", "innovation_joint_s8")
        }
        return self

    def _score(self, runs):
        times = endpoints(runs.shape[1])
        X = window_array(runs, times)
        shape = (len(runs), len(times))
        scores = {
            "tep_dynamic": self.dynamic.score(X, shape),
            "ashrae_precision": self.precision.score(X).reshape(shape),
        }
        for name, values in self.innovation.score(runs).items():
            scores[f"innovation_{name}_s8"] = values[:, times - 1]
            scores[f"innovation_{name}_s1"] = values[:, WINDOW - 1 :]
        return scores

    def score(self, runs):
        pieces = {}
        for start in range(0, len(runs), 40):
            scores = self._score(runs[start : start + 40])
            scores["max_precision_innovation_s8"] = np.maximum.reduce(
                [scores[name] / scale for name, scale in self.hybrid_scales.items()]
            )
            for name, values in scores.items():
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite scores for {name}.")
                pieces.setdefault(name, []).append(values)
        return {name: np.concatenate(values) for name, values in pieces.items()}


def alarm_times(scores, times, threshold, first_trip):
    eligible = times[None, :] < first_trip[:, None]
    active = (scores > threshold) & eligible
    before = (active & (times < FAULT_ONSET_SAMPLE)).any(axis=1)
    after = times >= FAULT_ONSET_SAMPLE
    crossings = active & ~np.column_stack(
        [np.zeros(len(active), dtype=bool), active[:, :-1]]
    )

    # NaN denotes no alert by the end of the valid observation period.
    def first(mask):
        mask = mask & after
        return np.where(mask.any(axis=1), times[mask.argmax(axis=1)], np.nan)

    return before, first(active), first(crossings)


def score_cases(suite, runs, ids, first_trip, case, thresholds):
    rows = []
    for name, scores in suite.score(runs).items():
        stride = 1 if name.endswith("_s1") else STRIDE
        times = endpoints(runs.shape[1], stride=stride)
        threshold = thresholds[name]
        before, first, crossing = alarm_times(scores, times, threshold, first_trip)
        rows.append(
            pd.DataFrame(
                {
                    "fault": case,
                    "method": name,
                    "run": ids,
                    "alarm_before_onset": before,
                    "delay_min": (first - LAST_NORMAL_SAMPLE) * SAMPLE_PERIOD_MINUTES,
                    "crossing_delay_min": (crossing - LAST_NORMAL_SAMPLE)
                    * SAMPLE_PERIOD_MINUTES,
                    "shutdown_sample": np.where(
                        first_trip <= runs.shape[1], first_trip, np.nan
                    ),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def aggregate(rows, thresholds):
    results = []
    for (fault, method), part in rows.groupby(["fault", "method"]):
        delay = part.delay_min
        crossing = part.crossing_delay_min
        stats = {
            "fault": fault,
            "method": method,
            "runs": len(part),
            "alarm_before_onset": part.alarm_before_onset.mean(),
            "any_alarm": (part.alarm_before_onset | delay.notna()).mean(),
            "detected": delay.notna().mean(),
            "median_delay_min_detected": delay.median()
            if delay.notna().any()
            else np.nan,
        }
        for horizon in HORIZONS:
            stats[f"by_{horizon}min"] = delay.le(horizon).mean()
            stats[f"crossing_by_{horizon}min"] = crossing.le(horizon).mean()
        results.append(stats)
    per_fault = pd.DataFrame(results)
    normal = per_fault.loc[per_fault.fault.eq(0)].copy()
    normal["threshold"] = normal.method.map(thresholds)
    summary = pd.concat(
        [
            subset.groupby("method")
            .mean(numeric_only=True)
            .drop(columns=["fault", "runs"])
            .assign(subset=label)
            for label, subset in (
                ("IDV_3_9_15", per_fault.loc[per_fault.fault.isin(HARD)]),
                ("other_faults", per_fault.loc[~per_fault.fault.isin((0, *HARD))]),
            )
            if not subset.empty
        ]
    ).reset_index()
    summary["normal_fpr_full_run"] = summary.method.map(
        normal.set_index("method").any_alarm
    )
    return per_fault, normal, summary


def paired_differences(rows, *, baselines=("tep_dynamic", "ashrae_precision")):
    """Bootstrap run IDs, keeping the three weak faults and methods paired."""
    weak = rows.loc[rows.fault.isin(HARD)]
    if set(weak.fault) != set(HARD):
        return pd.DataFrame()
    delays = weak.pivot(index="run", columns=["fault", "method"], values="delay_min")
    methods = delays.columns.get_level_values("method").unique()
    rng = np.random.default_rng(1043)
    weights = rng.multinomial(
        len(delays), np.full(len(delays), 1 / len(delays)), size=4000
    )
    comparisons = []
    for horizon in HORIZONS:
        detected = delays.le(horizon).T.groupby("method").mean().T
        for baseline in baselines:
            for method in methods.difference([baseline]):
                delta = (detected[method] - detected[baseline]).to_numpy()
                estimates = weights @ delta / len(delta)
                lower, upper = np.quantile(estimates, [0.025, 0.975])
                comparisons.append(
                    {
                        "method": method,
                        "baseline": baseline,
                        "horizon_minutes": horizon,
                        "delta_recall": delta.mean(),
                        "q025": lower,
                        "q975": upper,
                    }
                )
    return pd.DataFrame(comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--faults", nargs="+", type=int, default=list(range(1, 21)))
    args = parser.parse_args()
    if len(set(args.faults)) != len(args.faults) or not set(args.faults) <= set(
        range(1, 21)
    ):
        parser.error("--faults must contain distinct values from 1 to 20")
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with threadpool_limits(limits=4):
        train = load_tep_file(args.data / "TEP_FaultFree_Training.RData")
        fit_ids, ref_ids = split_run_ids(train.simulationRun, val_fraction=0.2, seed=0)
        mask = train.simulationRun.isin(fit_ids)
        scaler = ChannelScaler().fit(train.loc[mask], ALL_CHANNELS)
        _, fit, _ = runs_from_frame(train.loc[mask], scaler)
        _, reference, _ = runs_from_frame(train.loc[~mask], scaler)
        del train
        suite = ReplaySuite().fit(fit, reference)
        print(f"Fit complete in {perf_counter() - start:.1f}s", flush=True)
        normal = load_tep_file(args.data / "TEP_FaultFree_Testing.RData")
        normal_ids = np.sort(normal.simulationRun.unique())
        np.random.default_rng(20261030).shuffle(normal_ids)
        calibration = normal.simulationRun.isin(normal_ids[:200])
        _, cal, cal_trips = runs_from_frame(normal.loc[calibration], scaler)
        test_ids, test, test_trips = runs_from_frame(normal.loc[~calibration], scaler)
        del normal
        if (cal_trips <= cal.shape[1]).any() or (test_trips <= test.shape[1]).any():
            raise ValueError("Expected fault-free runs with a common complete horizon.")
        thresholds = {
            name: empirical_threshold(values.max(axis=1), ALPHA)
            for name, values in suite.score(cal).items()
        }
        manifest = {
            "alpha": ALPHA,
            "window": WINDOW,
            "stride_baseline": STRIDE,
            "horizons_minutes": HORIZONS,
            "fault_onset_sample": FAULT_ONSET_SAMPLE,
            "fit_run_ids": fit_ids.tolist(),
            "reference_run_ids": ref_ids.tolist(),
            "calibration_run_ids": sorted(normal_ids[:200].tolist()),
            "test_run_ids": test_ids.tolist(),
            "innovation_order": suite.innovation.order,
            "innovation_widths": suite.innovation.widths,
            "covariance_selection": suite.precision.diagnostics_,
            "faults": args.faults,
            "shutdown_pressure_cutoff_kpa": SHUTDOWN_PRESSURE_KPA,
            "thresholds": thresholds,
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        records = [score_cases(suite, test, test_ids, test_trips, 0, thresholds)]
        faulty = load_tep_file(args.data / "TEP_Faulty_Testing.RData")
        for fault in args.faults:
            ids, runs, trips = runs_from_frame(
                faulty.loc[faulty.faultNumber.eq(fault)], scaler
            )
            records.append(score_cases(suite, runs, ids, trips, fault, thresholds))
            print(
                f"IDV {fault:02d} complete ({perf_counter() - start:.1f}s)", flush=True
            )
    rows = pd.concat(records, ignore_index=True)
    per_fault, normal, summary = aggregate(rows, thresholds)
    rows.to_csv(
        args.output / "per_run.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    per_fault.to_csv(args.output / "per_fault.csv", index=False)
    normal.to_csv(args.output / "normal.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    paired_differences(rows).to_csv(args.output / "paired_weak.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
