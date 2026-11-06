"""Combine TEP dynamics and VAR mean innovations at one full-run alarm budget."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from run_tep_early import (
    ALPHA,
    ROOT,
    STRIDE,
    WINDOW,
    DynamicReplay,
    aggregate,
    empirical_threshold,
    endpoints,
    paired_differences,
    runs_from_frame,
    score_cases,
    window_array,
)
from threadpoolctl import threadpool_limits

from industrial_novelty.data.experiment import split_run_ids
from industrial_novelty.data.loading import load_tep_file
from industrial_novelty.data.scaling import ChannelScaler
from industrial_novelty.data.schema import ALL_CHANNELS
from industrial_novelty.models.innovation import InnovationScan
from industrial_novelty.models.subspace import tail_scale

BRANCHES = ("tep_dynamic", "innovation_mean_s8")
FUSIONS = ("fusion_max", "fusion_log_mean", "fusion_split_alpha")


def combine_scores(scores, reference_scales, branch_limits):
    """The three combinations use the same two causal branch scores."""
    scaled = [scores[name] / reference_scales[name] for name in BRANCHES]
    split = [scores[name] / branch_limits[name] for name in BRANCHES]
    return {
        "fusion_max": np.maximum.reduce(scaled),
        "fusion_log_mean": np.mean(
            [np.log1p(np.maximum(values, 0)) for values in scaled], axis=0
        ),
        "fusion_split_alpha": np.maximum.reduce(split),
    }


class FusionReplay:
    def fit(self, fit, reference):
        self.dynamic = DynamicReplay().fit(window_array(fit, endpoints(fit.shape[1])))
        self.innovation = InnovationScan().fit(fit, reference)
        self.reference_scales = {
            name: tail_scale(values.ravel(), 0.01)
            for name, values in self.branch_scores(reference).items()
        }
        return self

    def branch_scores(self, runs):
        times = endpoints(runs.shape[1])
        dynamic, mean = [], []
        for start in range(0, len(runs), 40):
            part = runs[start : start + 40]
            dynamic.append(
                self.dynamic.score(window_array(part, times), (len(part), len(times)))
            )
            mean.append(self.innovation.score_mean(part)[:, times - 1])
        return dict(
            zip(BRANCHES, (np.concatenate(dynamic), np.concatenate(mean)), strict=True)
        )

    def calibrate(self, runs):
        branches = self.branch_scores(runs)
        self.branch_limits = {
            name: empirical_threshold(scores.max(axis=1), ALPHA / len(BRANCHES))
            for name, scores in branches.items()
        }
        if any(not np.isfinite(q) or q <= 0 for q in self.branch_limits.values()):
            raise ValueError("Split alpha requires finite positive branch thresholds.")
        scores = branches | combine_scores(
            branches, self.reference_scales, self.branch_limits
        )
        return {
            name: 1.0
            if name == "fusion_split_alpha"
            else empirical_threshold(values.max(axis=1), ALPHA)
            for name, values in scores.items()
        }

    def score(self, runs):
        branches = self.branch_scores(runs)
        result = branches | combine_scores(
            branches, self.reference_scales, self.branch_limits
        )
        if not all(np.isfinite(values).all() for values in result.values()):
            raise ValueError("Nonfinite TEP fusion scores.")
        return result


def normal_fpr_differences(rows):
    normal = (
        rows.loc[rows.fault.eq(0)]
        .assign(alarm=lambda x: x.alarm_before_onset | x.delay_min.notna())
        .pivot(index="run", columns="method", values="alarm")
    )
    n = len(normal)
    draws = np.random.default_rng(1043).multinomial(n, np.full(n, 1 / n), size=4000)
    records = []
    for name in FUSIONS:
        for baseline in BRANCHES:
            delta = (normal[name].astype(int) - normal[baseline].astype(int)).to_numpy()
            lower, upper = np.quantile(draws @ delta / n, [0.025, 0.975])
            records.append(
                {
                    "method": name,
                    "baseline": baseline,
                    "delta_fpr": delta.mean(),
                    "q025": lower,
                    "q975": upper,
                }
            )
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with threadpool_limits(limits=4):
        train = load_tep_file(args.data / "TEP_FaultFree_Training.RData")
        fit_ids, ref_ids = split_run_ids(train.simulationRun, val_fraction=0.2, seed=0)
        mask = train.simulationRun.isin(fit_ids)
        scaler = ChannelScaler().fit(train.loc[mask], ALL_CHANNELS)
        _, fit, _ = runs_from_frame(train.loc[mask], scaler)
        _, reference, _ = runs_from_frame(train.loc[~mask], scaler)
        del train
        suite = FusionReplay().fit(fit, reference)
        normal = load_tep_file(args.data / "TEP_FaultFree_Testing.RData")
        normal_ids = np.sort(normal.simulationRun.unique())
        np.random.default_rng(20261030).shuffle(normal_ids)
        cal_mask = normal.simulationRun.isin(normal_ids[:200])
        cal_ids, cal, cal_trips = runs_from_frame(normal.loc[cal_mask], scaler)
        test_ids, test, test_trips = runs_from_frame(normal.loc[~cal_mask], scaler)
        del normal
        if (cal_trips <= cal.shape[1]).any() or (test_trips <= test.shape[1]).any():
            raise ValueError("Expected fault-free runs with a common complete horizon.")
        thresholds = suite.calibrate(cal)
        manifest = {
            "nominal_alpha_per_run": ALPHA,
            "window_dynamic": WINDOW,
            "stride": STRIDE,
            "fit_run_ids": fit_ids.tolist(),
            "reference_run_ids": ref_ids.tolist(),
            "calibration_run_ids": cal_ids.tolist(),
            "test_run_ids": test_ids.tolist(),
            "reference_window_q99": suite.reference_scales,
            "split_alpha": ALPHA / len(BRANCHES),
            "split_alpha_branch_limits": suite.branch_limits,
            "thresholds": thresholds,
            "faults": list(range(1, 21)),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        records = [score_cases(suite, test, test_ids, test_trips, 0, thresholds)]
        print(
            f"Fit and calibration complete ({perf_counter() - start:.1f}s)", flush=True
        )
        faulty = load_tep_file(args.data / "TEP_Faulty_Testing.RData")
        for fault in manifest["faults"]:
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
        out / "per_run.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    per_fault.to_csv(out / "per_fault.csv", index=False)
    normal.to_csv(out / "normal.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired_differences(rows, baselines=BRANCHES).to_csv(
        out / "paired_weak.csv", index=False
    )
    normal_fpr_differences(rows).to_csv(out / "paired_normal.csv", index=False)
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
