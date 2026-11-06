"""Compare TEP, RP-1043 and SDAHU normal-only winners on each target dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_ashrae_benchmark import empirical_threshold
from run_lbnl_sdahu import frame_day, maxima, paired_maxima, unpack

from industrial_novelty.data.ashrae import CHANNELS as CHILLER_CHANNELS
from industrial_novelty.data.ashrae import FAULTS, chiller_windows, normal_split
from industrial_novelty.data.experiment import ExperimentConfig, build_experiment
from industrial_novelty.data.lbnl import (
    BASIC,
    INTERVAL,
    NORMAL,
    SCENARIOS,
    fold_roles,
    occupied_windows,
    read_sdahu,
    week_index,
)
from industrial_novelty.data.schema import LAST_NORMAL_SAMPLE, SAMPLE_PERIOD_MINUTES
from industrial_novelty.models.chiller_directions import contiguous_groups
from industrial_novelty.models.lbnl import windows
from industrial_novelty.models.transfer import TransferSuite

ALPHAS = {"tep": (0.05, 0.10), "ashrae": (0.05, 0.10), "lbnl": (0.05, 0.10)}
PRIMARY = {"tep": 0.05, "ashrae": 0.05, "lbnl": 0.10}
TEP_HARD = (3, 9, 15)
AHU_WEAK = {"sa_bias": 2, "coil_stuck": 10, "damper_stuck": 10}
METRICS = ["recall", "auroc", "pauc05", "normal_fpr", "matched_normal_fpr"]


class Benchmark:
    def __init__(self, dataset, protocol, split):
        self.key = {"dataset": dataset, "protocol": protocol, "split": split}
        self.alpha = ALPHAS[dataset]
        self.per_case = []
        self.normal = []
        self.thresholds = {}

    def calibrate(self, calibration, normal):
        if calibration.empty or normal.empty:
            raise ValueError(f"Empty calibration or normal decisions: {self.key}")
        self.normal_scores = normal
        for alpha in self.alpha:
            for method in normal:
                q = empirical_threshold(calibration[method], alpha)
                self.thresholds[alpha, method] = q
                self.normal.append(
                    {
                        **self.key,
                        "alpha": alpha,
                        "method": method,
                        "threshold": q,
                        "calibration_n": len(calibration),
                        "normal_n": len(normal),
                        "normal_fpr": (normal[method] > q).mean(),
                    }
                )

    def evaluate(self, scores, case, family, level, *, normal=None, first=None):
        normal = self.normal_scores if normal is None else normal
        if scores.empty or normal.empty:
            raise ValueError(f"Empty scored case: {self.key}, {case}")
        y = np.r_[np.zeros(len(normal)), np.ones(len(scores))]
        for method in scores:
            pool = np.r_[normal[method], scores[method]]
            auc = roc_auc_score(y, pool)
            pauc = roc_auc_score(y, pool, max_fpr=0.05)
            for alpha in self.alpha:
                threshold = self.thresholds[alpha, method]
                row = {
                    **self.key,
                    "scenario": case,
                    "family": family,
                    "level": level,
                    "method": method,
                    "alpha": alpha,
                    "fault_n": len(scores),
                    "matched_normal_n": len(normal),
                    "recall": (scores[method] > threshold).mean(),
                    "matched_normal_fpr": (normal[method] > threshold).mean(),
                    "auroc": auc,
                    "pauc05": pauc,
                }
                if first is not None:
                    detected = first[method].loc[first[method]["score"] > threshold]
                    delay = detected.groupby("group")["end"].min() - LAST_NORMAL_SAMPLE
                    row["detected_by_120min"] = delay.le(
                        120 / SAMPLE_PERIOD_MINUTES
                    ).sum() / len(scores)
                    row["median_delay_min_detected"] = (
                        delay.median() * SAMPLE_PERIOD_MINUTES
                    )
                self.per_case.append(row)

    def save(self, out, suite, metadata):
        prefix = (
            out / f"{self.key['dataset']}_{self.key['protocol']}_{self.key['split']}"
        )
        pd.DataFrame(self.per_case).to_csv(
            prefix.with_suffix(".cases.csv"), index=False
        )
        pd.DataFrame(self.normal).to_csv(prefix.with_suffix(".normal.csv"), index=False)
        manifest = {
            **self.key,
            **metadata,
            "selected_shrinkage": suite.precision.diagnostics_,
            "virtual_graph": suite.virtual.graph(),
            "fusion_scales": suite.fusion_scales,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        }
        prefix.with_suffix(".json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def score(suite, X, groups, ends, *, index=None):
    values, methods = suite.score(X, np.asarray(groups), np.asarray(ends))
    return pd.DataFrame(values, columns=methods, index=index)


def tep_scores(suite, batch):
    meta = batch.meta
    return score(suite, batch.X, meta.simulation_run, meta.sample_end)


def run_tep(raw, out):
    experiment = build_experiment(
        ExperimentConfig(data_dir=raw, window_size=24, stride=8, channels="all")
    )
    train, reference, normal = experiment.train, experiment.val, experiment.normal_test
    suite = TransferSuite(train.channel_names, pre_scaled=True).fit(
        train.X, reference.X, reference.meta.simulation_run, reference.meta.sample_end
    )
    ids = normal.meta.simulation_run.unique().copy()
    np.random.default_rng(20261030).shuffle(ids)
    calibration_ids = np.sort(ids[:200])
    cal = normal.meta.simulation_run.isin(calibration_ids)
    scored_normal = tep_scores(suite, normal)
    maxima_normal = scored_normal.groupby(normal.meta.simulation_run).max()
    bench = Benchmark("tep", "full_run", 0)
    bench.calibrate(
        maxima_normal.loc[calibration_ids], maxima_normal.drop(calibration_ids)
    )
    # Ranking compares post-onset scores on both classes. Operational FPR still
    # includes the whole normal run, as in the source TEP experiment.
    after_onset = normal.meta.sample_start.gt(LAST_NORMAL_SAMPLE) & ~cal
    matched = (
        scored_normal.loc[after_onset]
        .groupby(normal.meta.simulation_run.loc[after_onset])
        .max()
    )
    for item in experiment.iter_faulty_test():
        batch = item.batch
        meta = batch.meta
        scored = tep_scores(suite, batch)
        eligible = meta.label.eq(1) & ~meta.post_shutdown
        post = scored.loc[eligible]
        ids = meta.simulation_run.loc[eligible]
        first = {
            method: pd.DataFrame(
                {
                    "score": post[method],
                    "group": ids,
                    "end": meta.sample_end.loc[eligible],
                }
            )
            for method in post
        }
        bench.evaluate(
            post.groupby(ids).max(),
            f"idv_{item.fault_number:02d}",
            item.fault_number,
            None,
            normal=matched,
            first=first,
        )
        print(f"TEP fault {item.fault_number:02d} done", flush=True)
    bench.save(
        out,
        suite,
        {
            "window": 24,
            "stride": 8,
            "channels": train.channel_names,
            "calibration_run_ids": calibration_ids.tolist(),
            "test_run_ids": maxima_normal.index.difference(calibration_ids).tolist(),
            "fit_run_ids": sorted(train.meta.simulation_run.unique().tolist()),
            "reference_run_ids": sorted(
                reference.meta.simulation_run.unique().tolist()
            ),
            "fault_scoreable": "fully after onset and not fully after recorded shutdown",
        },
    )


def ashrae_groups(meta):
    parts = contiguous_groups(meta)
    return np.repeat(np.arange(len(parts)), [len(part) for part in parts])


def ashrae_scores(suite, batch):
    return score(suite, batch.X, ashrae_groups(batch.meta), batch.meta.row_end)


def run_ashrae(raw, out):
    normal = chiller_windows(raw / "normal10.csv", window=48)
    faulty = {
        (fault, level): chiller_windows(raw / f"{fault}{level}.csv", window=48)
        for fault in FAULTS
        for level in (10, 20, 30, 40)
    }
    for band in (None, 50):
        protocol = "blocked" if band is None else "hold50"
        for seed in range(5):
            parts, assignment = normal_split(normal, seed, held_out_band=band)
            fit, ref = parts["fit"], parts["reference"]
            suite = TransferSuite(CHILLER_CHANNELS).fit(
                fit.X, ref.X, ashrae_groups(ref.meta), ref.meta.row_end
            )
            bench = Benchmark("ashrae", protocol, seed)
            bench.calibrate(
                ashrae_scores(suite, parts["calibration"]),
                ashrae_scores(suite, parts["test"]),
            )
            for (fault, level), part in faulty.items():
                if band is not None:
                    part = part.take(
                        np.flatnonzero(part.meta.temperature_band.eq(band))
                    )
                bench.evaluate(
                    ashrae_scores(suite, part), f"{fault}{level}", fault, level
                )
            bench.save(
                out,
                suite,
                {
                    "window": 48,
                    "stride": 6,
                    "channels": CHILLER_CHANNELS,
                    "normal_block_roles": assignment,
                    "source": "Author CSV derivative of ASHRAE RP-1043",
                },
            )
            print(f"ASHRAE {protocol} split {seed} done", flush=True)


def lbnl_coords(index):
    days = frame_day(index)
    end = index.asi8
    if index.unit != "ns":
        end = index.as_unit("ns").asi8
    boundary = (days[1:] != days[:-1]) | (index[1:] - index[:-1] != INTERVAL)
    return np.cumsum(np.r_[True, boundary]), end


def lbnl_scores(suite, frame, ends):
    index = frame.index[ends]
    X = windows(frame.loc[:, BASIC].to_numpy(), ends, 4)
    return score(suite, X, *lbnl_coords(index), index=index)


def run_lbnl(archive, out):
    frames, inventory = read_sdahu(unpack(archive))
    unknown = {
        row["family"] for row in inventory if row["canonical"] != row["scenario"]
    }
    normal = frames[NORMAL]
    ends = occupied_windows(normal, 4)
    origin = frame_day(normal.index)[0]
    raw = normal.loc[:, BASIC].to_numpy()
    coords = {
        name: occupied_windows(frame, 4)
        for name, frame in frames.items()
        if name != NORMAL
    }
    for fold in range(4):
        roles = fold_roles(week_index(normal.index[ends], origin), fold)
        ref_end = ends[roles["reference"]]
        suite = TransferSuite(BASIC, ahu_graph=True).fit(
            windows(raw, ends[roles["fit"]], 4),
            windows(raw, ref_end, 4),
            *lbnl_coords(normal.index[ref_end]),
        )
        bench = Benchmark("lbnl", "calendar", fold)
        normal_scores = lbnl_scores(suite, normal, ends[roles["test"]])
        cal = lbnl_scores(suite, normal, ends[roles["calibration"]])
        bench.calibrate(maxima(cal), maxima(normal_scores))
        for name, ids in coords.items():
            frame = frames[name]
            selected = ids[
                fold_roles(week_index(frame.index[ids], origin), fold)["test"]
            ]
            faulty_scores = lbnl_scores(suite, frame, selected)
            matched, faulty = paired_maxima(normal_scores, faulty_scores)
            family, level = SCENARIOS[name]
            bench.evaluate(
                faulty,
                name,
                family,
                None if family in unknown else level,
                normal=matched,
            )
        bench.save(
            out,
            suite,
            {
                "window": 4,
                "stride": 1,
                "interval_minutes": 15,
                "channels": BASIC,
                "week_origin": str(origin.date()),
                "archive": archive.name,
                "csv_inventory": inventory,
            },
        )
        print(f"LBNL fold {fold} done", flush=True)


def summarize(out):
    case = pd.concat(
        [pd.read_csv(p) for p in sorted(out.glob("*.cases.csv"))], ignore_index=True
    )
    normal = pd.concat(
        [pd.read_csv(p) for p in sorted(out.glob("*.normal.csv"))], ignore_index=True
    )
    keys = ["dataset", "protocol", "split", "alpha", "method"]
    rows = []
    for key, part in case.groupby(keys):
        dataset = key[0]
        if dataset == "tep":
            weak = part.family.astype(str).isin([str(n) for n in TEP_HARD])
        elif dataset == "ashrae":
            weak = part.level.eq(10)
        else:
            weak = part.level.abs().eq(part.family.map(AHU_WEAK))
        subsets = {"all": part, "weak": part.loc[weak]}
        if dataset == "tep":
            subsets["other_17"] = part.loc[~weak]
        for label, sub in subsets.items():
            metrics = sub.groupby("family")[
                ["recall", "auroc", "pauc05", "matched_normal_fpr"]
            ].mean()
            rows.append(
                dict(zip(keys, key, strict=True))
                | {"subset": label}
                | metrics.mean().to_dict()
            )
    per_split = pd.DataFrame(rows).merge(normal[keys + ["normal_fpr"]], on=keys)
    summary = per_split.groupby(["dataset", "protocol", "subset", "alpha", "method"])[
        METRICS
    ].mean()
    summary.to_csv(out / "summary.csv")
    per_split.to_csv(out / "per_split.csv", index=False)
    case.to_csv(out / "per_scenario.csv", index=False)
    normal.to_csv(out / "normal_calibration.csv", index=False)
    print(
        summary.reset_index().query("subset == 'weak'").round(3).to_string(index=False)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--datasets", nargs="+", choices=tuple(PRIMARY), default=list(PRIMARY)
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        start = perf_counter()
        with threadpool_limits(limits=4):
            for dataset in args.datasets:
                if dataset == "tep":
                    run_tep(args.data, out)
                elif dataset == "ashrae":
                    run_ashrae(args.data / "ashrae_public", out)
                else:
                    run_lbnl(args.data / "lbnl_fdd/sdahu-figshare.zip", out)
        print(f"Completed model runs in {perf_counter() - start:.1f}s", flush=True)
    summarize(out)


if __name__ == "__main__":
    main()
