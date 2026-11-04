"""Second exploratory RP-1043 round: predeclared literature-guided ablations.

Prior tests have been inspected. This is development, not confirmatory SOTA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from run_ashrae_benchmark import empirical_threshold, metrics
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from industrial_novelty.data.ashrae import FAULTS, chiller_windows, normal_split
from industrial_novelty.models.chiller_directions import (
    ConditionalResidual,
    FeatureGaussian,
    SequentialResidual,
)


def factories():
    return {
        "raw_lw": lambda: FeatureGaussian("raw"),
        "raw_oas": lambda: FeatureGaussian("raw", covariance="oas"),
        "invariant_lw": lambda: FeatureGaussian("invariant"),
        "thermo_lw": lambda: FeatureGaussian("thermo"),
        "thermo_oas": lambda: FeatureGaussian("thermo", covariance="oas"),
        "thermo_eos_lw": lambda: FeatureGaussian("thermo_eos"),
        "raw_pca_spe": lambda: FeatureGaussian("raw", pca_residual=True),
        "thermo_pca_spe": lambda: FeatureGaussian("thermo", pca_residual=True),
        "raw_spectral_lw": lambda: FeatureGaussian("spectral"),
        "thermo_spectral_lw": lambda: FeatureGaussian("thermo_spectral"),
        "thermo_poly1": lambda: ConditionalResidual("thermo", degree=1),
        "thermo_poly2": lambda: ConditionalResidual("thermo", degree=2),
        "thermo_gp": lambda: ConditionalResidual("thermo", mean_model="gp"),
        "raw_gp": lambda: ConditionalResidual("raw", mean_model="gp"),
        "raw_mewma": lambda: SequentialResidual(FeatureGaussian("raw")),
        "thermo_mewma": lambda: SequentialResidual(FeatureGaussian("thermo")),
        "thermo_scan": lambda: SequentialResidual(
            FeatureGaussian("thermo"), mode="scan"
        ),
    }


def block_alarm_rate(scores, meta, threshold):
    frame = meta[["block"]].copy()
    frame["alarm"] = scores > threshold
    return float(frame.groupby("block").alarm.max().mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/ashrae_public"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/ashrae_directions_2026-11-04")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--windows", type=int, nargs="+", default=[12, 48])
    parser.add_argument(
        "--protocols", nargs="+", default=["blocked", "loto40", "loto45", "loto50"]
    )
    parser.add_argument("--methods", nargs="+", default=list(factories()))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    score_dir = args.output / "scores"
    score_dir.mkdir(exist_ok=True)
    normal = {
        w: chiller_windows(args.data / "normal10.csv", window=w) for w in args.windows
    }
    plans = []
    for w in args.windows:
        for protocol in args.protocols:
            for seed in args.seeds:
                band = None if protocol == "blocked" else int(protocol[4:])
                parts, assignments = normal_split(normal[w], seed, held_out_band=band)
                plans.append(
                    {
                        "window": w,
                        "protocol": protocol,
                        "seed": seed,
                        "band": band,
                        "blocks": assignments,
                        "counts": {k: len(v.X) for k, v in parts.items()},
                    }
                )
    manifest = {
        "date": "2026-11-04",
        "status": "exploratory development; earlier tests were inspected",
        "methods": args.methods,
        "plans": plans,
        "threshold_budgets": [0.05, 0.1],
        "selection": "fixed grid; no fault-based hyperparameter tuning within this round",
        "physical_features": "Electronics 2026 Table 3 formulas; raw subcooling is already a delta",
        "eos": "separate R134a/Fahrenheit/psig ablation; 1 atm reference",
        "sequential": "reset per artificial block and any missing window; no runwise guarantee",
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "scikit-learn", "CoolProp")
        },
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path("src/industrial_novelty/models/chiller_directions.py"),
                Path("src/industrial_novelty/data/ashrae.py"),
            ]
        },
        "data_manifest_sha256": hashlib.sha256(
            Path("results/ashrae_2026-11-04/data_manifest.json").read_bytes()
        ).hexdigest(),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    all_results, timing = [], []
    for w in args.windows:
        faults = {
            (f, s): chiller_windows(args.data / f"{f}{s}.csv", window=w)
            for f in FAULTS
            for s in (10, 20, 30, 40)
        }
        for plan in [p for p in plans if p["window"] == w]:
            parts, _ = normal_split(normal[w], plan["seed"], held_out_band=plan["band"])
            for name in args.methods:
                model = factories()[name]()
                started = time.perf_counter()
                with threadpool_limits(limits=2):
                    model.fit(parts["fit"].X, parts["reference"].X)
                    cal = model.score(parts["calibration"].X, parts["calibration"].meta)
                    norm = model.score(parts["test"].X, parts["test"].meta)
                    if not np.isfinite(cal).all() or not np.isfinite(norm).all():
                        raise ValueError(f"Nonfinite normal scores: {name}")
                    thresholds = {a: empirical_threshold(cal, a) for a in (0.05, 0.1)}
                    stored = {
                        "calibration": cal,
                        "normal": norm,
                        "normal_row_end": parts["test"].meta.row_end.to_numpy(),
                        "normal_block": parts["test"].meta.block.to_numpy(),
                    }
                    for (fault, severity), batch in faults.items():
                        keep = (
                            np.ones(len(batch.X), bool)
                            if plan["band"] is None
                            else batch.meta.temperature_band.to_numpy() == plan["band"]
                        )
                        selected = batch.take(np.flatnonzero(keep))
                        scores = model.score(selected.X, selected.meta)
                        if not np.isfinite(scores).all():
                            raise ValueError(f"Nonfinite faulty scores: {name}")
                        stored[f"{fault}{severity}"] = scores
                        stored[f"{fault}{severity}_block"] = (
                            selected.meta.block.to_numpy()
                        )
                        y = np.r_[np.zeros(len(norm)), np.ones(len(scores))]
                        pauc = roc_auc_score(y, np.r_[norm, scores], max_fpr=0.05)
                        for alpha, threshold in thresholds.items():
                            all_results.append(
                                {
                                    "window": w,
                                    "protocol": plan["protocol"],
                                    "seed": plan["seed"],
                                    "method": name,
                                    "fault": fault,
                                    "severity": severity // 10,
                                    "alpha": alpha,
                                    "threshold": threshold,
                                    "pauc05": pauc,
                                    "normal_block_alarm_rate": block_alarm_rate(
                                        norm, parts["test"].meta, threshold
                                    ),
                                    "fault_block_alarm_rate": block_alarm_rate(
                                        scores, selected.meta, threshold
                                    ),
                                    **metrics(norm, scores, threshold),
                                }
                            )
                np.savez_compressed(
                    score_dir / f"w{w}_{plan['protocol']}_s{plan['seed']}_{name}.npz",
                    **stored,
                )
                timing.append(
                    {
                        "window": w,
                        "protocol": plan["protocol"],
                        "seed": plan["seed"],
                        "method": name,
                        "fit_and_all_inference_seconds": time.perf_counter() - started,
                    }
                )
                print(
                    f"w={w} {plan['protocol']} seed={plan['seed']} {name}: {timing[-1]['fit_and_all_inference_seconds']:.2f}s",
                    flush=True,
                )
            pd.DataFrame(all_results).to_csv(args.output / "per_fault.csv", index=False)
            pd.DataFrame(timing).to_csv(args.output / "timing.csv", index=False)
    frame = pd.DataFrame(all_results)
    measures = [
        "auroc",
        "pauc05",
        "balanced_ap",
        "recall",
        "normal_fpr",
        "normal_block_alarm_rate",
        "fault_block_alarm_rate",
    ]
    by_seed = (
        frame.groupby(["window", "protocol", "seed", "method", "severity", "alpha"])[
            measures
        ]
        .mean()
        .reset_index()
    )
    by_seed.to_csv(args.output / "per_seed.csv", index=False)
    summary = by_seed.groupby(["window", "protocol", "method", "severity", "alpha"])[
        measures
    ].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(args.output / "summary.csv", index=False)
    print(f"Completed {len(timing)} development fits.")


if __name__ == "__main__":
    main()
