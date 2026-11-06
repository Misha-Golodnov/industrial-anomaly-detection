"""Compare 12/48-row scores on identical calibration and test window ends.

This controls the test-population change; training contexts still differ.
It is a post-hoc development analysis, not a new untouched test.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from run_ashrae_benchmark import empirical_threshold, metrics

from industrial_novelty.data.ashrae import FAULTS, chiller_windows, normal_split


def main():
    root = Path("results/ashrae_directions_2026-11-04")
    normal = {
        w: chiller_windows("data/raw/ashrae_public/normal10.csv", window=w)
        for w in (12, 48)
    }
    faults = {
        (w, f, s): chiller_windows(f"data/raw/ashrae_public/{f}{s}.csv", window=w)
        for w in (12, 48)
        for f in FAULTS
        for s in (10, 20, 30, 40)
    }
    sources = {
        "raw_lw": root,
        "invariant_lw": root,
        "raw_cov_cv": root / "precision",
        "invariant_cov_cv": root / "precision",
        "raw_graphical_cv": root / "precision",
    }
    records = []
    for protocol in ("blocked", "loto40", "loto45", "loto50"):
        band = None if protocol == "blocked" else int(protocol[4:])
        for seed in range(5):
            parts = {
                w: normal_split(normal[w], seed, held_out_band=band)[0]
                for w in (12, 48)
            }
            _, c12, c48 = np.intersect1d(
                parts[12]["calibration"].meta.row_end,
                parts[48]["calibration"].meta.row_end,
                return_indices=True,
            )
            _, n12, n48 = np.intersect1d(
                parts[12]["test"].meta.row_end,
                parts[48]["test"].meta.row_end,
                return_indices=True,
            )
            for method, source in sources.items():
                score = {
                    w: np.load(
                        source / "scores" / f"w{w}_{protocol}_s{seed}_{method}.npz"
                    )
                    for w in (12, 48)
                }
                for fault in FAULTS:
                    for severity in (10, 20, 30, 40):
                        ends = {}
                        for w in (12, 48):
                            batch = faults[w, fault, severity]
                            keep = (
                                np.ones(len(batch.X), bool)
                                if band is None
                                else batch.meta.temperature_band.to_numpy() == band
                            )
                            ends[w] = batch.meta.loc[keep, "row_end"].to_numpy()
                        _, f12, f48 = np.intersect1d(
                            ends[12], ends[48], return_indices=True
                        )
                        for w, cal_index, normal_index, fault_index in [
                            (12, c12, n12, f12),
                            (48, c48, n48, f48),
                        ]:
                            threshold = empirical_threshold(
                                score[w]["calibration"][cal_index], 0.05
                            )
                            records.append(
                                {
                                    "window": w,
                                    "protocol": protocol,
                                    "seed": seed,
                                    "method": method,
                                    "fault": fault,
                                    "severity": severity // 10,
                                    "calibration_windows": len(cal_index),
                                    "normal_windows": len(normal_index),
                                    "fault_windows": len(fault_index),
                                    **metrics(
                                        score[w]["normal"][normal_index],
                                        score[w][f"{fault}{severity}"][fault_index],
                                        threshold,
                                    ),
                                }
                            )
                for archive in score.values():
                    archive.close()
    frame = pd.DataFrame(records)
    frame.to_csv(root / "matched_windows.csv", index=False)
    by_seed = (
        frame.groupby(["window", "protocol", "seed", "method", "severity"])[
            ["auroc", "recall", "normal_fpr"]
        ]
        .mean()
        .reset_index()
    )
    summary = by_seed.groupby(["window", "protocol", "method", "severity"])[
        ["auroc", "recall", "normal_fpr"]
    ].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(root / "matched_windows_summary.csv", index=False)
    print("Matched calibration/test endpoints analyzed.")


if __name__ == "__main__":
    main()
