"""Exploratory external validation on the two LBNL field RTU installations."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from run_lbnl_sdahu import daily_threshold, frame_day, maxima, unpack
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from industrial_novelty.data.lbnl import INTERVAL, occupied_windows
from industrial_novelty.models.lbnl import ResidualScan, windows
from industrial_novelty.models.subspace import CalibratedSubspace

SITES = {"Site1": "Staging_Fault", "Site2": "Undercharged40"}
TARGETS = ("sensible_proxy", "RTU_COMP_WATT", "lift_1", "lift_2", "discharge_asymmetry")
CONTEXT = ("RTU_OA_TEMP", "RTU_OA_HUM", "RTU_MA_TEMP", "RTU_RA_TEMP", "RTU_SA_FLOW")
WIDTH = 4
ALPHA = 0.20


def read_rtu(source, site, suffix):
    channels = (
        *CONTEXT,
        "RTU_SA_TEMP",
        "RTU_COMP_WATT",
        "RTU_SA_FAN_WATT",
        "RTU_REFG_DISC_PRES_1",
        "RTU_REFG_DISC_PRES_2",
        "RTU_REFG_SUCT_PRES_1",
        "RTU_REFG_SUCT_PRES_2",
        "RTU_REFG_DISC_TEMP_1",
        "RTU_REFG_DISC_TEMP_2",
    )
    with source.open(f"Field RTU/{site}_{suffix}.csv") as stream:
        data = pd.read_csv(
            stream, usecols=["Datetime", *channels], parse_dates=["Datetime"]
        )
    data = data.set_index("Datetime").replace([np.inf, -np.inf], np.nan)
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError(f"Invalid timestamps in {site} {suffix}.")
    data["SF_SPD_DM"] = data["RTU_SA_FAN_WATT"].where(
        data["RTU_SA_FAN_WATT"].isna(), data["RTU_SA_FAN_WATT"] > 100
    )
    data["SYS_CTL"] = 1
    data["sensible_proxy"] = data["RTU_SA_FLOW"] * (
        data["RTU_MA_TEMP"] - data["RTU_SA_TEMP"]
    )
    for circuit in (1, 2):
        data[f"lift_{circuit}"] = (
            data[f"RTU_REFG_DISC_PRES_{circuit}"]
            - data[f"RTU_REFG_SUCT_PRES_{circuit}"]
        )
    data["discharge_asymmetry"] = (
        data["RTU_REFG_DISC_TEMP_1"] - data["RTU_REFG_DISC_TEMP_2"]
    )
    buckets = data.resample(INTERVAL, label="right")
    frame = buckets.mean().loc[buckets.count().eq(15).all(axis=1)]
    return frame, occupied_windows(frame, WIDTH)


def patch_roles(frame, ends, origin, fold):
    phase = ((frame_day(frame.index[ends]) - origin).days // 4 - fold) % 5
    return [ends[phase >= 3], ends[phase == 2], ends[phase == 1], ends[phase == 0]]


def scores_for_site(normal, fault, ends, fault_ends, site):
    origin = min(frame_day(normal.index)[0], frame_day(fault.index)[0])
    columns = (
        *CONTEXT,
        "RTU_SA_TEMP",
        "RTU_COMP_WATT",
        "lift_1",
        "lift_2",
        "discharge_asymmetry",
    )
    rows = []
    for fold in range(5):
        fit, reference, calibration, test = patch_roles(normal, ends, origin, fold)
        fault_test = patch_roles(fault, fault_ends, origin, fold)[-1]
        train = normal.iloc[fit]
        scaler = StandardScaler().fit(train.loc[:, columns])

        def raw_patches(frame, ids, scaler=scaler):
            raw = scaler.transform(frame.loc[:, columns])
            return windows(raw, ids, WIDTH).reshape(len(ids), len(columns) * WIDTH)

        pca = CalibratedSubspace(max_rank=12, standardize=False).fit(
            raw_patches(normal, fit)
        )
        models = [
            HistGradientBoostingRegressor(
                max_iter=100,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                early_stopping=False,
            ).fit(train.loc[:, CONTEXT], train[name])
            for name in TARGETS
        ]

        def residuals(frame, ids, models=models):
            context = frame.loc[:, CONTEXT]
            predicted = np.column_stack([model.predict(context) for model in models])
            return windows(frame.loc[:, TARGETS].to_numpy() - predicted, ids, WIDTH)

        def modes(frame, ids):
            return np.searchsorted([20.0, 27.0], frame["RTU_OA_TEMP"].to_numpy()[ids])

        scanner = ResidualScan((1000, 100, 0.15, 0.15, 0.5)).fit(
            residuals(normal, reference), modes(normal, reference)
        )

        scanner_15 = ResidualScan(scanner.floors, conditional=False, scales=(1,)).fit(
            residuals(normal, reference), modes(normal, reference)
        )

        def day_scores(frame, ids, scanner=scanner, scanner_15=scanner_15, pca=pca):
            errors, bands = residuals(frame, ids), modes(frame, ids)
            point = {
                "pca": pca.score(raw_patches(frame, ids)),
                "virtual_15min": scanner_15.score(errors, bands)["diagonal"],
                "hybrid_scan": scanner.score(errors, bands)["scan"],
            }
            return maxima(pd.DataFrame(point, index=frame.index[ids]))

        cal = day_scores(normal, calibration)
        thresholds = cal.apply(daily_threshold, alpha=ALPHA)
        for label, frame, ids in ((0, normal, test), (1, fault, fault_test)):
            daily = day_scores(frame, ids)
            for method, threshold in thresholds.items():
                rows.append(
                    pd.DataFrame(
                        {
                            "site": site,
                            "fold": fold,
                            "method": method,
                            "label": label,
                            "day": daily.index,
                            "score": daily[method].to_numpy(),
                            "threshold": threshold,
                            "alarm": (daily[method] > threshold).to_numpy(),
                            "calibration_days": len(cal),
                        }
                    )
                )
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4), ZipFile(unpack(args.archive)) as source:
        rows = []
        for site, suffix in SITES.items():
            normal, ends = read_rtu(source, site, "Unfaulted")
            fault, fault_ends = read_rtu(source, site, suffix)
            if len(frame_day(normal.index).intersection(frame_day(fault.index))):
                raise ValueError(f"Overlapping normal and fault days at {site}.")
            rows.append(scores_for_site(normal, fault, ends, fault_ends, site))
    scores = pd.concat(rows, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output / "field_days.csv", index=False)
    summary = (
        scores.groupby(["site", "method", "label"])["alarm"]
        .agg(["mean", "size"])
        .unstack()
    )
    summary.columns = [
        "normal_day_fpr",
        "fault_day_recall",
        "normal_days",
        "fault_days",
    ]
    comparable = scores.groupby(["site", "method", "fold"]).filter(
        lambda x: x["label"].nunique() == 2
    )
    aucs = [
        {
            "site": site,
            "method": method,
            "auc": roc_auc_score(part["label"], part["score"]),
        }
        for (site, method, _), part in comparable.groupby(["site", "method", "fold"])
    ]
    summary["auroc_mean_comparable_folds"] = (
        pd.DataFrame(aucs).groupby(["site", "method"])["auc"].mean()
    )
    summary.to_csv(args.output / "field_summary.csv")
    with args.archive.open("rb") as raw:
        digest = hashlib.file_digest(raw, "sha256").hexdigest()
    (args.output / "archive_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
