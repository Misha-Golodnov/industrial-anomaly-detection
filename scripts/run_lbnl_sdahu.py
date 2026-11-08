"""Run the LBNL SDAHU audit, normal-only detection and held-severity diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import sklearn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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
from industrial_novelty.models.lbnl import (
    PHYSICS_FLOORS,
    REGRESSION_FLOORS,
    AHUResiduals,
    ResidualScan,
    regime,
    windows,
)
from industrial_novelty.models.lbnl_patch import PatchMemory
from industrial_novelty.models.neural import NeuralTransformationIndex, _torch
from industrial_novelty.models.subspace import PrincipalSubspace, tail_scale

WIDTH = 4
ALPHA = 0.10
SEED = 1043
MIN_WINDOWS = 8
LOW_LEVEL = {"sa_bias": 2, "coil_stuck": 10, "damper_stuck": 10}


def frame_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return (index - INTERVAL).normalize()


def daily_threshold(maxima: pd.Series, alpha: float = ALPHA) -> float:
    rank = math.ceil((len(maxima) + 1) * (1 - alpha))
    if rank > len(maxima):
        raise ValueError(f"Only {len(maxima)} days to calibrate alpha={alpha}.")
    return float(np.partition(maxima.to_numpy(), rank - 1)[rank - 1])


def unpack(archive: Path) -> Path:
    with ZipFile(archive) as source:
        nested = [name for name in source.namelist() if name.lower().endswith(".zip")]
        if not nested:
            return archive
        if len(nested) != 1:
            raise ValueError(f"Expected one nested SDAHU archive, got {nested}.")
        target = archive.with_name(archive.stem + "-inner.zip")
        partial = target.with_suffix(".part")
        with source.open(nested[0]) as raw, partial.open("wb") as out:
            shutil.copyfileobj(raw, out, length=1024 * 1024)
        partial.replace(target)
        return target


def residual_arrays(residuals, frame, ends, *, feedback=False):
    regression = residuals.regression(frame)
    physics = residuals.physics(frame)
    return {
        "regression": windows(regression, ends, WIDTH),
        "physics": windows(physics[:, : len(PHYSICS_FLOORS)], ends, WIDTH),
        "hybrid": windows(
            np.column_stack([regression, physics[:, : len(PHYSICS_FLOORS)]]),
            ends,
            WIDTH,
        ),
        **(
            {"feedback": windows(np.column_stack([regression, physics]), ends, WIDTH)}
            if feedback
            else {}
        ),
    }


class DetectorSuite:
    def __init__(self, normal, ends, fit, ref, neural):
        self.neural = neural
        raw = normal.loc[:, BASIC].to_numpy()
        self.scaler = StandardScaler().fit(raw[ends[fit]])
        fit_windows = windows(self.scaler.transform(raw), ends[fit], WIDTH)
        reference = windows(self.scaler.transform(raw), ends[ref], WIDTH)
        flat = fit_windows.reshape(len(fit_windows), -1)
        self.forest = IsolationForest(n_estimators=200, random_state=SEED).fit(flat)
        rng = np.random.default_rng(SEED)
        bank = rng.choice(len(flat), size=min(4096, len(flat)), replace=False)
        self.knn = NearestNeighbors(n_neighbors=3, algorithm="brute").fit(flat[bank])
        self.pca = PrincipalSubspace(max_rank=12, standardize=False).fit(flat)
        self.pca_scale = np.array(
            [
                tail_scale(score, 0.01)
                for score in self.pca.pair(reference.reshape(len(reference), -1))
            ]
        )
        if neural:
            self.neutral = NeuralTransformationIndex(epochs=12, seed=SEED)
            self.neutral.fit_windows(fit_windows)
            self.patch = PatchMemory(seed=SEED).fit(
                fit_windows, normal.index[ends[fit]]
            )
        self.residuals = AHUResiduals(feedback=True).fit(normal.iloc[ends[fit]])
        ref_frame, ref_ends = normal, ends[ref]
        ref_arrays = residual_arrays(self.residuals, ref_frame, ref_ends, feedback=True)
        bands = regime(normal)[ref_ends]
        hybrid_floors = (*REGRESSION_FLOORS, *PHYSICS_FLOORS)
        configs = {
            "virtual": ("regression", REGRESSION_FLOORS, False, (1,)),
            "physics": ("physics", PHYSICS_FLOORS, True, (1, 2, 4)),
            "hybrid_15min": ("hybrid", hybrid_floors, True, (1,)),
            "hybrid_global": ("hybrid", hybrid_floors, False, (1, 2, 4)),
            "hybrid": ("hybrid", hybrid_floors, True, (1, 2, 4)),
            "hybrid_feedback": (
                "feedback",
                (*hybrid_floors, 0.01, 0.01, 0.01),
                True,
                (1, 2, 4),
            ),
        }
        self.scans = {
            name: (
                key,
                ResidualScan(floors, conditional=conditional, scales=scales).fit(
                    ref_arrays[key], bands
                ),
            )
            for name, (key, floors, conditional, scales) in configs.items()
        }

    def score(self, frame, ends):
        patches = windows(
            self.scaler.transform(frame.loc[:, BASIC].to_numpy()), ends, WIDTH
        )
        flat = patches.reshape(len(patches), -1)
        scores = {
            "pca": (np.column_stack(self.pca.pair(flat)) / self.pca_scale).sum(axis=1),
            "isolation_forest": -self.forest.score_samples(flat),
            "patch_knn": self.knn.kneighbors(flat)[0].mean(axis=1),
        }
        if self.neural:
            scores["neutral_ad"] = self.neutral.anomaly_score(patches)
            scores["paano_causal"] = self.patch.score(patches)
        arrays = residual_arrays(self.residuals, frame, ends, feedback=True)
        bands = regime(frame)[ends]
        for name, (key, scanner) in self.scans.items():
            evidence = scanner.score(arrays[key], bands)
            scores[name] = (
                evidence["diagonal"] if name == "virtual" else evidence["scan"]
            )
            if name == "hybrid":
                for component in ("dense", "sparse", "diagonal"):
                    scores[f"hybrid_{component}"] = evidence[component]
        scores["command_feedback"] = abs(arrays["feedback"][:, -1, -3:]).max(axis=1)
        result = pd.DataFrame(scores, index=frame.index[ends])
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError("Nonfinite detection score.")
        return result

    def diagnosis_features(self, frame, ends):
        raw = windows(
            self.scaler.transform(frame.loc[:, BASIC].to_numpy()), ends, WIDTH
        )
        res = residual_arrays(self.residuals, frame, ends)["hybrid"]
        return np.column_stack(
            [raw.mean(axis=1), raw[:, -1] - raw[:, 0], res.mean(axis=1)]
        )


def maxima(frame):
    groups = frame.groupby(frame_day(frame.index))
    return groups.max().loc[groups.size() >= MIN_WINDOWS]


def paired_maxima(normal, fault):
    common = normal.index.intersection(fault.index)
    return maxima(normal.loc[common]), maxima(fault.loc[common])


def held_severity(suite, frames, endpoints, origin, fold, deduplicated):
    classes = sorted({"normal", *LOW_LEVEL} - deduplicated)
    fit, test, test_meta = [], [], []
    for name, frame in frames.items():
        family, level = SCENARIOS[name]
        if family not in classes:
            continue
        ends = endpoints[name]
        index = frame.index[ends]
        roles = fold_roles(week_index(index, origin), fold)
        if family == "normal" or abs(level) > LOW_LEVEL[family]:
            chosen = ends[roles["fit"]]
            fit.append(
                (
                    suite.diagnosis_features(frame, chosen),
                    np.repeat(family, len(chosen)),
                )
            )
        if family == "normal" or abs(level) == LOW_LEVEL[family]:
            chosen = ends[roles["test"]]
            test.append(suite.diagnosis_features(frame, chosen))
            test_meta.extend(
                zip(np.repeat(family, len(chosen)), frame_day(frame.index[chosen]))
            )
    x, y = (np.concatenate(parts) for parts in zip(*fit))
    features = np.concatenate(test)
    model_set = {
        "shrinkage_lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150,
            max_leaf_nodes=23,
            class_weight="balanced",
            early_stopping=False,
        ),
    }
    truth = pd.DataFrame(test_meta, columns=["truth", "day"])
    predictions = []
    for method, model in model_set.items():
        model.fit(x, y)
        probabilities = pd.DataFrame(
            model.predict_proba(features), columns=model.classes_
        )
        # Pool patches only within a scenario; signed biases at the same day are separate.
        offset = 0
        pieces = []
        for piece in test:
            selection = slice(offset, offset + len(piece))
            part = probabilities.iloc[selection]
            keys = truth.iloc[selection]
            pieces.append(
                part.groupby([keys["truth"], keys["day"]]).mean().idxmax(axis=1)
            )
            offset += len(piece)
        predictions.append(
            pd.concat(pieces)
            .rename("prediction")
            .reset_index()
            .assign(fold=fold, method=method)
        )
    return pd.concat(predictions, ignore_index=True)


def summarize(pairs, normal, out):
    rows = []
    for (method, case), part in pairs.groupby(["method", "scenario"]):
        # Scores across folds can have different scales. Average within-fold rankings.
        rankings = []
        for _, fold in part.groupby("fold"):
            scores = np.r_[fold["normal_score"], fold["fault_score"]]
            y = np.r_[np.zeros(len(fold)), np.ones(len(fold))]
            rankings.append(
                (
                    len(fold),
                    roc_auc_score(y, scores),
                    average_precision_score(y, scores),
                )
            )
        weights, aucs, aps = np.asarray(rankings).T
        difference = part["fault_score"] - part["normal_score"]
        rows.append(
            {
                "method": method,
                "scenario": case,
                "family": SCENARIOS[case][0],
                "days": len(part),
                "auroc": np.average(aucs, weights=weights),
                "average_precision_balanced": np.average(aps, weights=weights),
                "paired_win": (difference.gt(0) + 0.5 * difference.eq(0)).mean(),
                "fault_day_recall": part["fault_alarm"].mean(),
                "matched_normal_day_fpr": part["normal_alarm"].mean(),
            }
        )
    per_case = pd.DataFrame(rows)
    per_case.to_csv(out / "per_scenario.csv", index=False)
    metrics = [
        "auroc",
        "average_precision_balanced",
        "paired_win",
        "fault_day_recall",
        "matched_normal_day_fpr",
    ]
    family = per_case.groupby(["method", "family"])[metrics].mean()
    family.to_csv(out / "per_family.csv")
    summary = family.groupby("method").mean()
    summary["normal_day_fpr"] = normal.groupby("method")["alarm"].mean()
    summary.sort_values("fault_day_recall", ascending=False).to_csv(out / "summary.csv")
    return summary


def bootstrap(pairs, origin, out):
    """Paired week bootstrap of fault-day recall differences, fixed fitted models."""
    long = pairs.assign(week=(pd.to_datetime(pairs["day"]) - origin).dt.days // 7)
    methods = sorted(long["method"].unique())
    counts = long.groupby(["week", "scenario", "method"])["fault_alarm"].agg(
        ["sum", "size"]
    )
    weeks = sorted(long["week"].unique())
    cases = sorted(long["scenario"].unique())
    counts = counts.reindex(
        pd.MultiIndex.from_product([weeks, cases, methods]), fill_value=0
    )
    hits, totals = (
        counts[column].to_numpy().reshape(len(weeks), len(cases) * len(methods))
        for column in ("sum", "size")
    )
    draws = np.random.default_rng(SEED).multinomial(
        len(weeks), [1 / len(weeks)] * len(weeks), 2000
    )
    denom = draws @ totals
    valid = (denom > 0).all(axis=1)
    samples = ((draws[valid] @ hits) / denom[valid]).reshape(
        -1, len(cases), len(methods)
    )
    groups = np.asarray([SCENARIOS[case][0] for case in cases])
    samples = np.stack(
        [samples[:, groups == group].mean(axis=1) for group in np.unique(groups)]
    ).mean(axis=0)
    delta = samples - samples[:, [methods.index("hybrid")]]
    limits = np.quantile(delta, [0.025, 0.975], axis=0).T
    pd.DataFrame(
        limits,
        index=pd.Index(methods, name="method"),
        columns=["delta_vs_hybrid_q025", "delta_vs_hybrid_q975"],
    ).to_csv(out / "weekly_bootstrap.csv")


def run(args):
    start = time.monotonic()
    archive = unpack(args.archive)
    frames, inventory = read_sdahu(archive)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(inventory).to_csv(out / "inventory.csv", index=False)
    deduplicated = {
        row["family"] for row in inventory if row["canonical"] != row["scenario"]
    }
    endpoints = {name: occupied_windows(frame, WIDTH) for name, frame in frames.items()}
    normal = frames[NORMAL]
    origin = frame_day(normal.index)[0]
    pairs, normals, calibration, diagnoses = [], [], [], []
    print(
        f"Validated {len(inventory)} CSVs, {len(frames)} unique trajectories",
        flush=True,
    )
    for fold in range(4):
        print(f"Fold {fold + 1}/4 ({time.monotonic() - start:.1f}s)", flush=True)
        ends = endpoints[NORMAL]
        roles = fold_roles(week_index(normal.index[ends], origin), fold)
        suite = DetectorSuite(
            normal, ends, roles["fit"], roles["reference"], not args.no_neural
        )
        cal_scores = maxima(suite.score(normal, ends[roles["calibration"]]))
        thresholds = cal_scores.apply(daily_threshold)
        calibration.extend(
            {
                "fold": fold,
                "method": name,
                "threshold": score,
                "normal_calibration_days": len(cal_scores),
            }
            for name, score in thresholds.items()
        )
        normal_scores = suite.score(normal, ends[roles["test"]])
        daily_normal = maxima(normal_scores)
        normals.append(
            daily_normal.rename_axis("day")
            .stack()
            .rename("score")
            .reset_index(name="score")
            .rename(columns={"level_1": "method"})
            .assign(fold=fold)
        )
        for name, frame in frames.items():
            if name == NORMAL:
                continue
            ends = endpoints[name]
            role = fold_roles(week_index(frame.index[ends], origin), fold)["test"]
            scored = suite.score(frame, ends[role])
            matched, faulty = paired_maxima(normal_scores, scored)
            for method, threshold in thresholds.items():
                pairs.append(
                    pd.DataFrame(
                        {
                            "fold": fold,
                            "scenario": name,
                            "method": method,
                            "day": matched.index,
                            "normal_score": matched[method].to_numpy(),
                            "fault_score": faulty[method].to_numpy(),
                            "normal_alarm": (matched[method] > threshold).to_numpy(),
                            "fault_alarm": (faulty[method] > threshold).to_numpy(),
                        }
                    )
                )
        diagnoses.append(
            held_severity(suite, frames, endpoints, origin, fold, deduplicated)
        )
    paired = pd.concat(pairs, ignore_index=True)
    cal = pd.DataFrame(calibration)
    normal = pd.concat(normals, ignore_index=True).merge(cal, on=["fold", "method"])
    normal["alarm"] = normal["score"] > normal["threshold"]
    normal.to_csv(out / "normal_days.csv", index=False)
    paired.to_csv(
        out / "paired_days.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    cal.to_csv(out / "calibration.csv", index=False)
    predictions = pd.concat(diagnoses, ignore_index=True)
    confusion = predictions.groupby(["method", "truth", "prediction"]).size()
    confusion.rename("days").to_csv(out / "diagnosis_confusion.csv")
    predictions.assign(
        correct=predictions["truth"].eq(predictions["prediction"])
    ).groupby(["method", "truth"])["correct"].agg(["mean", "size"]).to_csv(
        out / "diagnosis_recall.csv"
    )
    summary = summarize(paired, normal, out)
    bootstrap(paired, origin, out)
    with args.archive.open("rb") as raw:
        sha256 = hashlib.file_digest(raw, "sha256").hexdigest()
    manifest = {
        "source": "https://doi.org/10.6084/m9.figshare.22338283.v1",
        "archive_sha256": sha256,
        "basic_channels": BASIC,
        "alpha": ALPHA,
        "seed": SEED,
        "interval_minutes": 15,
        "width_intervals": WIDTH,
        "minimum_windows_day": MIN_WINDOWS,
        "week_origin": str(origin.date()),
        "neural": not args.no_neural,
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": None if args.no_neural else _torch().__version__,
        "seconds": time.monotonic() - start,
        "rows": {name: len(frame) for name, frame in frames.items()},
        "eligible_windows": {name: len(indices) for name, indices in endpoints.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-neural", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if not args.no_neural:
            _torch().set_num_threads(4)
        run(args)


if __name__ == "__main__":
    main()
