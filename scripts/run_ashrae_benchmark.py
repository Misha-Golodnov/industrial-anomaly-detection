"""Normal-only exploratory benchmark on the audited S-DAGMM CSV derivative.

This is not an independent-run SOTA benchmark. See the research report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from threadpoolctl import threadpool_limits

from industrial_novelty.data.ashrae import (
    CHANNELS,
    CONTEXT,
    FAULTS,
    RESPONSES,
    chiller_windows,
    normal_split,
)
from industrial_novelty.models.canonical import CanonicalVariateIndex
from industrial_novelty.models.chiller import ConditionalRBF, TorchMeanDetector
from industrial_novelty.models.neural import NeuralTransformationIndex
from industrial_novelty.models.subspace import WindowSubspaceIndex


class MeanBaseline:
    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed

    def fit(self, X):
        y = X.mean(axis=1).astype(float)
        d = y.shape[1]
        if self.kind == "mahalanobis":
            self.model = LedoitWolf().fit(y)
        elif self.kind == "iforest":
            self.model = IsolationForest(
                n_estimators=200, random_state=self.seed, n_jobs=2
            ).fit(y)
        elif self.kind == "ocsvm":
            self.model = OneClassSVM(nu=0.05, gamma=1 / d).fit(y)
        elif self.kind == "lof":
            self.model = LocalOutlierFactor(
                n_neighbors=min(20, len(y) - 1), novelty=True
            ).fit(y)
        elif self.kind == "kpca":
            self.train_ = y
            self.model = KernelPCA(
                n_components=min(16, len(y) - 1), kernel="rbf", gamma=1 / d
            ).fit(y)
            self.kernel_mean_ = rbf_kernel(y, y, gamma=1 / d).mean()
        else:
            raise ValueError(self.kind)
        return self

    def score(self, X):
        y = X.mean(axis=1).astype(float)
        if self.kind == "mahalanobis":
            return self.model.mahalanobis(y)
        if self.kind == "kpca":
            centered_norm = (
                1
                - 2 * rbf_kernel(y, self.train_, gamma=1 / y.shape[1]).mean(axis=1)
                + self.kernel_mean_
            )
            return np.maximum(
                centered_norm - np.sum(self.model.transform(y) ** 2, axis=1), 0
            )
        return -self.model.score_samples(y)


def factories(seed, epochs, neural):
    candidates = {
        name: (lambda name=name: MeanBaseline(name, seed))
        for name in ("mahalanobis", "iforest", "ocsvm", "lof", "kpca")
    }
    candidates.update(
        {
            "dpca": lambda: WindowSubspaceIndex(variance=0.95, max_rank=32, seed=seed),
            "cva": lambda: CanonicalVariateIndex(max_block_dim=64, seed=seed),
            "conditional_rbf": lambda: ConditionalRBF(seed=seed),
            "conditional_tangent": lambda: ConditionalRBF(
                seed=seed, parallel_weight=0.1
            ),
        }
    )
    if neural:
        candidates.update(
            {
                "ae_mean": lambda: TorchMeanDetector(
                    kind="ae", epochs=epochs, seed=seed
                ),
                "conditional_flow": lambda: TorchMeanDetector(
                    kind="flow", epochs=epochs, seed=seed
                ),
                "neutral_mlp": lambda: NeuralTransformationIndex(
                    epochs=epochs, seed=seed, hidden=64, latent=16
                ),
            }
        )
    return candidates


def empirical_threshold(scores, alpha):
    """Finite-sample rank rule. No exchangeability claim for these time blocks."""
    scores = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid calibration scores/alpha.")
    rank = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return (
        float(np.partition(scores, rank - 1)[rank - 1])
        if rank <= len(scores)
        else float("inf")
    )


def metrics(normal, faulty, threshold):
    y = np.r_[np.zeros(len(normal)), np.ones(len(faulty))]
    s = np.r_[normal, faulty]
    # Fix the class prior for AP comparisons; raw dataset prevalence is arbitrary.
    weights = np.r_[
        np.full(len(normal), 0.5 / len(normal)), np.full(len(faulty), 0.5 / len(faulty))
    ]
    return {
        "auroc": roc_auc_score(y, s),
        "balanced_ap": average_precision_score(y, s, sample_weight=weights),
        "recall": float(np.mean(faulty > threshold)),
        "normal_fpr": float(np.mean(normal > threshold)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/ashrae_public"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/ashrae_2026-11-04")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--no-neural", action="store_true")
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["blocked", "loto40", "loto45", "loto50"],
        choices=["blocked", "loto40", "loto45", "loto50"],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.no_neural:
        import torch

        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
    normal = chiller_windows(args.data / "normal10.csv")
    plans = []
    for protocol in args.protocols:
        band = None if protocol == "blocked" else int(protocol[4:])
        for seed in args.seeds:
            parts, assignments = normal_split(normal, seed, held_out_band=band)
            plans.append(
                {
                    "protocol": protocol,
                    "seed": seed,
                    "held_out_band": band,
                    "blocks": assignments,
                    "windows": {k: len(v.X) for k, v in parts.items()},
                }
            )
    manifest = {
        "status": "exploratory; one normal trajectory; no independent-run uncertainty or SOTA claim",
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "schema": {"context": CONTEXT, "responses": RESPONSES},
        "window": 12,
        "stride": 6,
        "block_size": 180,
        "edge_embargo": 12,
        "methods": list(factories(0, args.epochs, not args.no_neural)),
        "threshold_budgets": [0.01, 0.05],
        "threshold_rule": "score > finite-sample order statistic",
        "tuning": "none; fixed a priori hyperparameters; no fault selection",
        "plans": plans,
        "python": platform.python_version(),
        "versions": {
            x: importlib.metadata.version(x)
            for x in ("numpy", "pandas", "scikit-learn", "scipy")
        },
        "normal_source_sha256": hashlib.sha256(
            (args.data / "normal10.csv").read_bytes()
        ).hexdigest(),
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                Path("src/industrial_novelty/data/ashrae.py"),
                Path("src/industrial_novelty/models/chiller.py"),
            ]
        },
    }
    # Freeze protocol before opening any faulty files.
    (args.output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    faults = {
        (fault, level): chiller_windows(args.data / f"{fault}{level}.csv")
        for fault in FAULTS
        for level in (10, 20, 30, 40)
    }
    results, normal_results, training = [], [], []
    for plan in plans:
        seed, protocol, band = plan["seed"], plan["protocol"], plan["held_out_band"]
        parts, _ = normal_split(normal, seed, held_out_band=band)
        fit = parts["fit"].X
        mean, scale = (
            fit.reshape(-1, len(CHANNELS)).mean(axis=0),
            fit.reshape(-1, len(CHANNELS)).std(axis=0),
        )
        scale = np.maximum(scale, 1e-6)
        z = {
            role: ((part.X - mean) / scale).astype(np.float32)
            for role, part in parts.items()
        }
        for name, factory in factories(seed, args.epochs, not args.no_neural).items():
            start = time.perf_counter()
            model = factory()
            with threadpool_limits(limits=2):
                if isinstance(model, ConditionalRBF):
                    model.fit(z["fit"], z["reference"])
                else:
                    model.fit(z["fit"])
                cal_scores, normal_scores = (
                    model.score(z["calibration"]),
                    model.score(z["test"]),
                )
                training.append(
                    {
                        "protocol": protocol,
                        "seed": seed,
                        "method": name,
                        "fit_and_normal_score_seconds": time.perf_counter() - start,
                        "first_loss": getattr(model, "losses_", [None])[0],
                        "last_loss": getattr(model, "losses_", [None])[-1],
                    }
                )
                thresholds = {
                    alpha: empirical_threshold(cal_scores, alpha)
                    for alpha in (0.01, 0.05)
                }
                for alpha, threshold in thresholds.items():
                    normal_results.append(
                        {
                            "protocol": protocol,
                            "seed": seed,
                            "method": name,
                            "alpha": alpha,
                            "threshold": threshold,
                            "normal_windows": len(normal_scores),
                            "normal_fpr": float(np.mean(normal_scores > threshold)),
                        }
                    )
                for (fault, level), batch in faults.items():
                    mask = (
                        np.ones(len(batch.X), dtype=bool)
                        if band is None
                        else batch.meta.temperature_band.to_numpy() == band
                    )
                    if not mask.any():
                        raise ValueError(
                            f"No fault test windows for {fault}{level}, {protocol}."
                        )
                    scores = model.score(
                        ((batch.X[mask] - mean) / scale).astype(np.float32)
                    )
                    if not np.isfinite(scores).all():
                        raise ValueError(f"Nonfinite score: {name}.")
                    for alpha, threshold in thresholds.items():
                        results.append(
                            {
                                "protocol": protocol,
                                "seed": seed,
                                "method": name,
                                "fault": fault,
                                "severity": level // 10,
                                "alpha": alpha,
                                "threshold": threshold,
                                "normal_windows": len(normal_scores),
                                "fault_windows": len(scores),
                                **metrics(normal_scores, scores, threshold),
                            }
                        )
            print(
                f"{protocol} seed={seed} {name}: {time.perf_counter() - start:.2f}s",
                flush=True,
            )
        pd.DataFrame(results).to_csv(args.output / "per_fault.csv", index=False)
        pd.DataFrame(normal_results).to_csv(
            args.output / "normal_metrics.csv", index=False
        )
        pd.DataFrame(training).to_csv(args.output / "training.csv", index=False)
    frame = pd.DataFrame(results)
    # First average across fault categories, then summarize split/initialization
    # variability. These are not confidence intervals or independent repetitions.
    by_seed = (
        frame.groupby(["protocol", "seed", "method", "severity", "alpha"])[
            ["auroc", "balanced_ap", "recall", "normal_fpr"]
        ]
        .mean()
        .reset_index()
    )
    by_seed.to_csv(args.output / "per_seed.csv", index=False)
    summary = (
        by_seed.groupby(["protocol", "method", "severity", "alpha"])
        .agg(
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            balanced_ap_mean=("balanced_ap", "mean"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            normal_fpr_mean=("normal_fpr", "mean"),
            normal_fpr_std=("normal_fpr", "std"),
            splits=("seed", "size"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    print("Completed exploratory benchmark. No SOTA claim.")


if __name__ == "__main__":
    main()
