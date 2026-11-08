"""Separate eight-class diagnosis track: seen SL1 and SL2-4 -> unseen SL1.

Same sensor allowlist, blocked holdouts and source-only scaling as novelty track.
No fault labels enter that separate novelty experiment.
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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from industrial_novelty.data.ashrae import (
    CHANNELS,
    FAULTS,
    chiller_windows,
    normal_split,
)

LABELS = ["normal", *FAULTS]


def features(X):
    return np.column_stack([X.mean(axis=1), X.std(axis=1)])


class TemporalClassifier:
    """Small supervised temporal CNN; completed windows, no future outside them."""

    def __init__(self, seed, epochs):
        self.seed, self.epochs = seed, epochs

    def fit(self, X, y):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        self.network = nn.Sequential(
            nn.Conv1d(len(CHANNELS), 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, dilation=4, padding=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(32, 8),
        )
        self.network.train()
        optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=1e-3, weight_decay=1e-4
        )
        counts = np.bincount(y, minlength=8)
        weights = torch.tensor(len(y) / (8 * counts), dtype=torch.float32)
        criterion = nn.CrossEntropyLoss(weight=weights)
        tensor = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
        target = torch.tensor(y, dtype=torch.long)
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            for indices in torch.randperm(len(y), generator=generator).split(128):
                optimizer.zero_grad()
                loss = criterion(self.network(tensor[indices]), target[indices])
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 5)
                optimizer.step()
        self.network.eval()
        return self

    def predict(self, X):
        import torch

        with torch.no_grad():
            tensor = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32)
            return np.concatenate(
                [self.network(b).argmax(dim=1).numpy() for b in tensor.split(1024)]
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/ashrae_public"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/ashrae_2026-11-04/supervised")
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--protocols", nargs="+", default=["blocked", "loto40", "loto45", "loto50"]
    )
    args = parser.parse_args()
    import torch

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    methods = [
        "logistic",
        "rbf_svm",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "temporal_cnn",
    ]
    manifest = {
        "track": "supervised eight-class diagnosis, separate from normal-only novelty",
        "status": "exploratory; one normal trajectory; seed SD is not an independent-run CI",
        "classes": LABELS,
        "methods": methods,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "scenarios": {
            "seen_sl1": "train SL1 fit blocks; test SL1 held-out blocks",
            "transfer_sl1": "train SL2-4 fit blocks; test SL1 held-out blocks; normal fit/test unchanged",
        },
        "protocols": args.protocols,
        "channels": CHANNELS,
        "features": "mean+std; temporal CNN uses full windows; scalers fitted on supervised training only",
        "tuning": "fixed hyperparameters; no test-based selection or oversampling",
        "source_manifest": "../data_manifest.json",
        "sklearn": importlib.metadata.version("scikit-learn"),
        "torch": torch.__version__,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    # Register the experiment before reading labeled fault data.
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    normal = chiller_windows(args.data / "normal10.csv")
    faults = {
        (f, level): chiller_windows(args.data / f"{f}{level}.csv")
        for f in FAULTS
        for level in (10, 20, 30, 40)
    }
    records, class_records, split_records = [], [], []
    for protocol in args.protocols:
        band = None if protocol == "blocked" else int(protocol[4:])
        for seed in args.seeds:
            normal_parts, _ = normal_split(normal, seed, held_out_band=band)
            fault_parts = {
                key: normal_split(batch, seed, held_out_band=band)[0]
                for key, batch in faults.items()
            }
            for scenario in ("seen_sl1", "transfer_sl1"):
                train_x, train_y = (
                    [normal_parts["fit"].X],
                    [np.zeros(len(normal_parts["fit"].X), dtype=int)],
                )
                test_x, test_y = (
                    [normal_parts["test"].X],
                    [np.zeros(len(normal_parts["test"].X), dtype=int)],
                )
                for label, fault in enumerate(FAULTS, start=1):
                    for level in [10] if scenario == "seen_sl1" else [20, 30, 40]:
                        part = fault_parts[fault, level]["fit"]
                        train_x.append(part.X)
                        train_y.append(np.full(len(part.X), label))
                        split_records.append(
                            {
                                "protocol": protocol,
                                "seed": seed,
                                "scenario": scenario,
                                "file": f"{fault}{level}.csv",
                                "fit_blocks": sorted(part.meta.block.unique().tolist()),
                            }
                        )
                    part = fault_parts[fault, 10]["test"]
                    test_x.append(part.X)
                    test_y.append(np.full(len(part.X), label))
                X, y, test, truth = (
                    np.concatenate(train_x),
                    np.concatenate(train_y),
                    np.concatenate(test_x),
                    np.concatenate(test_y),
                )
                mean = X.reshape(-1, len(CHANNELS)).mean(axis=0)
                scale = np.maximum(X.reshape(-1, len(CHANNELS)).std(axis=0), 1e-6)
                X, test = (
                    ((X - mean) / scale).astype(np.float32),
                    ((test - mean) / scale).astype(np.float32),
                )
                scaler = StandardScaler().fit(features(X))
                tabular, test_tabular = (
                    scaler.transform(features(X)),
                    scaler.transform(features(test)),
                )
                sample_weight = compute_sample_weight("balanced", y)
                candidates = {
                    "logistic": LogisticRegression(
                        C=1, max_iter=1000, class_weight="balanced"
                    ),
                    "rbf_svm": SVC(C=10, gamma="scale", class_weight="balanced"),
                    "random_forest": RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=2,
                    ),
                    "extra_trees": ExtraTreesClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=2,
                    ),
                    "hist_gradient_boosting": HistGradientBoostingClassifier(
                        max_iter=100,
                        max_leaf_nodes=15,
                        l2_regularization=1,
                        random_state=seed,
                    ),
                    "temporal_cnn": TemporalClassifier(seed, args.epochs),
                }
                for name, model in candidates.items():
                    started = time.perf_counter()
                    with threadpool_limits(limits=2):
                        if name == "temporal_cnn":
                            prediction = model.fit(X, y).predict(test)
                        elif name == "hist_gradient_boosting":
                            prediction = model.fit(
                                tabular, y, sample_weight=sample_weight
                            ).predict(test_tabular)
                        else:
                            prediction = model.fit(tabular, y).predict(test_tabular)
                    cm = confusion_matrix(truth, prediction, labels=np.arange(8))
                    recalls = recall_score(
                        truth,
                        prediction,
                        labels=np.arange(8),
                        average=None,
                        zero_division=0,
                    )
                    # Each fault is equally weighted; normal FPR uses its own class.
                    fpr = float(np.mean(prediction[truth == 0] != 0))
                    missed = float(
                        np.mean(
                            [np.mean(prediction[truth == k] == 0) for k in range(1, 8)]
                        )
                    )
                    records.append(
                        {
                            "protocol": protocol,
                            "seed": seed,
                            "scenario": scenario,
                            "method": name,
                            "macro_f1": f1_score(
                                truth,
                                prediction,
                                labels=np.arange(8),
                                average="macro",
                                zero_division=0,
                            ),
                            "balanced_accuracy": balanced_accuracy_score(
                                truth, prediction
                            ),
                            "normal_fpr": fpr,
                            "fault_detection_recall": 1 - missed,
                            "train_windows": len(y),
                            "test_windows": len(truth),
                            "fit_and_predict_seconds": time.perf_counter() - started,
                        }
                    )
                    for label, recall in enumerate(recalls):
                        class_records.append(
                            {
                                "protocol": protocol,
                                "seed": seed,
                                "scenario": scenario,
                                "method": name,
                                "class": LABELS[label],
                                "recall": recall,
                                "confusion_row": json.dumps(cm[label].tolist()),
                            }
                        )
                    print(
                        f"{protocol} seed={seed} {scenario} {name}: {records[-1]['macro_f1']:.3f}",
                        flush=True,
                    )
                pd.DataFrame(records).to_csv(args.output / "per_seed.csv", index=False)
                pd.DataFrame(class_records).to_csv(
                    args.output / "per_class.csv", index=False
                )
                (args.output / "split_blocks.json").write_text(
                    json.dumps(split_records, indent=2), encoding="utf-8"
                )
    frame = pd.DataFrame(records)
    summary = (
        frame.groupby(["protocol", "scenario", "method"])
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            normal_fpr_mean=("normal_fpr", "mean"),
            fault_detection_recall_mean=("fault_detection_recall", "mean"),
            seeds=("seed", "size"),
        )
        .reset_index()
    )
    summary.to_csv(args.output / "summary.csv", index=False)
    print("Completed supervised track. No SOTA claim.")


if __name__ == "__main__":
    main()
