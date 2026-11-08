"""Second supervised track: physical feature ablations and severity-profiled GLR."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from industrial_novelty.data.ashrae import FAULTS, chiller_windows, normal_split
from industrial_novelty.models.chiller_diagnosis import FaultSignatureClassifier
from industrial_novelty.models.chiller_directions import window_features

SPECS = {
    "raw_logistic": ("raw_moments", "logistic"),
    "thermo_logistic": ("thermo_moments", "logistic"),
    "thermo_eos_logistic": ("thermo_eos", "logistic"),
    "raw_thermo_logistic": ("raw_thermo_moments", "logistic"),
    "thermo_svm": ("thermo_moments", "svm"),
    "raw_thermo_svm": ("raw_thermo_moments", "svm"),
    "thermo_lda": ("thermo_moments", "lda"),
    "raw_thermo_lda": ("raw_thermo_moments", "lda"),
    "ray_invariant": ("invariant", "ray"),
    "ray_thermo": ("thermo", "ray"),
    "ray_conditional_thermo": ("thermo", "conditional_ray"),
    "subspace_thermo": ("thermo", "subspace"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/ashrae_public"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ashrae_directions_2026-11-04/supervised"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--windows", type=int, nargs="+", default=[12, 48])
    parser.add_argument(
        "--protocols", nargs="+", default=["blocked", "loto40", "loto45", "loto50"]
    )
    parser.add_argument("--methods", nargs="+", default=list(SPECS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    files = [
        Path(__file__),
        Path("src/industrial_novelty/models/chiller_diagnosis.py"),
        Path("src/industrial_novelty/models/chiller_directions.py"),
        Path("src/industrial_novelty/data/ashrae.py"),
    ]
    manifest = {
        "date": "2026-11-04",
        "status": "exploratory development; earlier test results were inspected",
        "specifications": {name: SPECS[name] for name in args.methods},
        "seeds": args.seeds,
        "windows": args.windows,
        "protocols": args.protocols,
        "scenarios": ["seen_sl1", "transfer_sl1"],
        "selection": "fixed feature/head combinations; no per-test tuning",
        "signature_gate": "normal reference covariance plus normal calibration, nominal 5% window risk",
        "signature_warning": "positive-ray severity model is an approximation; two-dimensional subspace is sign-unrestricted",
        "code_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    records, class_records, split_records = [], [], []
    for w in args.windows:
        normal = chiller_windows(args.data / "normal10.csv", window=w)
        faults = {
            (f, s): chiller_windows(args.data / f"{f}{s}.csv", window=w)
            for f in FAULTS
            for s in (10, 20, 30, 40)
        }
        for protocol in args.protocols:
            band = None if protocol == "blocked" else int(protocol[4:])
            for seed in args.seeds:
                norm_parts, norm_blocks = normal_split(normal, seed, held_out_band=band)
                parts = {}
                split_records.append(
                    {
                        "window": w,
                        "protocol": protocol,
                        "seed": seed,
                        "file": "normal10.csv",
                        "blocks": norm_blocks,
                    }
                )
                for key, batch in faults.items():
                    parts[key], blocks = normal_split(batch, seed, held_out_band=band)
                    split_records.append(
                        {
                            "window": w,
                            "protocol": protocol,
                            "seed": seed,
                            "file": f"{key[0]}{key[1]}.csv",
                            "blocks": blocks,
                        }
                    )
                for scenario in ("seen_sl1", "transfer_sl1"):
                    train_x, train_y = (
                        [norm_parts["fit"].X],
                        [np.zeros(len(norm_parts["fit"].X), int)],
                    )
                    test_x, test_y = (
                        [norm_parts["test"].X],
                        [np.zeros(len(norm_parts["test"].X), int)],
                    )
                    for label, fault in enumerate(FAULTS, start=1):
                        for severity in (
                            [10] if scenario == "seen_sl1" else [20, 30, 40]
                        ):
                            x = parts[fault, severity]["fit"].X
                            train_x.append(x)
                            train_y.append(np.full(len(x), label))
                        x = parts[fault, 10]["test"].X
                        test_x.append(x)
                        test_y.append(np.full(len(x), label))
                    X, y, test, truth = (
                        np.concatenate(train_x),
                        np.concatenate(train_y),
                        np.concatenate(test_x),
                        np.concatenate(test_y),
                    )
                    for name in args.methods:
                        kind, head = SPECS[name]
                        started = time.perf_counter()
                        with threadpool_limits(limits=2):
                            if head in ("ray", "conditional_ray", "subspace"):
                                model = FaultSignatureClassifier(
                                    kind=kind,
                                    conditional=head == "conditional_ray",
                                    subspace=head == "subspace",
                                )
                                model.fit(
                                    X,
                                    y,
                                    norm_parts["reference"].X,
                                    norm_parts["calibration"].X,
                                )
                                prediction = model.predict(test)
                                threshold = model.threshold_
                            else:
                                scaler = StandardScaler().fit(window_features(X, kind))
                                train_features, test_features = (
                                    scaler.transform(window_features(X, kind)),
                                    scaler.transform(window_features(test, kind)),
                                )
                                if head == "logistic":
                                    model = LogisticRegression(
                                        C=1, max_iter=1500, class_weight="balanced"
                                    )
                                elif head == "svm":
                                    model = SVC(
                                        C=10, gamma="scale", class_weight="balanced"
                                    )
                                else:
                                    model = LinearDiscriminantAnalysis(
                                        solver="lsqr",
                                        shrinkage="auto",
                                        priors=np.full(8, 1 / 8),
                                    )
                                prediction = model.fit(train_features, y).predict(
                                    test_features
                                )
                                threshold = float("nan")
                        cm = confusion_matrix(truth, prediction, labels=np.arange(8))
                        recalls = recall_score(
                            truth,
                            prediction,
                            labels=np.arange(8),
                            average=None,
                            zero_division=0,
                        )
                        fpr = float(np.mean(prediction[truth == 0] != 0))
                        detected = float(
                            np.mean(
                                [
                                    np.mean(prediction[truth == k] != 0)
                                    for k in range(1, 8)
                                ]
                            )
                        )
                        records.append(
                            {
                                "window": w,
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
                                "fault_detection_recall": detected,
                                "threshold": threshold,
                                "seconds": time.perf_counter() - started,
                            }
                        )
                        for label, recall in enumerate(recalls):
                            class_records.append(
                                {
                                    "window": w,
                                    "protocol": protocol,
                                    "seed": seed,
                                    "scenario": scenario,
                                    "method": name,
                                    "class": (["normal", *FAULTS])[label],
                                    "recall": recall,
                                    "confusion_row": json.dumps(cm[label].tolist()),
                                }
                            )
                        print(
                            f"w={w} {protocol} seed={seed} {scenario} {name}: F1={records[-1]['macro_f1']:.3f}",
                            flush=True,
                        )
                    pd.DataFrame(records).to_csv(
                        args.output / "per_seed.csv", index=False
                    )
                    pd.DataFrame(class_records).to_csv(
                        args.output / "per_class.csv", index=False
                    )
                    (args.output / "split_blocks.json").write_text(
                        json.dumps(split_records, indent=2), encoding="utf-8"
                    )
    frame = pd.DataFrame(records)
    summary = frame.groupby(["window", "protocol", "scenario", "method"])[
        ["macro_f1", "balanced_accuracy", "normal_fpr", "fault_detection_recall"]
    ].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(args.output / "summary.csv", index=False)
    print(f"Completed {len(records)} supervised development fits.")


if __name__ == "__main__":
    main()
