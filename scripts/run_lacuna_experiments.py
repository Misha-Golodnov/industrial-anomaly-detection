"""Воспроизводимое сравнение пяти идей Lacuna на TEP, без настройки по отказам.

Запуск: python scripts/run_lacuna_experiments.py
Пилот: --max-runs 3 --epochs 2 --seeds 0 --output data/processed/lacuna_smoke
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

# Ограничение потоков до импорта NumPy: не размножать BLAS-потоки на 16 CPU.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd

from industrial_novelty import (
    CanonicalVariateIndex,
    ExperimentConfig,
    KernelMMDIndex,
    NeuralTransformationIndex,
    SlowFeatureIndex,
    WindowBatch,
    WindowDependenceIndex,
    WindowMomentIndex,
    WindowSpectrumIndex,
    build_experiment,
)
from industrial_novelty.eval.calibration import threshold_at_run_risk
from industrial_novelty.eval.detection import (
    roc_auc,
    run_detections,
    summarize_faults,
    threshold_at_fpr,
)
from industrial_novelty.models.chart import accumulate_runs
from industrial_novelty.models.subspace import tail_scale


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def smooth(values: np.ndarray, meta: pd.DataFrame, center: float, spread: float):
    return accumulate_runs(
        values,
        meta["simulation_run"].to_numpy(),
        meta["sample_end"].to_numpy(),
        lam=0.2,
        center=center,
        spread=spread,
    )


def wilson(rate: float, n: int) -> tuple[float, float]:
    if not n:
        return float("nan"), float("nan")
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (rate + z * z / (2 * n)) / denominator
    radius = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return float(center - radius), float(center + radius)


def alarm_stats(scores: np.ndarray, meta: pd.DataFrame, threshold: float):
    frame = meta.loc[:, ["simulation_run", "sample_end"]].copy()
    frame["alarm"] = scores >= threshold
    ordered = frame.sort_values(["simulation_run", "sample_end"])
    previous = ordered.groupby("simulation_run")["alarm"].shift(fill_value=False)
    episodes = int((ordered["alarm"] & ~previous).sum())
    rate = float(ordered.groupby("simulation_run")["alarm"].any().mean())
    # Знаменатель — наблюдаемое время на сетке решений, включая первый интервал.
    hours = float(ordered.groupby("simulation_run")["sample_end"].max().sum()) * 3 / 60
    return rate, episodes / hours * 1000, len(ordered["simulation_run"].unique())


def score_only_batch(meta: pd.DataFrame) -> WindowBatch:
    return WindowBatch(
        np.zeros((len(meta), 1, 1), dtype=np.float32), meta, ("score",), "report"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить локальный расчёт с тем же manifest и весами",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/lacuna_2026-10-30")
    )
    args = parser.parse_args()
    if args.max_runs is not None and args.max_runs < 2:
        parser.error("Для раздельной длинной калибровки и теста нужно --max-runs >= 2.")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seed не должны повторяться.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cache = (
        Path("data/processed/lacuna_2026-10-30")
        if args.max_runs is None
        else output / "cache"
    )
    cache.mkdir(parents=True, exist_ok=True)
    import torch

    torch.set_num_threads(4)
    config = ExperimentConfig(
        window_size=24, stride=8, channels="all", max_runs=args.max_runs
    )
    log("Чтение нормальных данных")
    experiment = build_experiment(config)
    experiment.counts().to_csv(output / "counts.csv", index=False)
    assert experiment.val is not None

    models = {
        "cva": CanonicalVariateIndex(),
        "moments": WindowMomentIndex(),
        "slow": SlowFeatureIndex(),
        "dependence": WindowDependenceIndex(),
        "second_moments": WindowDependenceIndex(centered=False),
        "spectrum": WindowSpectrumIndex(),
        "mmd1": KernelMMDIndex(history=1),
        "mmd4": KernelMMDIndex(history=4),
        "mmd8": KernelMMDIndex(history=8),
    }
    for seed in args.seeds:
        models[f"neutral{seed}"] = NeuralTransformationIndex(
            epochs=args.epochs, seed=seed
        )
    models["fixed0"] = NeuralTransformationIndex(
        epochs=args.epochs,
        seed=0,
        learn_transformations=False,
    )
    baseline = ("cva", "moments", "slow")
    assemblies = {"baseline": baseline}
    for member in models:
        if member in baseline:
            continue
        assemblies[member] = (member,)
        assemblies[f"baseline+{member}"] = (*baseline, member)

    normal_ids = np.sort(experiment.normal_test.meta["simulation_run"].unique())
    shuffled = normal_ids.copy()
    np.random.default_rng(20261030).shuffle(shuffled)
    n_cal = min(200, max(1, int(len(shuffled) * 0.4)))
    full_cal_ids = np.sort(shuffled[:n_cal])
    full_test_ids = np.sort(shuffled[n_cal:])
    manifest = {
        "window_size": 24,
        "stride": 8,
        "channels": "all",
        "epochs": args.epochs,
        "seeds": args.seeds,
        "max_runs": args.max_runs,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "training_runs": sorted(
            map(int, experiment.train.meta["simulation_run"].unique())
        ),
        "calibration_500_runs_from_training_file": sorted(
            map(int, experiment.val.meta["simulation_run"].unique())
        ),
        "calibration_960_runs_from_normal_test_file": list(map(int, full_cal_ids)),
        "test_960_runs_from_normal_test_file": list(map(int, full_test_ids)),
        "assemblies": assemblies,
        "protocols": [
            "window_1pct_960",
            "run_5pct_500",
            "run_10pct_500",
            "run_5pct_960",
            "run_10pct_960",
        ],
        "note": "Parameters fixed before fault evaluation; prior TEP test exposure makes results exploratory.",
    }
    if args.resume:
        previous_manifest = json.loads(
            (output / "manifest.json").read_text(encoding="utf-8")
        )
        if previous_manifest != json.loads(json.dumps(manifest)):
            raise ValueError("Для resume нужны те же параметры, версии и прогоны.")
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    raw_train, raw_val, raw_normal, timing, neural_history = {}, {}, {}, [], {}
    previous_timing = {}
    previous_history = {}
    if args.resume:
        previous_timing = (
            pd.read_csv(output / "timing.csv").set_index("model").to_dict("index")
        )
        previous_history = json.loads(
            (output / "neural_training.json").read_text(encoding="utf-8")
        )
    scales = {}
    for name, model in models.items():
        log(f"{'Загрузка' if args.resume else 'Обучение'} {name}")
        started = time.perf_counter()
        if args.resume:
            if isinstance(model, NeuralTransformationIndex):
                from industrial_novelty.models.neural import _network

                model.shape_ = experiment.train.X.shape[1:]
                model.network_ = _network(
                    int(np.prod(model.shape_)),
                    model.hidden,
                    model.latent,
                    model.transforms,
                )
                model.network_.load_state_dict(
                    torch.load(cache / f"{name}.pt", weights_only=True)
                )
                model.network_.eval()
                model.train_history_ = previous_history[name]["loss"]
                model.n_parameters_ = previous_history[name]["n_parameters"]
            else:
                # Только собственные локальные checkpoints этого эксперимента.
                with (cache / f"{name}.pkl").open("rb") as stream:
                    model = pickle.load(stream)
                models[name] = model
            fit_seconds = previous_timing[name]["fit_seconds"]
        else:
            model.fit(experiment.train.X)
            fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        raw_train[name] = model.score_batch(experiment.train)
        raw_val[name] = model.score_batch(experiment.val)
        raw_normal[name] = model.score_batch(experiment.normal_test)
        scales[name] = tail_scale(raw_train[name], 0.01)
        normal_seconds = time.perf_counter() - started
        timing.append(
            {
                "model": name,
                "fit_seconds": fit_seconds,
                "normal_score_seconds": normal_seconds,
                "fault_score_seconds": previous_timing.get(name, {}).get(
                    "fault_score_seconds", 0.0
                ),
            }
        )
        if isinstance(model, NeuralTransformationIndex):
            neural_history[name] = {
                "loss": model.train_history_,
                "n_parameters": model.n_parameters_,
            }
            torch.save(model.network_.state_dict(), cache / f"{name}.pt")
        else:
            with (cache / f"{name}.pkl").open("wb") as stream:
                pickle.dump(model, stream)
        log(
            f"{name}: обучение {fit_seconds:.1f} с, нормальные оценки {normal_seconds:.1f} с"
        )
    (output / "neural_training.json").write_text(
        json.dumps(neural_history, indent=2), encoding="utf-8"
    )

    centers, spreads = {}, {}

    def compose(raw, meta, *, fitting=False):
        result = {}
        for name, members in assemblies.items():
            unsmoothed = np.maximum.reduce([raw[key] / scales[key] for key in members])
            if fitting:
                centers[name], spreads[name] = (
                    float(unsmoothed.mean()),
                    float(unsmoothed.std()),
                )
            result[name] = smooth(unsmoothed, meta, centers[name], spreads[name])
        return result

    compose(raw_train, experiment.train.meta, fitting=True)
    val = compose(raw_val, experiment.val.meta)
    normal = compose(raw_normal, experiment.normal_test.meta)
    if args.resume:
        with np.load(cache / "normal_scores.npz") as previous_normal:
            for name in assemblies:
                np.testing.assert_allclose(
                    normal[name], previous_normal[name], rtol=1e-10, atol=1e-10
                )
    pd.DataFrame({"model": list(scales), "scale": list(scales.values())}).to_csv(
        output / "tail_scales.csv", index=False
    )
    np.savez_compressed(cache / "normal_scores.npz", **normal)
    experiment.normal_test.meta.to_csv(cache / "normal_meta.csv", index=False)

    protocols = []
    normal_meta = experiment.normal_test.meta
    for label, horizon, alpha in [
        ("window_1pct_960", 960, None),
        ("run_5pct_500", 500, 0.05),
        ("run_10pct_500", 500, 0.1),
        ("run_5pct_960", 960, 0.05),
        ("run_10pct_960", 960, 0.1),
    ]:
        test_mask = normal_meta["sample_end"].to_numpy() <= horizon
        if alpha is not None and horizon == 960:
            test_mask &= normal_meta["simulation_run"].isin(full_test_ids).to_numpy()
        for name in assemblies:
            if alpha is None:
                threshold = threshold_at_fpr(val[name], 0.01)
                n_calibration = experiment.val.meta["simulation_run"].nunique()
            elif horizon == 500:
                threshold = threshold_at_run_risk(
                    val[name], experiment.val.meta, alpha=alpha, horizon=horizon
                )
                n_calibration = experiment.val.meta["simulation_run"].nunique()
            else:
                cal_mask = normal_meta["simulation_run"].isin(full_cal_ids).to_numpy()
                threshold = threshold_at_run_risk(
                    normal[name][cal_mask],
                    normal_meta.loc[cal_mask],
                    alpha=alpha,
                    horizon=horizon,
                )
                n_calibration = len(full_cal_ids)
            chosen = normal[name][test_mask]
            meta = normal_meta.loc[test_mask]
            run_rate, episodes, n_runs = alarm_stats(chosen, meta, threshold)
            low, high = wilson(run_rate, n_runs)
            protocols.append(
                {
                    "model": name,
                    "protocol": label,
                    "horizon": horizon,
                    "budget": 0.01 if alpha is None else alpha,
                    "budget_unit": "window" if alpha is None else "run",
                    "n_calibration_runs": int(n_calibration),
                    "n_normal_test_runs": n_runs,
                    "threshold": threshold,
                    "normal_window_fpr": float(np.mean(chosen >= threshold)),
                    "false_alarm_run_rate": run_rate,
                    "false_alarm_run_ci_low": low,
                    "false_alarm_run_ci_high": high,
                    "alarm_episodes_per_1000h": episodes,
                    "negative_scores": chosen,
                    "run_tables": [],
                    "auc": [],
                }
            )
    pd.DataFrame(
        [
            {
                k: v
                for k, v in p.items()
                if k not in {"negative_scores", "run_tables", "auc"}
            }
            for p in protocols
        ]
    ).to_csv(output / "normal_metrics.csv", index=False)
    log("Чтение неисправного теста; все параметры и пороги уже зафиксированы")
    for fault in experiment.iter_faulty_test():
        log(f"Отказ {fault.fault_number}/20: {len(fault.batch)} окон")
        score_path = cache / f"fault_{fault.fault_number:02d}_scores.npz"
        meta_path = cache / f"fault_{fault.fault_number:02d}_meta.csv"
        if args.resume and score_path.is_file():
            pd.testing.assert_frame_equal(
                pd.read_csv(meta_path),
                fault.batch.meta.reset_index(drop=True),
                check_dtype=False,
            )
            with np.load(score_path) as saved:
                scores = {name: saved[name] for name in assemblies}
        else:
            raw = {}
            for i, (name, model) in enumerate(models.items()):
                started = time.perf_counter()
                raw[name] = model.score_batch(fault.batch)
                timing[i]["fault_score_seconds"] += time.perf_counter() - started
            scores = compose(raw, fault.batch.meta)
            np.savez_compressed(score_path, **scores)
            fault.batch.meta.to_csv(meta_path, index=False)
        prepared = {}
        for horizon in (500, 960):
            mask = fault.batch.meta["sample_end"].to_numpy() <= horizon
            meta = fault.batch.meta.loc[mask].reset_index(drop=True)
            eligible = (meta["label"].to_numpy() == 1) & ~meta[
                "post_shutdown"
            ].to_numpy(dtype=bool)
            prepared[horizon] = (mask, score_only_batch(meta), eligible)
        for protocol in protocols:
            mask, batch, eligible = prepared[protocol["horizon"]]
            values = scores[protocol["model"]][mask]
            table = run_detections(
                batch,
                values,
                threshold=protocol["threshold"],
                last_normal_sample=160,
                fault_number=fault.fault_number,
                simulation_runs=fault.simulation_runs,
            )
            protocol["run_tables"].append(table)
            protocol["auc"].append(
                roc_auc(values[eligible], protocol["negative_scores"])
            )
        pd.DataFrame(timing).to_csv(output / "timing.csv", index=False)

    summary, fault_tables = [], []
    for protocol in protocols:
        per_run = pd.concat(protocol.pop("run_tables"), ignore_index=True)
        per_fault = summarize_faults(per_run)
        per_fault["auroc"] = protocol.pop("auc")
        per_fault.insert(0, "model", protocol["model"])
        per_fault.insert(1, "protocol", protocol["protocol"])
        fault_tables.append(per_fault)
        per_run.to_csv(
            cache / f"{protocol['model']}_{protocol['protocol']}_runs.csv.gz",
            index=False,
        )
        protocol.pop("negative_scores")
        hard = per_fault["fault_number"].isin([3, 9, 15])
        easy, difficult = per_fault.loc[~hard], per_fault.loc[hard]
        protocol.update(
            {
                "easy_detection": float(easy["detection_rate"].mean()),
                "easy_early_detection": float(easy["early_detection_rate"].mean()),
                "easy_median_delay_minutes": float(
                    easy["median_delay_minutes"].median()
                ),
                "hard_detection": float(difficult["detection_rate"].mean()),
                "hard_auroc": float(difficult["auroc"].mean()),
                "hard_median_delay_minutes": float(
                    difficult["median_delay_minutes"].median()
                ),
            }
        )
        summary.append(protocol)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    pd.concat(fault_tables, ignore_index=True).to_csv(
        output / "per_fault.csv", index=False
    )
    log(f"Готово: {output.resolve()}")


if __name__ == "__main__":
    main()
