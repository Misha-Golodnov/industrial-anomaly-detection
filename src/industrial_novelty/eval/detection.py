"""Порог по доле ложных тревог и обнаружение отказа на уровне прогона."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from industrial_novelty.data.experiment import Experiment
from industrial_novelty.data.schema import SAMPLE_PERIOD_MINUTES
from industrial_novelty.data.windows import WindowBatch
from industrial_novelty.models.base import NoveltyDetector

RUN_COLUMNS = (
    "fault_number",
    "simulation_run",
    "scoreable",
    "detected",
    "early",
    "alarm_persistence",
    "delay_samples",
    "delay_minutes",
    "first_detection_sample",
    "shutdown_start_sample",
    "shutdown_margin_minutes",
    "n_scoreable_windows",
)

FAULT_COLUMNS = (
    "fault_number",
    "n_runs",
    "n_scoreable",
    "n_detected",
    "detection_rate",
    "early_detection_rate",
    "median_alarm_persistence",
    "median_delay_samples",
    "median_delay_minutes",
    "median_shutdown_margin_minutes",
    "n_shutdown_runs",
    "n_unscoreable",
    "auroc",
)


@dataclass(frozen=True)
class DetectionReport:
    """Итог сравнения на общем пороге.

    Порог выбран так, чтобы на нормальных валидационных окнах доля оценок
    не ниже порога не превышала ``target_fpr``. ``normal_test_fpr`` считается
    уже на отложенном нормальном тесте. Обнаружение отказа идёт только по окнам
    с меткой 1, которые не лежат целиком после аварийного останова.
    Задержка — число отсчётов от последнего нормального отсчёта до конца первого
    сработавшего окна. ``false_alarm_run_rate`` — доля нормальных тестовых
    прогонов, где хотя бы одно окно выше порога.
    """

    threshold: float
    target_fpr: float
    normal_test_fpr: float
    false_alarm_run_rate: float
    n_normal_test_windows: int
    n_normal_test_runs: int
    per_run: pd.DataFrame
    per_fault: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"DetectionReport(threshold={self.threshold:.4g}, "
            f"target_fpr={self.target_fpr:.3g}, "
            f"normal_test_fpr={self.normal_test_fpr:.3g}, "
            f"false_alarm_run_rate={self.false_alarm_run_rate:.3g}, "
            f"faults={len(self.per_fault)})"
        )


def threshold_at_fpr(scores: np.ndarray, target_fpr: float) -> float:
    """Вернуть порог, при котором доля ``score >= порог`` не выше ``target_fpr``.

    Окно считается тревогой при ``score >= threshold``. Если одинаковые оценки
    не дают попасть ровно в бюджет, порог поднимается и фактическая доля
    получается меньше запрошенной.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise ValueError("Нельзя выбрать порог по пустому набору оценок.")
    if not np.isfinite(values).all():
        raise ValueError("Оценки должны быть конечными.")
    if not 0 <= target_fpr <= 1:
        raise ValueError("target_fpr должен лежать в отрезке [0, 1].")
    n_allowed = int(np.floor(target_fpr * len(values)))
    if n_allowed == 0:
        return float(np.nextafter(np.max(values), np.inf))
    candidate = float(np.sort(values)[-n_allowed])
    if int(np.sum(values >= candidate)) > n_allowed:
        return float(np.nextafter(candidate, np.inf))
    return candidate


def evaluate_model(
    model: NoveltyDetector,
    experiment: Experiment,
    *,
    target_fpr: float = 0.01,
    threshold: float | None = None,
    fault_numbers: Sequence[int] | None = None,
) -> DetectionReport:
    """Посчитать FPR на нормальном тесте и обнаружение по каждой неисправности.

    Если ``threshold`` не задан, он выбирается по ``experiment.val``.
    Для этого при сборке эксперимента нужен ``val_fraction > 0``.
    """
    if threshold is None:
        if experiment.val is None:
            raise ValueError(
                "Нет валидационных окон для выбора порога. "
                "Задайте val_fraction > 0 или передайте threshold."
            )
        threshold = threshold_at_fpr(model.score(experiment.val.X), target_fpr)
    normal_scores = model.score(experiment.normal_test.X)
    normal_fpr = (
        float(np.mean(normal_scores >= threshold))
        if len(normal_scores)
        else float("nan")
    )
    run_alarm_rate, n_normal_runs = _false_alarm_run_rate(
        experiment.normal_test.meta, normal_scores, threshold
    )
    tables = []
    auc_rows = []
    for fault in experiment.iter_faulty_test(
        None if fault_numbers is None else tuple(fault_numbers)
    ):
        scores = _score_batch(model, fault.batch)
        tables.append(
            run_detections(
                fault.batch,
                scores,
                threshold=threshold,
                last_normal_sample=experiment.config.fault_onset_sample - 1,
                fault_number=fault.fault_number,
                simulation_runs=fault.simulation_runs,
            )
        )
        auc_rows.append(
            {
                "fault_number": fault.fault_number,
                "auroc": roc_auc(_scoreable_scores(fault.batch, scores), normal_scores),
            }
        )
    per_run = pd.concat(tables, ignore_index=True) if tables else _empty_runs()
    per_fault = summarize_faults(per_run)
    if auc_rows:
        per_fault = per_fault.drop(columns=["auroc"]).merge(
            pd.DataFrame(auc_rows), on="fault_number", how="left"
        )
        per_fault = per_fault.loc[:, list(FAULT_COLUMNS)]
    return DetectionReport(
        threshold=float(threshold),
        target_fpr=float(target_fpr),
        normal_test_fpr=normal_fpr,
        false_alarm_run_rate=run_alarm_rate,
        n_normal_test_windows=len(normal_scores),
        n_normal_test_runs=n_normal_runs,
        per_run=per_run,
        per_fault=per_fault,
    )


def run_detections(
    batch: WindowBatch,
    scores: np.ndarray,
    *,
    threshold: float,
    last_normal_sample: int,
    fault_number: int,
    simulation_runs: Sequence[int],
) -> pd.DataFrame:
    """Свести оконные оценки одного отказа к одной строке на прогон."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != len(batch):
        raise ValueError(f"Ожидалось {len(batch)} оценок, получено {len(values)}.")
    if (
        len(batch)
        and not np.isin(batch.meta["fault_number"].to_numpy(), [fault_number]).all()
    ):
        raise ValueError("В пачке окон есть чужой номер неисправности.")
    scored = batch.with_scores(values) if len(batch) else batch.meta.copy()
    rows = []
    seen: set[int] = set()
    if len(scored):
        grouped = scored.groupby("simulation_run", sort=True)
        for run, part in grouped:
            rows.append(
                _run_row(int(run), part, fault_number, threshold, last_normal_sample)
            )
            seen.add(int(run))
    for run in simulation_runs:
        if int(run) not in seen:
            rows.append(_missing_run(fault_number, int(run)))
    table = pd.DataFrame(rows, columns=list(RUN_COLUMNS))
    table["first_detection_sample"] = table["first_detection_sample"].astype("Int64")
    table["shutdown_start_sample"] = table["shutdown_start_sample"].astype("Int64")
    return table.sort_values("simulation_run", kind="mergesort").reset_index(drop=True)


def summarize_faults(per_run: pd.DataFrame) -> pd.DataFrame:
    """Доля обнаруженных прогонов среди тех, где был участок отказа до останова."""
    if len(per_run) == 0:
        return pd.DataFrame(columns=list(FAULT_COLUMNS))
    rows = []
    for fault, part in per_run.groupby("fault_number", sort=True):
        scoreable = part.loc[part["scoreable"]]
        detected = scoreable.loc[scoreable["detected"]]
        n_scoreable = len(scoreable)
        n_detected = len(detected)
        margins = part["shutdown_margin_minutes"]
        rows.append(
            {
                "fault_number": int(fault),
                "n_runs": len(part),
                "n_scoreable": n_scoreable,
                "n_detected": n_detected,
                "detection_rate": (n_detected / n_scoreable) if n_scoreable else np.nan,
                "early_detection_rate": (
                    float(scoreable["early"].mean()) if n_scoreable else np.nan
                ),
                "median_alarm_persistence": (
                    float(scoreable["alarm_persistence"].median())
                    if n_scoreable
                    else np.nan
                ),
                "median_delay_samples": (
                    float(detected["delay_samples"].median()) if n_detected else np.nan
                ),
                "median_delay_minutes": (
                    float(detected["delay_minutes"].median()) if n_detected else np.nan
                ),
                "median_shutdown_margin_minutes": (
                    float(margins.median()) if margins.notna().any() else np.nan
                ),
                "n_shutdown_runs": int(part["shutdown_start_sample"].notna().sum()),
                "n_unscoreable": int((~part["scoreable"]).sum()),
                "auroc": np.nan,
            }
        )
    return pd.DataFrame(rows, columns=list(FAULT_COLUMNS))


def roc_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """AUROC для оценок, где большее значение означает отказ.

    Совпадения оценок получают средний ранг. Пустой класс даёт NaN.
    """
    positive = np.asarray(positive, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative, dtype=np.float64).reshape(-1)
    n_pos = len(positive)
    n_neg = len(negative)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    scores = np.concatenate([positive, negative])
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    rank_sum = float(ranks[:n_pos].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _false_alarm_run_rate(
    meta: pd.DataFrame, scores: np.ndarray, threshold: float
) -> tuple[float, int]:
    if len(meta) == 0:
        return float("nan"), 0
    frame = meta.loc[:, ["simulation_run"]].copy()
    frame["alarm"] = np.asarray(scores, dtype=np.float64).reshape(-1) >= threshold
    per_run = frame.groupby("simulation_run", sort=False)["alarm"].any()
    return float(per_run.mean()), len(per_run)


def _scoreable_scores(batch: WindowBatch, scores: np.ndarray) -> np.ndarray:
    if len(batch) == 0:
        return np.empty(0, dtype=np.float64)
    meta = batch.meta
    eligible = (meta["label"].to_numpy() == 1) & (
        ~meta["post_shutdown"].to_numpy(dtype=bool)
    )
    return np.asarray(scores, dtype=np.float64).reshape(-1)[eligible]


def _score_batch(model: NoveltyDetector, batch: WindowBatch) -> np.ndarray:
    if len(batch) == 0:
        return np.empty(0, dtype=np.float64)
    return model.score(batch.X)


def _run_row(
    run: int,
    part: pd.DataFrame,
    fault_number: int,
    threshold: float,
    last_normal_sample: int,
) -> dict[str, object]:
    eligible = part.loc[(part["label"] == 1) & (~part["post_shutdown"].astype(bool))]
    hits = eligible.loc[eligible["score"] >= threshold]
    detected = len(hits) > 0 and len(eligible) > 0
    shutdown = part.loc[part["post_shutdown"].astype(bool)]
    shutdown_start = int(shutdown["sample_start"].min()) if len(shutdown) else None
    if detected:
        first = int(hits["sample_end"].min())
        delay = first - last_normal_sample
        first_sample: int | None = first
        delay_value: float = float(delay)
        early = first == int(eligible["sample_end"].min())
        margin = (
            float((shutdown_start - first) * SAMPLE_PERIOD_MINUTES)
            if shutdown_start is not None
            else np.nan
        )
    else:
        first_sample = None
        delay_value = np.nan
        early = False
        margin = np.nan
    persistence = float(len(hits) / len(eligible)) if len(eligible) else np.nan
    return {
        "fault_number": fault_number,
        "simulation_run": run,
        "scoreable": len(eligible) > 0,
        "detected": detected,
        "early": early,
        "alarm_persistence": persistence,
        "delay_samples": delay_value,
        "delay_minutes": delay_value * SAMPLE_PERIOD_MINUTES,
        "first_detection_sample": first_sample,
        "shutdown_start_sample": shutdown_start,
        "shutdown_margin_minutes": margin,
        "n_scoreable_windows": len(eligible),
    }


def _missing_run(fault_number: int, run: int) -> dict[str, object]:
    return {
        "fault_number": fault_number,
        "simulation_run": run,
        "scoreable": False,
        "detected": False,
        "early": False,
        "alarm_persistence": np.nan,
        "delay_samples": np.nan,
        "delay_minutes": np.nan,
        "first_detection_sample": None,
        "shutdown_start_sample": None,
        "shutdown_margin_minutes": np.nan,
        "n_scoreable_windows": 0,
    }


def _empty_runs() -> pd.DataFrame:
    return pd.DataFrame(columns=list(RUN_COLUMNS))
