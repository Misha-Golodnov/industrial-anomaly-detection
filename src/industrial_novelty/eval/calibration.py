"""Split-conformal порог для вероятности тревоги за фиксированный горизонт."""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_maxima(scores: np.ndarray, meta: pd.DataFrame, *, horizon: int) -> pd.Series:
    """Максимумы по прогонам до horizon; требует одинаковой полной сетки окон."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) != len(meta) or not len(values) or not np.isfinite(values).all():
        raise ValueError("Нужны конечные оценки и непустой паспорт той же длины.")
    if horizon < 1:
        raise ValueError("horizon должен быть положительным.")
    if (meta["label"] != 0).any():
        raise ValueError("Калибровка допускает только нормальные прогоны.")
    if meta.duplicated(["simulation_run", "sample_end"]).any():
        raise ValueError("В паспорте повторяются окна одного прогона.")
    selected = meta.loc[
        meta["sample_end"] <= horizon, ["simulation_run", "sample_end"]
    ].copy()
    selected["score"] = values[meta["sample_end"].to_numpy() <= horizon]
    grids = selected.groupby("simulation_run")["sample_end"].agg(
        lambda s: tuple(sorted(s))
    )
    if len(grids) != meta["simulation_run"].nunique() or not len(grids):
        raise ValueError("Некоторые прогоны не покрывают выбранный горизонт.")
    if any(grid != grids.iloc[0] for grid in grids):
        raise ValueError("Сетка калибровочных окон должна совпадать между прогонами.")
    grid = np.asarray(grids.iloc[0])
    if len(grid) > 1:
        steps = np.diff(grid)
        if np.any(steps != steps[0]) or grid[-1] + steps[0] <= horizon:
            raise ValueError("Нужна полная регулярная сетка до заданного горизонта.")
    elif grid[-1] != horizon:
        raise ValueError("По одному окну нельзя подтвердить покрытие горизонта.")
    return selected.groupby("simulation_run")["score"].max()


def threshold_at_run_risk(
    scores: np.ndarray, meta: pd.DataFrame, *, alpha: float, horizon: int
) -> float:
    """Порог для правила score >= threshold; маргинальный контроль по прогонам.

    Нужны обменные нормальные калибровочные/тестовые траектории одинакового
    горизонта и детектор, зафиксированный независимо от калибровки.
    Для недостаточной выборки возвращает +inf, то есть отсутствие тревог.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha должен лежать в (0, 1).")
    maxima = run_maxima(scores, meta, horizon=horizon).to_numpy()
    rank = int(np.ceil((len(maxima) + 1) * (1 - alpha)))
    if rank > len(maxima):
        return float("inf")
    quantile = np.partition(maxima, rank - 1)[rank - 1]
    return float(np.nextafter(quantile, np.inf))
