"""Сглаживание оценки вдоль одного прогона."""

from __future__ import annotations

import numpy as np

from industrial_novelty.models.base import NoveltyDetector


def accumulate_runs(
    scores: np.ndarray,
    run_ids: np.ndarray,
    sample_ends: np.ndarray,
    *,
    lam: float,
    center: float,
    spread: float,
) -> np.ndarray:
    """Одностороннее EWMA стандартизованной оценки, сброс на каждом прогоне.

    Это скалярная карта Робертса, не векторный MEWMA: на вход уже приходит
    одна оценка новизны. ``lam`` — вес нового окна. Меньший вес копит слабый
    сдвиг и гасит одиночный всплеск. Окна прогона упорядочиваются по
    ``sample_end``.
    """
    if not 0 < lam <= 1:
        raise ValueError("lam должен лежать в полуинтервале (0, 1].")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    runs = np.asarray(run_ids)
    ends = np.asarray(sample_ends)
    if not (len(values) == len(runs) == len(ends)):
        raise ValueError("Оценки, прогоны и концы окон должны быть одной длины.")
    out = np.empty(len(values), dtype=np.float64)
    if len(values) == 0:
        return out
    scale = max(float(spread), 1e-8)
    order = np.lexsort((ends, runs))
    start = 0
    keep = 1.0 - lam
    while start < len(order):
        run = runs[order[start]]
        stop = start + 1
        while stop < len(order) and runs[order[stop]] == run:
            stop += 1
        state = 0.0
        for position in order[start:stop]:
            state = keep * state + lam * (float(values[position]) - center) / scale
            out[position] = state
        start = stop
    return out


class WithinRunEWMA(NoveltyDetector):
    """EWMA внутренней оценки с разрывом между прогонами.

    Центр и размах берутся по обучающим окнам. Порог снаружи калибруется уже
    по сглаженным значениям, поэтому доля ложных окон остаётся под контролем
    ``threshold_at_fpr``. ``score`` без паспорта считает всю пачку одним
    прогоном. ``evaluate_model`` передаёт паспорт и сбрасывает состояние.
    """

    def __init__(self, inner: NoveltyDetector, *, lam: float = 0.2) -> None:
        if not isinstance(inner, NoveltyDetector):
            raise TypeError("Внутренняя модель должна быть NoveltyDetector.")
        if not 0 < lam <= 1:
            raise ValueError("lam должен лежать в полуинтервале (0, 1].")
        self.inner = inner
        self.lam = float(lam)

    def fit_windows(self, X: np.ndarray) -> None:
        self.inner.fit(X)
        scores = self.inner.score(X)
        self.center_ = float(np.mean(scores))
        self.spread_ = float(np.std(scores))

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "center_"):
            raise RuntimeError("Сначала вызовите fit.")
        raw = self.inner.score(X)
        if len(raw) == 0:
            return raw
        runs = np.zeros(len(raw), dtype=np.int32)
        ends = np.arange(len(raw))
        return accumulate_runs(
            raw, runs, ends, lam=self.lam, center=self.center_, spread=self.spread_
        )

    def score_batch(self, batch) -> np.ndarray:
        if not hasattr(self, "center_"):
            raise RuntimeError("Сначала вызовите fit.")
        raw = self.inner.score(batch.X)
        meta = batch.meta
        return accumulate_runs(
            raw,
            meta["simulation_run"].to_numpy(),
            meta["sample_end"].to_numpy(),
            lam=self.lam,
            center=self.center_,
            spread=self.spread_,
        )

    def __repr__(self) -> str:
        return f"WithinRunEWMA({type(self.inner).__name__}, lam={self.lam})"
