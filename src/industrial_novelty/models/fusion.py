"""Сумма оценок, каждая в единицах своего нормального хвоста."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from industrial_novelty.models.base import NoveltyDetector
from industrial_novelty.models.subspace import tail_scale


class TailSum(NoveltyDetector):
    """Сумма оценок нескольких детекторов после калибровки хвостов.

    У каждой модели свой масштаб. Деление на обучающий квантиль уровня
    ``1 - alpha`` переводит оценку в единицы «сколько нормальных хвостов
    этой модели». Сумма тогда не отдаёт предпочтение той, у которой числа
    просто крупнее. Порог решения по-прежнему выбирается снаружи.
    """

    def __init__(
        self, members: Sequence[NoveltyDetector], *, alpha: float = 0.01
    ) -> None:
        if not members:
            raise ValueError("TailSum нужен хотя бы один детектор.")
        if not all(isinstance(member, NoveltyDetector) for member in members):
            raise TypeError("Все слагаемые должны быть NoveltyDetector.")
        if not 0 < alpha < 1:
            raise ValueError("alpha должен лежать в интервале (0, 1).")
        self.members = tuple(members)
        self.alpha = float(alpha)

    def fit_windows(self, X: np.ndarray) -> None:
        scales = []
        for member in self.members:
            member.fit(X)
            scales.append(tail_scale(member.score(X), self.alpha))
        self.scales_ = np.asarray(scales, dtype=np.float64)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "scales_"):
            raise RuntimeError("Сначала вызовите fit.")
        total = np.zeros(len(X), dtype=np.float64)
        for member, scale in zip(self.members, self.scales_, strict=True):
            total += member.score(X) / scale
        return total

    def __repr__(self) -> str:
        names = ", ".join(type(member).__name__ for member in self.members)
        return f"TailSum({names})"
