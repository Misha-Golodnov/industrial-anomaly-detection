"""Сумма и максимум оценок в единицах нормального хвоста."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from industrial_novelty.models.base import NoveltyDetector
from industrial_novelty.models.subspace import tail_scale


class _CalibratedTails(NoveltyDetector):
    def __init__(
        self,
        members: Sequence[NoveltyDetector],
        *,
        alpha: float,
        reduce: str,
    ) -> None:
        if not members:
            raise ValueError("Нужен хотя бы один детектор.")
        if not all(isinstance(member, NoveltyDetector) for member in members):
            raise TypeError("Все слагаемые должны быть NoveltyDetector.")
        if not 0 < alpha < 1:
            raise ValueError("alpha должен лежать в интервале (0, 1).")
        if reduce not in {"sum", "max"}:
            raise ValueError("reduce должен быть 'sum' или 'max'.")
        self.members = tuple(members)
        self.alpha = float(alpha)
        self._reduce = reduce

    def fit_windows(self, X: np.ndarray) -> None:
        scales = []
        for member in self.members:
            member.fit(X)
            scales.append(tail_scale(member.score(X), self.alpha))
        self.scales_ = np.asarray(scales, dtype=np.float64)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "scales_"):
            raise RuntimeError("Сначала вызовите fit.")
        parts = [
            member.score(X) / scale
            for member, scale in zip(self.members, self.scales_, strict=True)
        ]
        if self._reduce == "max":
            return np.maximum.reduce(parts)
        total = np.zeros(len(X), dtype=np.float64)
        for part in parts:
            total += part
        return total

    def _names(self) -> str:
        return ", ".join(type(member).__name__ for member in self.members)


class TailSum(_CalibratedTails):
    """Сумма оценок нескольких детекторов после калибровки хвостов.

    У каждой модели свой масштаб. Деление на обучающий квантиль уровня
    ``1 - alpha`` переводит оценку в единицы «сколько нормальных хвостов
    этой модели». Сумма тогда не отдаёт предпочтение той, у которой числа
    просто крупнее. Порог решения по-прежнему выбирается снаружи.
    """

    def __init__(
        self, members: Sequence[NoveltyDetector], *, alpha: float = 0.01
    ) -> None:
        super().__init__(members, alpha=alpha, reduce="sum")

    def __repr__(self) -> str:
        return f"TailSum({self._names()})"


class TailMax(_CalibratedTails):
    """Максимум хвостов нескольких детекторов.

    В отличие от суммы, детектор, который видит отказ, не разбавляется
    остальными, если те в этот момент близки к норме. Порог по-прежнему один
    и выбирается снаружи.
    """

    def __init__(
        self, members: Sequence[NoveltyDetector], *, alpha: float = 0.01
    ) -> None:
        super().__init__(members, alpha=alpha, reduce="max")

    def __repr__(self) -> str:
        return f"TailMax({self._names()})"
