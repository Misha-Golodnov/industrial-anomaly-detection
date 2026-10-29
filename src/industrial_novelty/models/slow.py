"""Медленные признаки: проекции, которые при исправной работе почти не дрожат."""

from __future__ import annotations

import numpy as np

from industrial_novelty.models.base import NoveltyDetector
from industrial_novelty.models.subspace import ShrunkMahalanobis, tail_scale


class SlowFeatureIndex(NoveltyDetector):
    """Уровень и скорость медленных проекций окна.

    Медленный признак Визкотта и Сейновски минимизирует дисперсию приращения
    при единичной дисперсии самого сигнала. В мониторинге процесса по этим
    проекциям отдельно смотрят отклонение режима и скорость (Шан и соавторы).
    Здесь обе величины собраны в один вектор на окно и сравниваются с нормой
    расстоянием Махаланобиса после усадки Ледуа–Вулфа.

    Оставляются ``n_slow`` самых медленных направлений. Для отказа, который
    залипает и тем самым замедляет датчик относительно обычного шума, этого
    достаточно: быстрые проекции в оценку не входят.
    """

    def __init__(self, n_slow: int = 8, *, alpha: float = 0.01) -> None:
        if n_slow < 1:
            raise ValueError("n_slow должен быть >= 1.")
        if not 0 < alpha < 1:
            raise ValueError("alpha должен лежать в интервале (0, 1).")
        self.n_slow = int(n_slow)
        self.alpha = float(alpha)
        self.metric = ShrunkMahalanobis()

    def fit_windows(self, X: np.ndarray) -> None:
        windows = _as_windows(X)
        if windows.shape[1] < 2:
            raise ValueError("Медленным признакам нужно окно длины хотя бы 2.")
        left, delta = _pairs(windows)
        self.mean_ = left.mean(axis=0)
        covariance = (left - self.mean_).T @ (left - self.mean_) / len(left)
        delta_covariance = (delta.T @ delta) / len(delta)
        self.directions_ = _slow_directions(covariance, delta_covariance, self.n_slow)
        features = self._features(windows)
        self.metric.fit(features)
        self.scale_ = tail_scale(self.metric.score(features), self.alpha)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "scale_"):
            raise RuntimeError("Сначала вызовите fit.")
        return self.metric.score(self._features(_as_windows(X))) / self.scale_

    def _features(self, windows: np.ndarray) -> np.ndarray:
        projected = (
            np.asarray(windows, dtype=np.float64) - self.mean_
        ) @ self.directions_
        level = projected.mean(axis=1)
        speed = np.diff(projected, axis=1)
        energy = np.sqrt(np.mean(speed * speed, axis=1))
        return np.concatenate([level, energy], axis=1)

    def __repr__(self) -> str:
        if not hasattr(self, "directions_"):
            return "SlowFeatureIndex(unfitted)"
        return f"SlowFeatureIndex(n_slow={self.directions_.shape[1]})"


def _slow_directions(
    covariance: np.ndarray, delta_covariance: np.ndarray, rank: int
) -> np.ndarray:
    eigenvalues, vectors = np.linalg.eigh(covariance)
    ceiling = float(eigenvalues[-1]) if len(eigenvalues) else 0.0
    floor = 1e-6 * ceiling if ceiling > 0 else 1e-8
    keep = eigenvalues > floor
    if not np.any(keep):
        keep = np.zeros(len(eigenvalues), dtype=bool)
        keep[-1] = True
    whitener = vectors[:, keep] * (1.0 / np.sqrt(eigenvalues[keep]))
    slowness, basis = np.linalg.eigh(whitener.T @ delta_covariance @ whitener)
    count = min(rank, basis.shape[1])
    order = np.argsort(slowness)[:count]
    return whitener @ basis[:, order]


def _pairs(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = windows[:, :-1, :].reshape(-1, windows.shape[2])
    delta = windows[:, 1:, :].reshape(-1, windows.shape[2]) - left
    return left, delta


def _as_windows(X: np.ndarray) -> np.ndarray:
    windows = np.asarray(X, dtype=np.float64)
    if windows.ndim != 3:
        raise ValueError(
            "X должен иметь форму (n_windows, window_size, n_channels), "
            f"получено {windows.shape}."
        )
    if len(windows) and not np.isfinite(windows).all():
        raise ValueError("Окна содержат NaN или бесконечность.")
    return windows
