"""Канонические переменные: новизна как срыв нормальной динамики."""

from __future__ import annotations

import numpy as np

from industrial_novelty.models.base import NoveltyDetector
from industrial_novelty.models.subspace import (
    ShrunkMahalanobis,
    _leading_components,
    ledoit_wolf,
    tail_scale,
)


class _ScaledBlock:
    """Центрирование, масштаб и усечение блока, если его размерность велика."""

    def fit_transform(self, Z: np.ndarray, max_dim: int, seed: int) -> np.ndarray:
        values = np.asarray(Z, dtype=np.float64)
        self.mean_ = values.mean(axis=0)
        scale = values.std(axis=0)
        self.scale_ = np.where(scale < 1e-8, 1.0, scale)
        scaled = (values - self.mean_) / self.scale_
        if scaled.shape[1] <= max_dim:
            self.components_ = None
            self.train_leftover_ = np.zeros(len(values), dtype=np.float64)
            return scaled
        _singular, components = _leading_components(
            scaled, max_dim, n_iter=2, seed=seed
        )
        self.components_ = components
        coords = scaled @ components
        leftover = np.sum(scaled * scaled, axis=1) - np.sum(coords * coords, axis=1)
        self.train_leftover_ = np.maximum(leftover, 0.0)
        return coords

    def transform(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scaled = (np.asarray(Z, dtype=np.float64) - self.mean_) / self.scale_
        if self.components_ is None:
            return scaled, np.zeros(len(scaled), dtype=np.float64)
        coords = scaled @ self.components_
        leftover = np.sum(scaled * scaled, axis=1) - np.sum(coords * coords, axis=1)
        return coords, np.maximum(leftover, 0.0)


class CanonicalVariateIndex(NoveltyDetector):
    """Условное расстояние Махаланобиса будущего при известном прошлом.

    Окно делится пополам. Прошлое ``p`` и будущее ``f`` считаются совместно
    нормальными после стандартизации координат. При исправной работе

    ``f | p ~ N(B p, Σ_{f|p})``.

    Оценка складывается из двух квадратичных членов, каждый в единицах своего
    обучающего хвоста:

    - расстояние Махаланобиса прошлого — процесс ушёл из обычной области;
    - расстояние Махаланобиса остатка ``f - B p`` — нарушилась сама динамика.

    ``B`` и обе ковариации оцениваются с усадкой Ледуа–Вулфа, поэтому мелкие
    собственные числа не взрывают расстояние. Если половина окна шире
    ``max_block_dim``, блок сначала проецируется на ведущие компоненты, а
    отброшенная энергия добавляется отдельным слагаемым.
    """

    def __init__(
        self,
        *,
        alpha: float = 0.01,
        max_block_dim: int = 768,
        seed: int = 0,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha должен лежать в интервале (0, 1).")
        if max_block_dim < 1:
            raise ValueError("max_block_dim должен быть >= 1.")
        self.alpha = float(alpha)
        self.max_block_dim = int(max_block_dim)
        self.seed = int(seed)

    def fit_windows(self, X: np.ndarray) -> None:
        past, future = _past_future(X)
        self.past_block_ = _ScaledBlock()
        self.future_block_ = _ScaledBlock()
        past_coords = self.past_block_.fit_transform(
            past, self.max_block_dim, self.seed
        )
        future_coords = self.future_block_.fit_transform(
            future, self.max_block_dim, self.seed + 1
        )
        self.past_center_ = past_coords.mean(axis=0)
        self.future_center_ = future_coords.mean(axis=0)
        past_centered = past_coords - self.past_center_
        future_centered = future_coords - self.future_center_
        covariance, self.past_shrinkage_ = ledoit_wolf(
            past_centered, assume_centered=True
        )
        eigenvalues, vectors = np.linalg.eigh(covariance)
        ceiling = float(np.max(eigenvalues)) if len(eigenvalues) else 0.0
        floor = 1e-8 * ceiling if ceiling > 0 else 1e-12
        safe = np.maximum(eigenvalues, floor)
        cross = (past_centered.T @ future_centered) / len(past_centered)
        self.regression_ = (vectors * (1.0 / safe)) @ (vectors.T @ cross)
        residual = future_centered - past_centered @ self.regression_
        self.past_metric_ = ShrunkMahalanobis().fit(past_coords)
        self.innovation_metric_ = ShrunkMahalanobis().fit(residual)
        self.past_scale_ = tail_scale(self.past_metric_.score(past_coords), self.alpha)
        self.innovation_scale_ = tail_scale(
            self.innovation_metric_.score(residual), self.alpha
        )
        self._fit_leftover_scales()

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "regression_"):
            raise RuntimeError("Сначала вызовите fit.")
        past, future = _past_future(X)
        past_coords, past_leftover = self.past_block_.transform(past)
        future_coords, future_leftover = self.future_block_.transform(future)
        residual = (future_coords - self.future_center_) - (
            past_coords - self.past_center_
        ) @ self.regression_
        score = (
            self.past_metric_.score(past_coords) / self.past_scale_
            + self.innovation_metric_.score(residual) / self.innovation_scale_
        )
        if self.past_leftover_scale_ is not None:
            score = score + past_leftover / self.past_leftover_scale_
        if self.future_leftover_scale_ is not None:
            score = score + future_leftover / self.future_leftover_scale_
        return score

    def _fit_leftover_scales(self) -> None:
        self.past_leftover_scale_ = _leftover_scale(self.past_block_, self.alpha)
        self.future_leftover_scale_ = _leftover_scale(self.future_block_, self.alpha)

    def __repr__(self) -> str:
        if not hasattr(self, "regression_"):
            return "CanonicalVariateIndex(unfitted)"
        return (
            "CanonicalVariateIndex("
            f"past_dim={self.regression_.shape[0]}, "
            f"future_dim={self.regression_.shape[1]}, "
            f"past_shrinkage={self.past_shrinkage_:.3f})"
        )


def _leftover_scale(block: _ScaledBlock, alpha: float) -> float | None:
    if block.components_ is None:
        return None
    return tail_scale(block.train_leftover_, alpha)


def _past_future(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    windows = np.asarray(X)
    if windows.ndim != 3:
        raise ValueError(
            "X должен иметь форму (n_windows, window_size, n_channels), "
            f"получено {windows.shape}."
        )
    if windows.shape[1] < 2:
        raise ValueError("Каноническим переменным нужно окно длины хотя бы 2.")
    if len(windows) and not np.isfinite(windows).all():
        raise ValueError("Окна содержат NaN или бесконечность.")
    mid = int(windows.shape[1] // 2)
    n_channels = int(windows.shape[2])
    if len(windows) == 0:
        past = np.empty((0, mid * n_channels), dtype=windows.dtype)
        future = np.empty(
            (0, (windows.shape[1] - mid) * n_channels), dtype=windows.dtype
        )
        return past, future
    past = windows[:, :mid, :].reshape(len(windows), -1)
    future = windows[:, mid:, :].reshape(len(windows), -1)
    return past, future
