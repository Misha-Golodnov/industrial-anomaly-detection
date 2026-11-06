"""Статистические кандидаты из обзора Lacuna; обучение только на норме."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from industrial_novelty.models.base import NoveltyDetector, check_windows
from industrial_novelty.models.subspace import ShrunkMahalanobis, tail_scale


def _windows(X: np.ndarray) -> np.ndarray:
    values = check_windows(X, allow_empty=True)
    if values.shape[1] < 8 or values.shape[2] < 1:
        raise ValueError("Нужно хотя бы 8 отсчётов и один канал.")
    if not np.isfinite(values).all():
        raise ValueError("Окна должны быть конечными.")
    return np.asarray(values, dtype=np.float64)


class _FeatureIndex(NoveltyDetector):
    def fit_windows(self, X: np.ndarray) -> None:
        values = _windows(X)
        self.shape_ = values.shape[1:]
        self._prepare(values)
        features = self.features(values)
        self.feature_mean_ = features.mean(axis=0)
        self.feature_scale_ = np.maximum(features.std(axis=0), 1e-6)
        standardized = (features - self.feature_mean_) / self.feature_scale_
        self.metric_ = ShrunkMahalanobis().fit(standardized)
        self.tail_ = tail_scale(self.metric_.score(standardized), 0.01)

    def _prepare(self, X: np.ndarray) -> None:
        pass

    def features(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "tail_"):
            raise RuntimeError("Сначала вызовите fit.")
        values = _windows(X)
        if values.shape[1:] != self.shape_:
            raise ValueError("Форма окна отличается от обучающей.")
        if not len(values):
            return np.empty(0)
        features = (self.features(values) - self.feature_mean_) / self.feature_scale_
        return self.metric_.score(features) / self.tail_


class WindowDependenceIndex(_FeatureIndex):
    """Локальные ковариации выбранных по норме пар, лаги 0/1/3.

    Использует половину и полное окно; не обращает ковариацию внутри окна.
    Это статистическая адаптация signature matrices MSCRED, не ConvLSTM.
    """

    def __init__(self, max_pairs: int = 64, *, centered: bool = True) -> None:
        if max_pairs < 1:
            raise ValueError("max_pairs должен быть положительным.")
        self.max_pairs = int(max_pairs)
        self.centered = bool(centered)

    def _prepare(self, X: np.ndarray) -> None:
        sample = X[np.linspace(0, len(X) - 1, min(len(X), 2048), dtype=int)]
        sample = sample - sample.mean(axis=1, keepdims=True)
        flat = sample.reshape(-1, sample.shape[2])
        gram = flat.T @ flat / len(flat)
        scale = np.sqrt(np.maximum(np.diag(gram), 1e-12))
        correlation = gram / np.outer(scale, scale)
        left, right = np.triu_indices(X.shape[2], k=1)
        if not len(left):
            left, right = np.array([0]), np.array([0])
        order = np.argsort(-np.abs(correlation[left, right]), kind="stable")
        chosen = order[: self.max_pairs]
        self.left_, self.right_ = left[chosen], right[chosen]

    def features(self, X: np.ndarray) -> np.ndarray:
        values = _windows(X)
        width = len(self.left_) * 6
        result = np.empty((len(values), width))
        for start in range(0, len(values), 1024):
            chunk = values[start : start + 1024]
            parts = []
            for length in (values.shape[1] // 2, values.shape[1]):
                block = chunk[:, -length:]
                if self.centered:
                    block = block - block.mean(axis=1, keepdims=True)
                for lag in (0, 1, 3):
                    a = block[:, : length - lag, self.left_]
                    b = block[:, lag:, self.right_]
                    parts.append(np.mean(a * b, axis=1))
            result[start : start + len(chunk)] = np.concatenate(parts, axis=1)
        return result


class WindowSpectrumIndex(_FeatureIndex):
    """Энергии трёх полос и автокорреляции на двух масштабах окна.

    Ветвь инвариантна к уровню: при объединении уровень хранит WindowMomentIndex.
    Это адаптация принципа time/frequency TFAD, не воспроизведение TFAD.
    """

    def features(self, X: np.ndarray) -> np.ndarray:
        values = _windows(X)
        if values.shape[1] < 12:
            raise ValueError("Для трёх спектральных полос нужно окно длины >= 12.")
        result = np.empty((len(values), 12 * values.shape[2]))
        for start in range(0, len(values), 1024):
            chunk = values[start : start + 1024]
            parts = []
            for length in (values.shape[1] // 2, values.shape[1]):
                block = chunk[:, -length:]
                block = block - block.mean(axis=1, keepdims=True)
                taper = np.hanning(length)[None, :, None]
                power = np.abs(np.fft.rfft(block * taper, axis=1)) ** 2 / length
                for band in np.array_split(np.arange(1, power.shape[1]), 3):
                    parts.append(np.log(np.mean(power[:, band], axis=1) + 1e-8))
                for lag in (1, 2, 3):
                    left, right = block[:, :-lag], block[:, lag:]
                    numerator = np.mean(left * right, axis=1)
                    denominator = np.sqrt(
                        np.mean(left * left, axis=1) * np.mean(right * right, axis=1)
                    )
                    parts.append(numerator / np.maximum(denominator, 1e-8))
            result[start : start + len(chunk)] = np.concatenate(parts, axis=1)
        return result


def history_mean(
    values: np.ndarray, runs: np.ndarray, ends: np.ndarray, history: int
) -> np.ndarray:
    """Причинное среднее последних history окон; состояние отдельно у прогонов."""
    if history < 1:
        raise ValueError("history должен быть положительным.")
    if len(values) != len(runs) or len(values) != len(ends):
        raise ValueError("Длины значений и паспорта различаются.")
    result = np.empty_like(values, dtype=np.float64)
    order = np.lexsort((ends, runs))
    if not len(order):
        return result
    boundaries = np.r_[0, np.flatnonzero(np.diff(runs[order])) + 1, len(order)]
    for left, right in pairwise(boundaries):
        positions = order[left:right]
        sequence = values[positions]
        cumulative = np.vstack([np.zeros((1, sequence.shape[1])), sequence.cumsum(0)])
        stop = np.arange(1, len(sequence) + 1)
        begin = np.maximum(0, stop - history)
        result[positions] = (cumulative[stop] - cumulative[begin]) / (stop - begin)[
            :, None
        ]
    return result


class KernelMMDIndex(NoveltyDetector):
    """MMD² текущей истории с нормальным эталоном через random Fourier features.

    Смесь трёх RBF-ядер, 128 признаков на ядро. Референс фиксирован после fit.
    Это приближённая MMD с эмпирической калибровкой, не точный M-statistic статьи.
    """

    def __init__(self, history: int = 4, *, n_features: int = 128, seed: int = 0):
        if history < 1 or n_features < 2:
            raise ValueError("Нужно history >= 1 и n_features >= 2.")
        self.history = int(history)
        self.n_features = int(n_features)
        self.seed = int(seed)

    @staticmethod
    def features(X: np.ndarray) -> np.ndarray:
        values = _windows(X)
        return np.concatenate(
            [values.mean(1), values.std(1), np.diff(values, axis=1).std(1)], axis=1
        )

    def fit_windows(self, X: np.ndarray) -> None:
        self.shape_ = X.shape[1:]
        features = self.features(X)
        self.mean_ = features.mean(0)
        self.scale_ = np.maximum(features.std(0), 1e-6)
        z = (features - self.mean_) / self.scale_
        rng = np.random.default_rng(self.seed)
        sample = z[rng.choice(len(z), min(len(z), 512), replace=False)]
        squares = np.sum(sample * sample, axis=1)
        distances = np.maximum(
            squares[:, None] + squares[None, :] - 2 * sample @ sample.T, 0
        )
        positive = distances[np.triu_indices(len(sample), 1)]
        positive = positive[positive > 1e-10]
        bandwidth = np.sqrt(np.median(positive)) if len(positive) else 1.0
        self.bandwidth_ = float(bandwidth)
        self.weights_ = np.concatenate(
            [
                rng.normal(size=(z.shape[1], self.n_features)) / (bandwidth * factor)
                for factor in (0.5, 1.0, 2.0)
            ],
            axis=1,
        )
        self.phase_ = rng.uniform(0, 2 * np.pi, self.weights_.shape[1])
        self.reference_ = self._embedding(features).mean(0)

    def _embedding(self, features: np.ndarray) -> np.ndarray:
        z = (features - self.mean_) / self.scale_
        return np.sqrt(2 / self.weights_.shape[1]) * np.cos(
            z @ self.weights_ + self.phase_
        )

    def _transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "reference_"):
            raise RuntimeError("Сначала вызовите fit.")
        if X.shape[1:] != self.shape_:
            raise ValueError("Форма окна отличается от обучающей.")
        return self._embedding(self.features(X))

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        phi = self._transform(X)
        means = history_mean(
            phi, np.zeros(len(phi), dtype=int), np.arange(len(phi)), self.history
        )
        return np.sum((means - self.reference_) ** 2, axis=1)

    def score_batch(self, batch) -> np.ndarray:
        phi = self._transform(batch.X)
        means = history_mean(
            phi,
            batch.meta["simulation_run"].to_numpy(),
            batch.meta["sample_end"].to_numpy(),
            self.history,
        )
        return np.sum((means - self.reference_) ** 2, axis=1)
