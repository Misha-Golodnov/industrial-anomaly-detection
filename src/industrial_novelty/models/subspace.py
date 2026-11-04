"""Квадратичные оценки новизны: усечённый индекс T²+Q и усадка Ледуа–Вулфа."""

from __future__ import annotations

import numpy as np

from industrial_novelty.models.base import NoveltyDetector


def tail_scale(scores: np.ndarray, alpha: float) -> float:
    """Вернуть эмпирический квантиль уровня ``1 - alpha``.

    Это единица хвоста нормального класса, а не порог решения.
    Порог по доле ложных тревог ставится снаружи, на отложенных прогонах.
    Нулевой квантиль заменяется максимумом, чтобы постоянная оценка
    не получала бесконечный вес.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise ValueError("Нельзя откалибровать пустой набор оценок.")
    if not np.isfinite(values).all():
        raise ValueError("Оценки для калибровки должны быть конечными.")
    if not 0 < alpha < 1:
        raise ValueError("alpha должен лежать в интервале (0, 1).")
    scale = float(np.quantile(values, 1.0 - alpha))
    if scale <= 1e-12:
        scale = float(np.max(values))
    if scale <= 1e-12:
        scale = 1.0
    return scale


def ledoit_wolf(
    X: np.ndarray, *, assume_centered: bool = False
) -> tuple[np.ndarray, float]:
    """Ковариация Ледуа–Вулфа и коэффициент усадки к кратной единичной матрице.

    Оценка имеет вид ``(1 - ρ) S + ρ μ I``, где ``S`` — выборочная ковариация
    с делителем ``n``, ``μ = tr(S) / p``, а ``ρ`` минимизирует среднеквадратичную
    ошибку в режиме большой размерности. Формула ``ρ`` совпадает с реализацией
    scikit-learn (Ledoit and Wolf, 2004).
    """
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Ожидалась матрица (n, p), получено {data.shape}.")
    if len(data) == 0:
        raise ValueError("Нельзя оценить ковариацию по пустой матрице.")
    if not assume_centered:
        data = data - data.mean(axis=0)
    n_samples, n_features = data.shape
    if n_features == 1:
        variance = float(np.sum(data * data) / n_samples)
        return np.array([[variance]], dtype=np.float64), 0.0

    squares = data * data
    diagonal = np.sum(squares, axis=0) / n_samples
    mu = float(np.sum(diagonal) / n_features)
    beta_raw = float(np.sum(squares.T @ squares))
    delta_raw = float(np.sum((data.T @ data) ** 2) / n_samples**2)
    beta = (beta_raw / n_samples - delta_raw) / (n_features * n_samples)
    delta = (
        delta_raw - 2.0 * mu * float(np.sum(diagonal)) + n_features * mu**2
    ) / n_features
    if delta <= 0.0 or beta <= 0.0:
        shrinkage = 0.0
    else:
        shrinkage = float(min(beta, delta) / delta)
    covariance = (data.T @ data) / n_samples
    covariance *= 1.0 - shrinkage
    covariance.flat[:: n_features + 1] += shrinkage * mu
    return covariance, shrinkage


def _leading_components(
    Y: np.ndarray, rank: int, *, n_iter: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Сингулярные числа и правые сингулярные векторы ведущего ранга ``Y``."""
    matrix = np.ascontiguousarray(Y)
    n_rows, n_cols = matrix.shape
    rank = min(int(rank), n_rows, n_cols)
    if rank < 1:
        raise ValueError("Ранг должен быть не меньше 1.")
    if n_cols <= 1500 and n_rows >= n_cols:
        gram = np.asarray(matrix.T @ matrix, dtype=np.float64)
        eigenvalues, vectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1][:rank]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)
        return np.sqrt(eigenvalues), vectors[:, order]
    if n_rows <= 1500 and n_cols > n_rows:
        _left, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        return singular[:rank], vt[:rank].T

    rng = np.random.default_rng(seed)
    sketch = min(rank + 8, n_rows, n_cols)
    omega = rng.standard_normal((n_cols, sketch)).astype(matrix.dtype, copy=False)
    basis, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(n_iter):
        basis, _ = np.linalg.qr(matrix.T @ basis, mode="reduced")
        basis, _ = np.linalg.qr(matrix @ basis, mode="reduced")
    _left, singular, vt = np.linalg.svd(basis.T @ matrix, full_matrices=False)
    return singular[:rank], vt[:rank].T


class PrincipalSubspace:
    """Ведущее подпространство корреляционной или ковариационной матрицы.

    ``T²`` — расстояние Махаланобиса внутри подпространства.
    ``Q`` — энергия остатка. Вместе это разложение квадратичной формы,
    из которого Юэ и Цинь собирают совместный индекс.
    """

    def __init__(
        self,
        *,
        variance: float = 0.9,
        max_rank: int = 64,
        min_rank: int = 1,
        standardize: bool = True,
        seed: int = 0,
    ) -> None:
        if not 0 < variance <= 1:
            raise ValueError("variance должен лежать в полуинтервале (0, 1].")
        if max_rank < 1 or min_rank < 1 or min_rank > max_rank:
            raise ValueError("Нужно 1 <= min_rank <= max_rank.")
        self.variance = float(variance)
        self.max_rank = int(max_rank)
        self.min_rank = int(min_rank)
        self.standardize = bool(standardize)
        self.seed = int(seed)

    def fit(self, Z: np.ndarray) -> PrincipalSubspace:
        features = _as_feature_matrix(Z)
        if len(features) < 2:
            raise ValueError("Для подпространства нужны хотя бы два окна.")
        self.mean_ = features.mean(axis=0).astype(np.float64, copy=False)
        if self.standardize:
            scale = features.std(axis=0).astype(np.float64, copy=False)
            scale = np.where(scale < 1e-8, 1.0, scale)
        else:
            scale = np.ones(features.shape[1], dtype=np.float64)
        self.scale_ = scale
        centered = features - self.mean_
        scaled = centered / self.scale_
        if scaled.shape[0] * scaled.shape[1] <= 50_000_000:
            scaled = np.asarray(scaled, dtype=np.float64)
        total = float(np.sum(scaled * scaled, dtype=np.float64) / (len(scaled) - 1))
        self.total_variance_ = total
        if total <= 1e-12:
            self.components_ = np.zeros((features.shape[1], 1), dtype=np.float64)
            self.eigenvalues_ = np.ones(1, dtype=np.float64)
            return self

        singular, components = _leading_components(
            scaled,
            self.max_rank,
            n_iter=2,
            seed=self.seed,
        )
        eigenvalues = singular.astype(np.float64) ** 2 / (len(scaled) - 1)
        share = np.cumsum(eigenvalues) / total
        rank = int(np.searchsorted(share, self.variance, side="left") + 1)
        rank = min(max(rank, self.min_rank), len(eigenvalues))
        self.components_ = np.asarray(components[:, :rank], dtype=np.float64)
        kept = eigenvalues[:rank]
        floor = 1e-6 * float(kept[0]) if kept[0] > 0 else 1e-12
        self.eigenvalues_ = np.maximum(kept, floor)
        return self

    @property
    def rank_(self) -> int:
        return int(self.components_.shape[1])

    def pair(self, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Вернуть ``(T², Q)`` для каждой строки."""
        if not hasattr(self, "components_"):
            raise RuntimeError("Сначала вызовите fit.")
        features = _as_feature_matrix(Z)
        if features.shape[1] != len(self.mean_):
            raise ValueError(
                "Число признаков не совпадает с обучением: "
                f"ожидалось {len(self.mean_)}, получено {features.shape[1]}."
            )
        if len(features) == 0:
            empty = np.empty(0, dtype=np.float64)
            return empty, empty
        scaled = (features - self.mean_) / self.scale_
        coords = scaled @ self.components_
        hotelling = np.sum(coords * coords / self.eigenvalues_, axis=1)
        explained = np.sum(coords * coords, axis=1)
        total = np.sum(scaled * scaled, axis=1)
        residual = np.maximum(total - explained, 0.0)
        return hotelling.astype(np.float64, copy=False), residual.astype(
            np.float64, copy=False
        )


class CalibratedSubspace:
    """Совместный индекс ``T² / q(T²) + Q / q(Q)`` с эмпирическими квантилями."""

    def __init__(
        self,
        *,
        variance: float = 0.9,
        max_rank: int = 64,
        min_rank: int = 1,
        standardize: bool = True,
        alpha: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.alpha = float(alpha)
        self.subspace = PrincipalSubspace(
            variance=variance,
            max_rank=max_rank,
            min_rank=min_rank,
            standardize=standardize,
            seed=seed,
        )

    def fit(self, Z: np.ndarray) -> CalibratedSubspace:
        self.subspace.fit(Z)
        hotelling, residual = self.subspace.pair(Z)
        self.t2_scale_ = tail_scale(hotelling, self.alpha)
        self.q_scale_ = tail_scale(residual, self.alpha)
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        if not hasattr(self, "t2_scale_"):
            raise RuntimeError("Сначала вызовите fit.")
        hotelling, residual = self.subspace.pair(Z)
        return hotelling / self.t2_scale_ + residual / self.q_scale_


class ShrunkMahalanobis:
    """Полное расстояние Махаланобиса при обратимой усадке Ледуа–Вулфа."""

    def fit(self, Z: np.ndarray) -> ShrunkMahalanobis:
        features = _as_feature_matrix(Z)
        if len(features) < 2:
            raise ValueError("Для расстояния Махаланобиса нужны хотя бы два окна.")
        self.mean_ = features.mean(axis=0).astype(np.float64, copy=False)
        covariance, self.shrinkage_ = ledoit_wolf(
            features - self.mean_, assume_centered=True
        )
        eigenvalues, vectors = np.linalg.eigh(covariance)
        ceiling = float(np.max(eigenvalues)) if len(eigenvalues) else 0.0
        floor = 1e-8 * ceiling if ceiling > 0 else 1e-12
        self.eigenvalues_ = np.maximum(eigenvalues, floor)
        self.whitener_ = vectors * (1.0 / np.sqrt(self.eigenvalues_))
        return self

    def score(self, Z: np.ndarray) -> np.ndarray:
        if not hasattr(self, "whitener_"):
            raise RuntimeError("Сначала вызовите fit.")
        features = _as_feature_matrix(Z)
        if features.shape[1] != len(self.mean_):
            raise ValueError(
                "Число признаков не совпадает с обучением: "
                f"ожидалось {len(self.mean_)}, получено {features.shape[1]}."
            )
        if len(features) == 0:
            return np.empty(0, dtype=np.float64)
        whitened = (features - self.mean_) @ self.whitener_
        return np.sum(whitened * whitened, axis=1)


class WindowSubspaceIndex(NoveltyDetector):
    """Совместный индекс T² и Q по развёрнутому окну.

    Окно длины ``T`` — это лаговое наблюдение
    ``(x_t, x_{t-1}, ..., x_{t-T+1})``. PCA такого вектора есть динамический
    PCA Ку, Сторера и Георгакиса. Каналы уже приведены к единичному масштабу
    обучающим ``ChannelScaler``, поэтому второй раз признаки не стандартизируются:
    сохраняется ковариационная, а не корреляционная метрика.
    """

    def __init__(
        self,
        *,
        variance: float = 0.9,
        max_rank: int = 64,
        alpha: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.head = CalibratedSubspace(
            variance=variance,
            max_rank=max_rank,
            standardize=False,
            alpha=alpha,
            seed=seed,
        )

    def fit_windows(self, X: np.ndarray) -> None:
        self.head.fit(_flatten(X))

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return self.head.score(_flatten(X))

    @property
    def rank_(self) -> int:
        return self.head.subspace.rank_

    def __repr__(self) -> str:
        if not hasattr(self.head, "t2_scale_"):
            return "WindowSubspaceIndex(unfitted)"
        return (
            f"WindowSubspaceIndex(rank={self.rank_}, "
            f"variance={self.head.subspace.variance})"
        )


class WindowMomentIndex(NoveltyDetector):
    """Расстояние Махаланобиса среднего и размаха каналов внутри окна.

    Ядра MiniRocket имеют нулевую сумму, поэтому постоянный сдвиг канала
    не меняет их отклик. Для отказа, который уводит процесс в новую точку
    или раздувает шум, как раз нужны эти два момента. Вектор из ``2C`` чисел
    сравнивается с нормальным облаком через ковариацию Ледуа–Вулфа: сдвиг
    вдоль обычной корреляции датчиков получает меньший вес, чем сдвиг,
    который эту корреляцию ломает.
    """

    def __init__(self, *, alpha: float = 0.01) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha должен лежать в интервале (0, 1).")
        self.alpha = float(alpha)
        self.metric = ShrunkMahalanobis()

    def fit_windows(self, X: np.ndarray) -> None:
        features = window_moments(X)
        self.metric.fit(features)
        self.scale_ = tail_scale(self.metric.score(features), self.alpha)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "scale_"):
            raise RuntimeError("Сначала вызовите fit.")
        return self.metric.score(window_moments(X)) / self.scale_

    def __repr__(self) -> str:
        if not hasattr(self, "scale_"):
            return "WindowMomentIndex(unfitted)"
        return f"WindowMomentIndex(shrinkage={self.metric.shrinkage_:.3f})"


def window_moments(X: np.ndarray) -> np.ndarray:
    """Среднее и стандартное отклонение каждого канала, ось времени схлопнута."""
    windows = np.asarray(X)
    if windows.ndim != 3:
        raise ValueError(
            "X должен иметь форму (n_windows, window_size, n_channels), "
            f"получено {windows.shape}."
        )
    if len(windows) and not np.isfinite(windows).all():
        raise ValueError("Окна содержат NaN или бесконечность.")
    mean = windows.mean(axis=1, dtype=np.float64)
    std = windows.std(axis=1, dtype=np.float64)
    return np.concatenate([mean, std], axis=1)


def _flatten(X: np.ndarray) -> np.ndarray:
    windows = np.asarray(X)
    if windows.ndim != 3:
        raise ValueError(
            "X должен иметь форму (n_windows, window_size, n_channels), "
            f"получено {windows.shape}."
        )
    if len(windows) == 0:
        return np.empty(
            (0, int(windows.shape[1] * windows.shape[2])), dtype=windows.dtype
        )
    return windows.reshape(len(windows), -1)


def _as_feature_matrix(Z: np.ndarray) -> np.ndarray:
    features = np.asarray(Z)
    if features.ndim != 2:
        raise ValueError(
            f"Ожидалась матрица признаков (n, p), получено {features.shape}."
        )
    if len(features) and not np.isfinite(features).all():
        raise ValueError("Признаки содержат NaN или бесконечность.")
    return features
