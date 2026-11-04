"""Свёрточные признаки классификации рядов и тот же квадратичный индекс."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from industrial_novelty.models.base import NoveltyDetector
from industrial_novelty.models.subspace import CalibratedSubspace

_N_KERNELS = 84
_KERNEL_LENGTH = 9


def _kernel_matrix() -> np.ndarray:
    """84 ядра длины 9 с весами ``{-1, 2}`` и нулевой суммой.

    Три позиции веса 2 и шесть позиций веса −1 дают единственную длину,
    на которой такой набор центрирован: ``3 * 2 + 6 * (-1) = 0``.
    Отклик не зависит от постоянной составляющей канала.
    """
    weights = np.full((_N_KERNELS, _KERNEL_LENGTH), -1.0, dtype=np.float32)
    for row, positions in enumerate(combinations(range(_KERNEL_LENGTH), 3)):
        weights[row, positions] = 2.0
    return weights


def _golden_quantiles(n_levels: int) -> np.ndarray:
    """Низкодисперсная последовательность квантилей MiniRocket."""
    phi = (np.sqrt(5.0) + 1.0) / 2.0
    return np.array([(index * phi) % 1.0 for index in range(1, n_levels + 1)])


def _fit_dilations(
    length: int, n_features: int, max_dilations_per_kernel: int
) -> tuple[np.ndarray, np.ndarray]:
    """Расписание дилатаций MiniRocket: логарифмическая сетка до длины ряда."""
    per_kernel = n_features // _N_KERNELS
    if per_kernel < 1:
        raise ValueError(f"n_features должен быть не меньше {_N_KERNELS}.")
    n_candidates = min(per_kernel, max_dilations_per_kernel)
    max_exponent = np.log2((length - 1) / (_KERNEL_LENGTH - 1))
    if max_exponent < 0:
        raise ValueError(
            f"Окно длины {length} короче ядра {_KERNEL_LENGTH}. "
            "Для свёрточных признаков нужно хотя бы 10 отсчётов: "
            "девять на уровень ряда и ещё один на первую разность."
        )
    grid = np.logspace(0.0, float(max_exponent), n_candidates, base=2.0)
    dilations, counts = np.unique(grid.astype(np.int32), return_counts=True)
    multiplier = per_kernel / n_candidates
    counts = (counts.astype(np.float64) * multiplier).astype(np.int32)
    remainder = per_kernel - int(counts.sum())
    position = 0
    while remainder > 0:
        counts[position] += 1
        remainder -= 1
        position = (position + 1) % len(counts)
    return dilations, counts


class KernelBank:
    """Признаки MiniRocket по одному представлению окна.

    Для каждой дилатации и каждого из 84 ядер берётся случайная сумма
    не более чем девяти каналов, считается свёртка без дополнения, и по
    квантилям нормального класса считаются две статистики MultiRocket:
    доля положительных откликов (PPV) и их среднее (MPV).

    От опубликованного MiniRocket здесь два сознательных отличия.
    Пороги — квантили подвыборки всего нормального класса, а не одной
    случайной серии: для новизны нужен устойчивый эталон. Одна и та же
    свёртка без дополнения используется и при обучении порогов, и при счёте.
    """

    def __init__(
        self,
        *,
        n_features: int = 1680,
        max_dilations_per_kernel: int = 32,
        max_bias_windows: int = 512,
        seed: int = 0,
        chunk_size: int = 2048,
    ) -> None:
        if n_features < _N_KERNELS:
            raise ValueError(f"n_features должен быть не меньше {_N_KERNELS}.")
        if max_dilations_per_kernel < 1:
            raise ValueError("max_dilations_per_kernel должен быть >= 1.")
        if max_bias_windows < 1 or chunk_size < 1:
            raise ValueError("max_bias_windows и chunk_size должны быть >= 1.")
        self.n_features = int(n_features // _N_KERNELS * _N_KERNELS)
        self.max_dilations_per_kernel = int(max_dilations_per_kernel)
        self.max_bias_windows = int(max_bias_windows)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.kernels_ = _kernel_matrix()

    def fit(self, X: np.ndarray) -> KernelBank:
        windows = _as_windows(X)
        if len(windows) == 0:
            raise ValueError("Нет окон, чтобы оценить пороги свёрток.")
        self.length_ = int(windows.shape[1])
        self.n_channels_ = int(windows.shape[2])
        dilations, counts = _fit_dilations(
            self.length_, self.n_features, self.max_dilations_per_kernel
        )
        self.dilations_ = dilations
        self.bias_counts_ = counts
        rng = np.random.default_rng(self.seed)
        self.channel_coeff_ = _channel_coefficients(
            rng, len(dilations), self.n_channels_
        )
        sample_size = min(len(windows), self.max_bias_windows)
        sample_ids = rng.choice(len(windows), size=sample_size, replace=False)
        sample = np.ascontiguousarray(windows[sample_ids], dtype=np.float32)
        quantiles = _golden_quantiles(self.n_features)
        self.biases_: list[np.ndarray] = []
        cursor = 0
        for dilation_index, n_bias in enumerate(int(count) for count in counts):
            convolution = self._convolve(sample, dilation_index)
            # Порядок признаков: дилатация, затем ядро, затем пороги этого ядра.
            levels = quantiles[cursor : cursor + _N_KERNELS * n_bias]
            levels = levels.reshape(_N_KERNELS, n_bias)
            flat = convolution.reshape(-1, _N_KERNELS)
            bias = np.empty((_N_KERNELS, n_bias), dtype=np.float32)
            for kernel in range(_N_KERNELS):
                bias[kernel] = np.quantile(flat[:, kernel], levels[kernel])
            self.biases_.append(bias)
            cursor += _N_KERNELS * n_bias
        self.n_outputs_ = int(sum(int(count) for count in counts) * _N_KERNELS * 2)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "biases_"):
            raise RuntimeError("Сначала вызовите fit.")
        windows = _as_windows(X)
        if windows.shape[1] != self.length_ or windows.shape[2] != self.n_channels_:
            raise ValueError(
                "Форма окон не совпадает с обучением: "
                f"ожидалось ({self.length_}, {self.n_channels_}), "
                f"получено {windows.shape[1:]}."
            )
        if len(windows) == 0:
            return np.empty((0, self.n_outputs_), dtype=np.float32)
        parts = [
            self._transform_chunk(windows[start : start + self.chunk_size])
            for start in range(0, len(windows), self.chunk_size)
        ]
        return np.concatenate(parts, axis=0)

    def _transform_chunk(self, windows: np.ndarray) -> np.ndarray:
        chunk = np.ascontiguousarray(windows, dtype=np.float32)
        ppv_parts: list[np.ndarray] = []
        mpv_parts: list[np.ndarray] = []
        for dilation_index in range(len(self.dilations_)):
            ppv, mpv = _pool(
                self._convolve(chunk, dilation_index), self.biases_[dilation_index]
            )
            ppv_parts.append(ppv)
            mpv_parts.append(mpv)
        return np.concatenate(ppv_parts + mpv_parts, axis=1)

    def _convolve(self, windows: np.ndarray, dilation_index: int) -> np.ndarray:
        dilation = int(self.dilations_[dilation_index])
        mixed = np.einsum(
            "ntc,kc->ntk",
            windows,
            self.channel_coeff_[dilation_index],
            optimize=True,
        )
        length = self.length_ - (_KERNEL_LENGTH - 1) * dilation
        offsets = np.arange(_KERNEL_LENGTH, dtype=np.int32) * dilation
        index = np.arange(length, dtype=np.int32)[:, None] + offsets[None, :]
        patches = mixed[:, index, :]
        return np.einsum("nlak,ka->nlk", patches, self.kernels_, optimize=True)


class ConvolutionalKernelIndex(NoveltyDetector):
    """Индекс T²+Q в пространстве свёрточных признаков.

    Линейный классификатор, которым MiniRocket и MultiRocket заканчивают
    обучение, здесь невозможен: отрицательного класса нет. Его заменяет
    совместный индекс нормального облака в том же пространстве признаков.

    Вторая половина признаков считается по первой разности окна. Так
    разделяются уровень сигнала и его приращение: это основное дополнение
    MultiRocket к MiniRocket, и оно нужно отказам, которые меняют дисперсию,
    не сдвигая среднее.

    Постоянный сдвиг канала эта оценка не видит. Сумма весов ядра равна нулю,
    а первая разность нового установившегося уровня тоже нулевая. Сдвиг уровня
    и размах внутри окна отдаются :class:`WindowMomentIndex`.
    """

    def __init__(
        self,
        *,
        n_features: int = 1680,
        use_differences: bool = True,
        variance: float = 0.9,
        max_rank: int = 48,
        alpha: float = 0.01,
        seed: int = 0,
        chunk_size: int = 2048,
    ) -> None:
        self.use_differences = bool(use_differences)
        self.level_bank = KernelBank(
            n_features=n_features, seed=seed, chunk_size=chunk_size
        )
        self.difference_bank = (
            KernelBank(n_features=n_features, seed=seed + 1, chunk_size=chunk_size)
            if self.use_differences
            else None
        )
        self.head = CalibratedSubspace(
            variance=variance,
            max_rank=max_rank,
            standardize=True,
            alpha=alpha,
            seed=seed,
        )
        self.chunk_size = int(chunk_size)

    def fit_windows(self, X: np.ndarray) -> None:
        windows = _as_windows(X)
        _require_kernel_length(windows.shape[1], differences=self.use_differences)
        self.level_bank.fit(windows)
        parts = [self.level_bank.transform(windows)]
        if self.difference_bank is not None:
            diffs = np.diff(windows, axis=1)
            self.difference_bank.fit(diffs)
            parts.append(self.difference_bank.transform(diffs))
        self.head.fit(np.concatenate(parts, axis=1))

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self.head, "t2_scale_"):
            raise RuntimeError("Сначала вызовите fit.")
        windows = _as_windows(X)
        if len(windows) == 0:
            return np.empty(0, dtype=np.float64)
        scores = np.empty(len(windows), dtype=np.float64)
        for start in range(0, len(windows), self.chunk_size):
            chunk = windows[start : start + self.chunk_size]
            scores[start : start + len(chunk)] = self.head.score(self._features(chunk))
        return scores

    @property
    def rank_(self) -> int:
        return self.head.subspace.rank_

    @property
    def n_features_(self) -> int:
        width = self.level_bank.n_outputs_
        if self.difference_bank is not None:
            width += self.difference_bank.n_outputs_
        return width

    def _features(self, windows: np.ndarray) -> np.ndarray:
        parts = [self.level_bank.transform(windows)]
        if self.difference_bank is not None:
            parts.append(self.difference_bank.transform(np.diff(windows, axis=1)))
        return np.concatenate(parts, axis=1)

    def __repr__(self) -> str:
        if not hasattr(self.head, "t2_scale_"):
            return "ConvolutionalKernelIndex(unfitted)"
        return (
            f"ConvolutionalKernelIndex(n_features={self.n_features_}, "
            f"rank={self.rank_}, differences={self.use_differences})"
        )


def _channel_coefficients(
    rng: np.random.Generator, n_dilations: int, n_channels: int
) -> np.ndarray:
    coeff = np.zeros((n_dilations, _N_KERNELS, n_channels), dtype=np.float32)
    max_channels = min(n_channels, 9)
    max_exponent = np.log2(max_channels + 1)
    for dilation in range(n_dilations):
        for kernel in range(_N_KERNELS):
            drawn = int(2 ** rng.uniform(0.0, max_exponent))
            n_pick = min(max(drawn, 1), max_channels)
            chosen = rng.choice(n_channels, size=n_pick, replace=False)
            coeff[dilation, kernel, chosen] = 1.0
    return coeff


def _pool(convolution: np.ndarray, biases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_rows, length, n_kernels = convolution.shape
    n_bias = biases.shape[1]
    ppv = np.empty((n_rows, n_kernels * n_bias), dtype=np.float32)
    mpv = np.empty((n_rows, n_kernels * n_bias), dtype=np.float32)
    step = 12
    for start in range(0, n_kernels, step):
        stop = min(start + step, n_kernels)
        block = convolution[:, :, start:stop]
        bias = biases[start:stop]
        above = block[:, :, :, None] > bias[None, None, :, :]
        count = above.sum(axis=1, dtype=np.float32)
        width = stop - start
        view = slice(start * n_bias, stop * n_bias)
        ppv[:, view] = (count / length).reshape(n_rows, width * n_bias)
        weighted = (block[:, :, :, None] * above).sum(axis=1)
        values = np.divide(
            weighted, count, out=np.zeros_like(weighted), where=count > 0
        )
        mpv[:, view] = values.reshape(n_rows, width * n_bias)
    return ppv, mpv


def _require_kernel_length(length: int, *, differences: bool) -> None:
    needed = _KERNEL_LENGTH + (1 if differences else 0)
    if length < needed:
        raise ValueError(
            f"Свёрточным признакам нужно окно длины хотя бы {needed}, получено {length}."
        )


def _as_windows(X: np.ndarray) -> np.ndarray:
    windows = np.asarray(X)
    if windows.ndim != 3:
        raise ValueError(
            "X должен иметь форму (n_windows, window_size, n_channels), "
            f"получено {windows.shape}."
        )
    if len(windows) and not np.isfinite(windows).all():
        raise ValueError("Окна содержат NaN или бесконечность.")
    return windows
