"""Контракт модели: нормальные окна на вход, одна оценка новизны на выход."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NoveltyDetector(ABC):
    """Общий интерфейс одноклассовой модели.

    Наследник реализует :meth:`fit_windows` и :meth:`anomaly_score`.
    ``fit`` и ``score`` проверяют форму, их не нужно переопределять.

    ``X`` имеет форму ``(n_windows, window_size, n_channels)``.
    Оценка тем больше, чем сильнее окно отличается от нормальной работы.
    """

    def fit(self, X: np.ndarray) -> NoveltyDetector:
        windows = check_windows(X, allow_empty=False)
        self.fit_windows(windows)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        windows = check_windows(X, allow_empty=True)
        scores = np.asarray(self.anomaly_score(windows), dtype=np.float64).reshape(-1)
        if len(scores) != len(windows):
            raise ValueError(
                f"anomaly_score вернул {len(scores)} значений на {len(windows)} окон."
            )
        if len(scores) and not np.isfinite(scores).all():
            raise ValueError("Оценки содержат NaN или бесконечность.")
        return scores

    def fit_windows(self, X: np.ndarray) -> None:
        """Запомнить нормальные окна. По умолчанию у модели нет состояния."""

    @abstractmethod
    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Вернуть по одному числу на окно. Больше — аномальнее."""


def check_windows(X: np.ndarray, *, allow_empty: bool) -> np.ndarray:
    windows = np.asarray(X)
    if windows.ndim != 3:
        raise ValueError(
            "X должен иметь форму (n_windows, window_size, n_channels), "
            f"получено {getattr(X, 'shape', None)}."
        )
    if not allow_empty and len(windows) == 0:
        raise ValueError("Нет окон для обучения.")
    return windows
