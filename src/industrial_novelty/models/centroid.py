"""Простая модель, чтобы проверить связку данных и оценки до настоящих методов."""

from __future__ import annotations

import numpy as np

from industrial_novelty.models.base import NoveltyDetector


class CentroidDistance(NoveltyDetector):
    """Евклидово расстояние до среднего нормального окна.

    Это ориентир для инфраструктуры, а не метод для отчёта.
    """

    def fit_windows(self, X: np.ndarray) -> None:
        flat = X.reshape(len(X), -1).astype(np.float64, copy=False)
        self.center_ = flat.mean(axis=0)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "center_"):
            raise RuntimeError("Сначала вызовите fit.")
        flat = X.reshape(len(X), -1).astype(np.float64, copy=False)
        if flat.shape[1] != len(self.center_):
            raise ValueError(
                "Форма окон не совпадает с обучением: "
                f"ожидалось {len(self.center_)} признаков, получено {flat.shape[1]}."
            )
        return np.linalg.norm(flat - self.center_, axis=1)
