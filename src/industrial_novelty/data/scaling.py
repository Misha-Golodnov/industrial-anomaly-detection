"""Масштаб каналов по сырым обучающим отсчётам, без повторов из пересекающихся окон."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class ChannelScaler:
    """Стандартизация каждого канала: среднее и стандартное отклонение по обучающим строкам."""

    def fit(self, frame: pd.DataFrame, columns: Sequence[str]) -> ChannelScaler:
        columns = tuple(columns)
        if not columns:
            raise ValueError("Список каналов пуст.")
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"В таблице нет столбцов: {', '.join(missing)}.")
        if len(frame) == 0:
            raise ValueError("Нельзя оценить масштаб по пустой таблице.")
        values = frame.loc[:, list(columns)].to_numpy(dtype=np.float64, copy=False)
        self.columns_ = columns
        self.mean_ = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        """Преобразовать массив формы ``(n_rows, n_channels)`` в ``float32``."""
        if not hasattr(self, "mean_"):
            raise RuntimeError("ChannelScaler ещё не обучен.")
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != len(self.mean_):
            raise ValueError(
                "Ожидался массив формы (n_rows, n_channels) "
                f"с {len(self.mean_)} каналами, получено {values.shape}."
            )
        scaled = (values - self.mean_) / self.scale_
        return scaled.astype(np.float32, copy=False)

    def __repr__(self) -> str:
        if not hasattr(self, "columns_"):
            return "ChannelScaler(unfitted)"
        return f"ChannelScaler(n_channels={len(self.columns_)})"
