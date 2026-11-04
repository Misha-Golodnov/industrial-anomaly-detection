"""Нарезка уже масштабированных рядов на окна одной длины."""

from __future__ import annotations

import numpy as np
import pandas as pd

from industrial_novelty.data.scaling import ChannelScaler
from industrial_novelty.data.schema import (
    ID_COLUMNS,
    META_COLUMNS,
    SHUTDOWN_PRESSURE_COLUMN,
)


class WindowBatch:
    """Окна ``X`` формы ``(n, window_size, n_channels)`` и паспорт каждой строки.

    Строка ``meta`` с индексом ``i`` описывает окно ``X[i]``. ``label`` равен 1,
    если окно целиком лежит после начала отказа. ``post_shutdown`` равен True,
    если окно целиком лежит на участке после аварийного останова.
    """

    def __init__(
        self,
        X: np.ndarray,
        meta: pd.DataFrame,
        channel_names: tuple[str, ...],
        split: str,
    ) -> None:
        windows = np.asarray(X)
        if windows.ndim != 3:
            raise ValueError(
                "X должен иметь форму (n_windows, window_size, n_channels), "
                f"получено {windows.shape}."
            )
        if windows.shape[2] != len(channel_names):
            raise ValueError(
                f"В X {windows.shape[2]} каналов, в channel_names — {len(channel_names)}."
            )
        missing = [column for column in META_COLUMNS if column not in meta.columns]
        if missing:
            raise ValueError(f"В паспорте окон нет столбцов: {', '.join(missing)}.")
        if len(windows) != len(meta):
            raise ValueError(
                f"Число окон ({len(windows)}) не совпадает с паспортом ({len(meta)})."
            )
        self.X = windows
        self.meta = meta.reset_index(drop=True)
        self.channel_names = tuple(channel_names)
        self.split = split

    def __len__(self) -> int:
        return len(self.meta)

    @property
    def window_size(self) -> int:
        return int(self.X.shape[1])

    def with_scores(self, scores: np.ndarray) -> pd.DataFrame:
        """Вернуть паспорт с колонкой ``score`` в том же порядке, что и окна."""
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        if len(values) != len(self):
            raise ValueError(f"Ожидалось {len(self)} оценок, получено {len(values)}.")
        scored = self.meta.copy()
        scored["score"] = values
        return scored

    def __repr__(self) -> str:
        return (
            f"WindowBatch(split={self.split!r}, n_windows={len(self)}, "
            f"window_size={self.window_size}, n_channels={len(self.channel_names)})"
        )


def windows_from_frame(
    frame: pd.DataFrame,
    *,
    channels: tuple[str, ...],
    window_size: int,
    stride: int,
    scaler: ChannelScaler,
    fault_onset_sample: int | None,
    shutdown_pressure: float,
    split: str,
) -> WindowBatch:
    """Нарезать прогоны из ``frame`` на окна.

    Окно, которое начинается до ``fault_onset_sample`` и заканчивается на нём
    или позже, выбрасывается: в нём смешаны норма и отказ. Для нормальных
    прогонов передайте ``fault_onset_sample=None``, тогда метка у всех окон 0.
    """
    if window_size < 1 or stride < 1:
        raise ValueError("window_size и stride должны быть >= 1.")
    required = (*channels, SHUTDOWN_PRESSURE_COLUMN, *ID_COLUMNS)
    missing = list(
        dict.fromkeys(column for column in required if column not in frame.columns)
    )
    if missing:
        raise ValueError(f"В таблице нет столбцов: {', '.join(missing)}.")

    parts_x: list[np.ndarray] = []
    parts_meta: list[pd.DataFrame] = []
    grouped = frame.groupby(["faultNumber", "simulationRun"], sort=True)
    for (fault, run), group in grouped:
        windows, meta = _windows_for_run(
            group,
            fault=int(fault),
            run=int(run),
            channels=channels,
            window_size=window_size,
            stride=stride,
            scaler=scaler,
            fault_onset_sample=fault_onset_sample,
            shutdown_pressure=shutdown_pressure,
            split=split,
        )
        if len(windows):
            parts_x.append(windows)
            parts_meta.append(meta)

    if not parts_x:
        return WindowBatch(
            X=np.empty((0, window_size, len(channels)), dtype=np.float32),
            meta=_empty_meta(),
            channel_names=channels,
            split=split,
        )
    return WindowBatch(
        X=np.concatenate(parts_x, axis=0),
        meta=pd.concat(parts_meta, ignore_index=True),
        channel_names=channels,
        split=split,
    )


def _windows_for_run(
    group: pd.DataFrame,
    *,
    fault: int,
    run: int,
    channels: tuple[str, ...],
    window_size: int,
    stride: int,
    scaler: ChannelScaler,
    fault_onset_sample: int | None,
    shutdown_pressure: float,
    split: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    ordered = group.sort_values("sample", kind="mergesort")
    samples = ordered["sample"].to_numpy(dtype=np.int32, copy=False)
    n_times = len(samples)
    empty_x = np.empty((0, window_size, len(channels)), dtype=np.float32)
    if n_times == 0:
        return empty_x, _empty_meta()
    expected = np.arange(int(samples[0]), int(samples[0]) + n_times, dtype=np.int32)
    if not np.array_equal(samples, expected):
        raise ValueError(
            f"Прогон fault={fault}, run={run} содержит пропуски или повторы sample."
        )
    if n_times < window_size:
        return empty_x, _empty_meta()

    raw = ordered.loc[:, list(channels)].to_numpy(dtype=np.float64, copy=False)
    scaled = scaler.transform_array(raw)
    pressure = ordered[SHUTDOWN_PRESSURE_COLUMN].to_numpy(dtype=np.float64, copy=False)
    trip_hits = np.flatnonzero(pressure >= shutdown_pressure)
    trip_sample = int(samples[int(trip_hits[0])]) if len(trip_hits) else None

    starts = np.arange(0, n_times - window_size + 1, stride, dtype=np.int32)
    sample_start = samples[starts]
    sample_end = samples[starts + window_size - 1]
    if fault_onset_sample is None:
        keep = np.ones(len(starts), dtype=bool)
        labels = np.zeros(len(starts), dtype=np.int8)
    else:
        normal = sample_end < fault_onset_sample
        anomalous = sample_start >= fault_onset_sample
        keep = normal | anomalous
        labels = anomalous.astype(np.int8)
    post_shutdown = (
        np.zeros(len(starts), dtype=bool)
        if trip_sample is None
        else sample_start >= trip_sample
    )

    sample_start = sample_start[keep]
    sample_end = sample_end[keep]
    labels = labels[keep]
    post_shutdown = post_shutdown[keep]
    starts = starts[keep]
    if len(starts) == 0:
        return empty_x, _empty_meta()

    offsets = np.arange(window_size, dtype=np.int32)
    windows = scaled[starts[:, None] + offsets[None, :]]
    meta = pd.DataFrame(
        {
            "fault_number": np.full(len(starts), fault, dtype=np.int32),
            "simulation_run": np.full(len(starts), run, dtype=np.int32),
            "sample_start": sample_start,
            "sample_end": sample_end,
            "label": labels,
            "post_shutdown": post_shutdown,
            "split": split,
        }
    )
    return windows, meta


def _empty_meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fault_number": pd.Series(dtype=np.int32),
            "simulation_run": pd.Series(dtype=np.int32),
            "sample_start": pd.Series(dtype=np.int32),
            "sample_end": pd.Series(dtype=np.int32),
            "label": pd.Series(dtype=np.int8),
            "post_shutdown": pd.Series(dtype=bool),
            "split": pd.Series(dtype=str),
        }
    )
