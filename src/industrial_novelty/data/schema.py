"""Столбцы TEP и правила, общие для всех моделей."""

from __future__ import annotations

from collections.abc import Sequence

ID_COLUMNS = ("faultNumber", "simulationRun", "sample")

# В тестовых прогонах Rieth отсчёты 1..160 — норма, отказ начинается с 161-го.
FAULT_ONSET_SAMPLE = 161
LAST_NORMAL_SAMPLE = FAULT_ONSET_SAMPLE - 1

# Стенд останавливается при давлении реактора 3000 кПа. 2990 отсекает шум вокруг предела.
SHUTDOWN_PRESSURE_COLUMN = "xmeas_7"
SHUTDOWN_PRESSURE_KPA = 2990.0

SAMPLE_PERIOD_MINUTES = 3

XMEAS_COLUMNS = tuple(f"xmeas_{index}" for index in range(1, 42))
XMV_COLUMNS = tuple(f"xmv_{index}" for index in range(1, 12))
ALL_CHANNELS = XMEAS_COLUMNS + XMV_COLUMNS

META_COLUMNS = (
    "fault_number",
    "simulation_run",
    "sample_start",
    "sample_end",
    "label",
    "post_shutdown",
    "split",
)


def resolve_channels(spec: str | Sequence[str]) -> tuple[str, ...]:
    """Вернуть имена каналов в фиксированном порядке.

    Строка ``xmeas``, ``xmv`` или ``all`` выбирает готовый набор.
    Один столбец передаётся кортежем: ``("xmeas_1",)``.
    """
    if isinstance(spec, str):
        if spec == "xmeas":
            return XMEAS_COLUMNS
        if spec == "xmv":
            return XMV_COLUMNS
        if spec == "all":
            return ALL_CHANNELS
        raise ValueError(
            "channels должен быть 'xmeas', 'xmv', 'all' или списком имён столбцов. "
            "Один столбец задаётся так: channels=('xmeas_1',)."
        )
    columns = tuple(spec)
    if not columns:
        raise ValueError("Список каналов пуст.")
    if any(not isinstance(column, str) or column == "" for column in columns):
        raise ValueError("Имена каналов должны быть непустыми строками.")
    if len(set(columns)) != len(columns):
        raise ValueError("Имена каналов повторяются.")
    return columns
