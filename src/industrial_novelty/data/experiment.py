"""Сборка обучающих, валидационных и тестовых окон без утечки между прогонами."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_novelty.data.loading import load_tep_file
from industrial_novelty.data.scaling import ChannelScaler
from industrial_novelty.data.schema import (
    FAULT_ONSET_SAMPLE,
    SHUTDOWN_PRESSURE_KPA,
    resolve_channels,
)
from industrial_novelty.data.windows import WindowBatch, windows_from_frame


@dataclass(frozen=True)
class ExperimentConfig:
    """Параметры, которые должны совпадать у всех сравниваемых моделей.

    ``max_runs`` оставлен для черновых прогонов: в каждую часть эксперимента
    попадают прогоны с наименьшими номерами, не больше этого числа.
    Для отчётных результатов оставьте ``None``.
    """

    data_dir: Path | None = None
    window_size: int = 50
    stride: int = 10
    channels: str | tuple[str, ...] = "xmeas"
    val_fraction: float = 0.2
    seed: int = 0
    fault_onset_sample: int = FAULT_ONSET_SAMPLE
    shutdown_pressure: float = SHUTDOWN_PRESSURE_KPA
    max_runs: int | None = None

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size должен быть >= 1.")
        if self.stride < 1:
            raise ValueError("stride должен быть >= 1.")
        if not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction должен лежать в полуинтервале [0, 1).")
        if self.fault_onset_sample < 1:
            raise ValueError("fault_onset_sample должен быть >= 1.")
        if self.max_runs is not None and self.max_runs < 1:
            raise ValueError("max_runs должен быть >= 1.")
        if isinstance(self.data_dir, str):
            object.__setattr__(self, "data_dir", Path(self.data_dir))
        if isinstance(self.channels, list):
            object.__setattr__(self, "channels", tuple(self.channels))
        resolve_channels(self.channels)


@dataclass(frozen=True)
class FaultWindows:
    """Окна одного типа неисправности и полный список прогонов, даже без окон."""

    fault_number: int
    batch: WindowBatch
    simulation_runs: tuple[int, ...]


class Experiment:
    """Нормальные окна уже нарезаны. Неисправный тест читается по одной неисправности."""

    def __init__(
        self,
        config: ExperimentConfig,
        scaler: ChannelScaler,
        train: WindowBatch,
        val: WindowBatch | None,
        normal_test: WindowBatch,
        faulty_testing_path: Path,
    ) -> None:
        self.config = config
        self.scaler = scaler
        self.train = train
        self.val = val
        self.normal_test = normal_test
        self.channel_names = train.channel_names
        self._faulty_testing_path = faulty_testing_path
        self._faulty_frame: pd.DataFrame | None = None

    def counts(self) -> pd.DataFrame:
        """Число окон в уже материализованных нормальных выборках."""
        rows = []
        for name, batch in (
            ("train", self.train),
            ("val", self.val),
            ("normal_test", self.normal_test),
        ):
            if batch is None:
                continue
            rows.append(
                {
                    "split": name,
                    "n_windows": len(batch),
                    "n_runs": int(batch.meta["simulation_run"].nunique())
                    if len(batch)
                    else 0,
                    "window_size": batch.window_size,
                    "n_channels": len(batch.channel_names),
                }
            )
        return pd.DataFrame(rows)

    def faulty_test(self, fault_number: int) -> FaultWindows:
        """Окна одной неисправности. Первый вызов читает весь файл неисправного теста."""
        found = list(self.iter_faulty_test((fault_number,)))
        return found[0]

    def iter_faulty_test(
        self, fault_numbers: tuple[int, ...] | list[int] | None = None
    ) -> Iterator[FaultWindows]:
        """Идти по неисправностям, не держа окна всех отказов в памяти одновременно."""
        frame = self._load_faulty_frame()
        available = {int(value) for value in frame["faultNumber"].unique()}
        selected = tuple(sorted(available) if fault_numbers is None else fault_numbers)
        missing = [fault for fault in selected if fault not in available]
        if missing:
            raise ValueError(f"В неисправном тесте нет номеров: {missing}.")
        channels = self.channel_names
        for fault in selected:
            part = frame.loc[frame["faultNumber"] == fault]
            run_ids = _cap_ids(
                np.array(
                    sorted(int(value) for value in part["simulationRun"].unique())
                ),
                self.config.max_runs,
            )
            subset = part.loc[part["simulationRun"].isin(run_ids)]
            batch = windows_from_frame(
                subset,
                channels=channels,
                window_size=self.config.window_size,
                stride=self.config.stride,
                scaler=self.scaler,
                fault_onset_sample=self.config.fault_onset_sample,
                shutdown_pressure=self.config.shutdown_pressure,
                split="faulty_test",
            )
            yield FaultWindows(
                fault_number=int(fault),
                batch=batch,
                simulation_runs=tuple(int(run) for run in run_ids),
            )

    def _load_faulty_frame(self) -> pd.DataFrame:
        if self._faulty_frame is None:
            self._faulty_frame = load_tep_file(self._faulty_testing_path)
        return self._faulty_frame

    def __repr__(self) -> str:
        n_val = 0 if self.val is None else len(self.val)
        return (
            f"Experiment(window_size={self.config.window_size}, stride={self.config.stride}, "
            f"n_channels={len(self.channel_names)}, train={len(self.train)}, "
            f"val={n_val}, normal_test={len(self.normal_test)})"
        )


def build_experiment(config: ExperimentConfig | None = None) -> Experiment:
    """Собрать окна, на которых можно обучать и сравнивать модели.

    Масштаб считается по отсчётам обучающих прогонов, каждый отсчёт один раз.
    Валидация, нормальный тест и неисправный тест этот масштаб только применяют.
    Разбиение идёт по ``simulationRun``: соседние отсчёты одного прогона не
    попадают в разные части нормального обучения.
    """
    config = config or ExperimentConfig()
    channels = resolve_channels(config.channels)
    paths = _resolve_paths(config.data_dir)
    _require_file(paths["fault_free_training"])
    _require_file(paths["fault_free_testing"])

    training = load_tep_file(paths["fault_free_training"])
    _require_columns(training, channels)
    train_ids, val_ids = split_run_ids(
        training["simulationRun"].to_numpy(),
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    train_ids = _cap_ids(train_ids, config.max_runs)
    val_ids = _cap_ids(val_ids, config.max_runs)
    scaler = ChannelScaler().fit(
        training.loc[training["simulationRun"].isin(train_ids)],
        channels,
    )
    train = _window_runs(
        training,
        train_ids,
        channels=channels,
        config=config,
        scaler=scaler,
        fault_onset_sample=None,
        split="train",
    )
    val = (
        None
        if len(val_ids) == 0
        else _window_runs(
            training,
            val_ids,
            channels=channels,
            config=config,
            scaler=scaler,
            fault_onset_sample=None,
            split="val",
        )
    )
    del training
    _require_windows(train, "train")
    if val is not None:
        _require_windows(val, "val")
        _require_normal(val, "val")
    _require_normal(train, "train")

    normal = load_tep_file(paths["fault_free_testing"])
    _require_columns(normal, channels)
    normal_ids = _cap_ids(
        np.array(sorted(int(value) for value in normal["simulationRun"].unique())),
        config.max_runs,
    )
    normal_test = _window_runs(
        normal,
        normal_ids,
        channels=channels,
        config=config,
        scaler=scaler,
        fault_onset_sample=None,
        split="normal_test",
    )
    del normal
    _require_windows(normal_test, "normal_test")
    _require_normal(normal_test, "normal_test")

    return Experiment(
        config=config,
        scaler=scaler,
        train=train,
        val=val,
        normal_test=normal_test,
        faulty_testing_path=paths["faulty_testing"],
    )


def split_run_ids(
    run_ids: np.ndarray,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Разделить номера прогонов на обучение и валидацию.

    Оба массива отсортированы. При ``val_fraction > 0`` в валидации остаётся
    хотя бы один прогон, в обучении тоже.
    """
    ids = np.array(sorted({int(value) for value in run_ids}), dtype=np.int32)
    if len(ids) == 0:
        raise ValueError("В данных нет прогонов.")
    if val_fraction == 0:
        return ids, np.empty(0, dtype=np.int32)
    if len(ids) < 2:
        raise ValueError("Для валидации нужно хотя бы два прогона.")
    order = ids.copy()
    np.random.default_rng(seed).shuffle(order)
    n_val = round(len(ids) * val_fraction)
    n_val = min(max(n_val, 1), len(ids) - 1)
    val_ids = np.sort(order[:n_val])
    train_ids = np.sort(order[n_val:])
    return train_ids.astype(np.int32, copy=False), val_ids.astype(np.int32, copy=False)


def _window_runs(
    frame: pd.DataFrame,
    run_ids: np.ndarray,
    *,
    channels: tuple[str, ...],
    config: ExperimentConfig,
    scaler: ChannelScaler,
    fault_onset_sample: int | None,
    split: str,
) -> WindowBatch:
    subset = frame.loc[frame["simulationRun"].isin({int(value) for value in run_ids})]
    return windows_from_frame(
        subset,
        channels=channels,
        window_size=config.window_size,
        stride=config.stride,
        scaler=scaler,
        fault_onset_sample=fault_onset_sample,
        shutdown_pressure=config.shutdown_pressure,
        split=split,
    )


def _cap_ids(run_ids: np.ndarray, max_runs: int | None) -> np.ndarray:
    ordered = np.array(sorted(int(value) for value in run_ids), dtype=np.int32)
    if max_runs is None:
        return ordered
    return ordered[:max_runs]


def _require_windows(batch: WindowBatch, name: str) -> None:
    if len(batch) == 0:
        raise ValueError(
            f"В выборке {name} не получилось ни одного окна. "
            "Уменьшите window_size относительно длины прогона."
        )


def _require_normal(batch: WindowBatch, name: str) -> None:
    labels = batch.meta["label"].to_numpy()
    if np.any(labels != 0):
        raise RuntimeError(f"В выборке {name} оказались окна с меткой отказа.")


def _require_columns(frame: pd.DataFrame, channels: tuple[str, ...]) -> None:
    missing = [column for column in channels if column not in frame.columns]
    if missing:
        raise ValueError(f"В данных нет каналов: {', '.join(missing)}.")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет файла {path}. Скачайте датасет Rieth и положите .RData в data/raw."
        )


def _resolve_paths(data_dir: Path | None) -> dict[str, Path]:
    root = Path(data_dir) if data_dir is not None else _default_data_dir()
    return {
        "fault_free_training": root / "TEP_FaultFree_Training.RData",
        "fault_free_testing": root / "TEP_FaultFree_Testing.RData",
        "faulty_testing": root / "TEP_Faulty_Testing.RData",
    }


def _default_data_dir() -> Path:
    starts = (Path.cwd().resolve(), Path(__file__).resolve())
    seen: set[Path] = set()
    for start in starts:
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "pyproject.toml").is_file():
                return candidate / "data" / "raw"
    raise FileNotFoundError(
        "Не найден каталог проекта с pyproject.toml. Укажите ExperimentConfig(data_dir=...)."
    )
