"""Чтение файлов Tennessee Eastman."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadr

from industrial_novelty.data.schema import ID_COLUMNS


def load_tep_file(path: Path | str) -> pd.DataFrame:
    """Прочитать один `.RData` и привести служебные столбцы к целым."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Нет файла {file_path}. Скачайте датасет Rieth и положите .RData в data/raw."
        )
    loaded = pyreadr.read_r(str(file_path))
    if not loaded:
        raise ValueError(f"В {file_path.name} нет объектов.")
    frame = next(iter(loaded.values()))
    missing = [column for column in ID_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"В {file_path.name} нет столбцов: {', '.join(missing)}.")
    frame = frame.copy()
    for column in ID_COLUMNS:
        frame[column] = frame[column].astype(np.int32)
    return frame
