"""Сводка и статические рисунки из результатов run_lacuna_experiments.py."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/lacuna_2026-10-30"
NAMES = {
    "baseline": "Текущая модель",
    "baseline+dependence": "+ ковариации",
    "baseline+second_moments": "+ вторые моменты",
    "baseline+spectrum": "+ спектр",
    "baseline+mmd1": "+ MMD, 1 окно",
    "baseline+mmd4": "+ MMD, 4 окна",
    "baseline+mmd8": "+ MMD, 8 окон",
    "baseline+neutral0": "+ NeuTraL, seed 0",
    "baseline+neutral1": "+ NeuTraL, seed 1",
    "baseline+neutral2": "+ NeuTraL, seed 2",
    "baseline+fixed0": "+ фикс. преобразования",
}


def table(frame: pd.DataFrame) -> str:
    headings = list(frame.columns)
    lines = [
        "| " + " | ".join(headings) + " |",
        "| " + " | ".join(["---"] * len(headings)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def paired_hard_intervals(cache: Path) -> pd.DataFrame:
    """Парный bootstrap по simulationRun, совместно для отказов 3/9/15."""
    baseline = pd.read_csv(cache / "baseline_run_5pct_960_runs.csv.gz")
    baseline = baseline.loc[baseline["fault_number"].isin([3, 9, 15])]
    rows = []
    rng = np.random.default_rng(20261030)
    for model in NAMES:
        if model == "baseline":
            continue
        candidate = pd.read_csv(cache / f"{model}_run_5pct_960_runs.csv.gz")
        merged = baseline.merge(
            candidate,
            on=["fault_number", "simulation_run"],
            suffixes=("_base", "_candidate"),
        )
        eligible = merged["scoreable_base"] & merged["scoreable_candidate"]
        merged["difference"] = np.where(
            eligible,
            merged["detected_candidate"].astype(float)
            - merged["detected_base"].astype(float),
            np.nan,
        )
        values = merged.pivot(
            index="simulation_run", columns="fault_number", values="difference"
        ).to_numpy()
        positions = rng.integers(0, len(values), size=(2000, len(values)))
        draws = np.nanmean(np.nanmean(values[positions], axis=1), axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append(
            {
                "model": model,
                "hard_detection_delta": float(np.nanmean(values, axis=0).mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def main():
    summary = pd.read_csv(RESULTS / "summary.csv")
    faults = pd.read_csv(RESULTS / "per_fault.csv")
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    summary["label"] = summary["model"].map(NAMES).fillna(summary["model"])
    main_table = summary.loc[
        (summary["protocol"] == "run_5pct_960") & summary["model"].isin(NAMES)
    ].copy()
    columns = [
        "label",
        "false_alarm_run_rate",
        "easy_early_detection",
        "hard_detection",
        "hard_auroc",
        "hard_median_delay_minutes",
    ]
    display = main_table[columns].copy()
    for column in columns[1:-1]:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    display[columns[-1]] = display[columns[-1]].map(lambda value: f"{value:.0f}")
    display.columns = [
        "Модель",
        "Тревожные нормальные прогоны",
        "Раннее обнаружение остальных",
        "Обнаружение 3/9/15",
        "AUROC 3/9/15",
        "Задержка 3/9/15, мин",
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    y = np.arange(len(main_table))
    rates = main_table["false_alarm_run_rate"].to_numpy()
    errors = np.vstack(
        [
            rates - main_table["false_alarm_run_ci_low"].to_numpy(),
            main_table["false_alarm_run_ci_high"].to_numpy() - rates,
        ]
    )
    axes[0].barh(y, rates * 100, color="#567da5", xerr=errors * 100, capsize=3)
    axes[0].axvline(5, linestyle="--", color="#b05a40", label="Бюджет калибровки 5%")
    axes[0].set_yticks(y, main_table["label"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Исправные прогоны с тревогой, % (95% Wilson CI)")
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02))
    axes[1].barh(
        y - 0.17,
        main_table["hard_detection"],
        height=0.32,
        label="Обнаружение 3/9/15",
        color="#63a08a",
    )
    axes[1].barh(
        y + 0.17,
        main_table["easy_early_detection"],
        height=0.32,
        label="Раннее обнаружение остальных",
        color="#d49a52",
    )
    axes[1].set_yticks(y, [])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1.04)
    axes[1].set_xlabel("Доля прогонов")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "TEP: одинаковый бюджет 5% тревожных прогонов, горизонт 960\n200 нормальных прогонов калибровки, 300 тестовых; 500 прогонов на отказ"
    )
    fig.savefig(RESULTS / "comparison.png", dpi=160)
    plt.close(fig)

    selected = [
        "baseline",
        "baseline+dependence",
        "baseline+spectrum",
        "baseline+mmd4",
        "baseline+neutral0",
        "baseline+neutral1",
        "baseline+neutral2",
    ]
    hard = faults.loc[
        (faults["protocol"] == "run_5pct_960")
        & faults["fault_number"].isin([3, 9, 15])
        & faults["model"].isin(selected)
    ]
    pivot = hard.pivot(
        index="model", columns="fault_number", values="detection_rate"
    ).reindex(selected)
    fig, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    pivot.rename(index=NAMES).plot.bar(
        ax=axis, width=0.8, color=["#567da5", "#63a08a", "#d49a52"]
    )
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("")
    axis.set_ylabel("Доля обнаруженных прогонов")
    axis.set_title("Слабые отказы: бюджет 5% за прогон, горизонт 960")
    axis.tick_params(axis="x", rotation=18)
    axis.legend(title="Отказ", loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(RESULTS / "hard_faults.png", dpi=160)
    plt.close(fig)

    history = json.loads((RESULTS / "neural_training.json").read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(8, 4), layout="constrained")
    for name, item in history.items():
        axis.plot(np.arange(1, len(item["loss"]) + 1), item["loss"], label=name)
    axis.set_yscale("log")
    axis.set_xlabel("Эпоха")
    axis.set_ylabel("Средний DCL на нормальном обучении")
    axis.legend()
    fig.savefig(RESULTS / "neural_training.png", dpi=160)
    plt.close(fig)

    calibration = summary.loc[(summary["model"] == "baseline")].copy()
    cache = ROOT / "data/processed/lacuna_2026-10-30"
    paired = paired_hard_intervals(cache)
    paired.to_csv(RESULTS / "paired_hard_differences.csv", index=False)
    normal_meta = pd.read_csv(cache / "normal_meta.csv")
    with np.load(cache / "normal_scores.npz") as saved:
        baseline_scores = saved["baseline"]
    matched = (
        normal_meta["simulation_run"]
        .isin(manifest["test_960_runs_from_normal_test_file"])
        .to_numpy()
    )
    matched_rows = []
    for protocol in ("window_1pct_960", "run_5pct_960", "run_10pct_960"):
        row = calibration.loc[calibration["protocol"] == protocol].iloc[0]
        frame = normal_meta.loc[matched, ["simulation_run"]].copy()
        frame["alarm"] = baseline_scores[matched] >= row["threshold"]
        matched_rows.append(
            {
                "protocol": protocol,
                "n_normal_test_runs": len(frame["simulation_run"].unique()),
                "false_alarm_run_rate": float(
                    frame.groupby("simulation_run")["alarm"].any().mean()
                ),
                "easy_detection": row["easy_detection"],
                "hard_detection": row["hard_detection"],
                "hard_median_delay_minutes": row["hard_median_delay_minutes"],
            }
        )
    matched_table = pd.DataFrame(matched_rows)
    matched_table.to_csv(RESULTS / "baseline_same_normal_runs.csv", index=False)
    cal_display = calibration[
        [
            "protocol",
            "n_calibration_runs",
            "n_normal_test_runs",
            "false_alarm_run_rate",
            "easy_detection",
            "easy_early_detection",
            "hard_detection",
            "hard_median_delay_minutes",
        ]
    ].round(4)
    text = "\n".join(
        [
            "# Проверка идей Lacuna на Tennessee Eastman",
            "",
            "Дата: 30 октября 2026 года.",
            "",
            "Статистические ветви и небольшой NeuTraL AD реализованы в проекте. Это адаптации принципов MSCRED, TFAD и kernel change detection; полные нейросетевые архитектуры MSCRED/TFAD/LNT не воспроизводились.",
            "",
            "## Результат проверки",
            "",
            "Ни одна новая ветвь в проверенных конфигурациях не дала убедительного общего выигрыша над текущей сборкой. Спектр и MMD практически повторяют её, зависимости перераспределяют ошибки между слабыми отказами, NeuTraL AD ухудшает результат во всех трёх запусках. Рабочую модель по этим данным заменять оснований нет.",
            "",
            "Калибровка по целым прогонам позволяет выбрать другую рабочую точку: на одних и тех же 300 нормальных прогонах доля тревог падает с 30.7% до 4.7%. При этом среднее обнаружение отказов 3/9/15 падает с 87.6% до 37.6%. Для остальных 17 отказов обнаружение остаётся 100%, а раннее обнаружение меняется с 77.6% до 75.2%. Это обмен чувствительности на меньшую нагрузку тревог.",
            "",
            "При бюджете 5% добавление спектра даёт +0.13 процентного пункта среднего обнаружения 3/9/15, MMD с четырьмя окнами — +0.07 п.п. Парный bootstrap по 500 simulationRun совместно для трёх отказов не выделяет эти изменения как надёжный выигрыш: интервалы включают ноль. Все интервалы и разности сохранены в paired_hard_differences.csv; они исследовательские, без поправки на выбор среди нескольких кандидатов.",
            "",
            "Общий вход: 52 канала, окно 24, шаг 8. Обучение на 400 исправных прогонах; 100 отдельных прогонов из нормального обучающего файла — для порогов до 500 отсчётов. Для горизонта 960 заранее случайно выделены 200 нормальных прогонов из тестового файла для калибровки и 300 для оценки. Все 20 отказов проверены на 500 прогонах каждый. Список прогонов и параметры записаны в manifest.json до оценки отказов.",
            "",
            "Каждая новая ветвь проверена отдельно и в максимуме с текущими CVA, моментами и медленными признаками. Все оценки дополнительно сглажены EWMA с весом 0.2. MMD проверен с историей 1/4/8 окон; NeuTraL AD — с seed 0/1/2 и контролем с фиксированными случайными преобразованиями. Нейросети обучались фиксированные 20 эпох без выбора по отказам.",
            "",
            "## Основное сравнение",
            "",
            "Одинаковый номинальный бюджет 5% за прогон и горизонт 960. Реализованные тестовые доли тревог могут различаться: их интервалы показаны на рисунке. Задержка — медиана по обнаруженным прогонам внутри отказа, затем медиана по трём отказам; пропуски отражает отдельная колонка обнаружения.",
            "",
            table(display),
            "",
            "![Сравнение](comparison.png)",
            "",
            "![Слабые отказы](hard_faults.png)",
            "",
            "## Что даёт сама калибровка",
            "",
            "Ниже прежний и новые пороги проверены на одних и тех же 300 нормальных прогонах. Отказы во всех трёх строках одинаковые: 500 прогонов на каждый тип, горизонт 960.",
            "",
            table(matched_table.round(4)),
            "",
            "Полная таблица протоколов с разными горизонтами:",
            "",
            table(cal_display),
            "",
            "Протокол window_1pct_960 воспроизводит прежний оконный порог на всех 500 нормальных тестовых прогонах. Протоколы run_*_500 используют 100 коротких нормальных калибровочных прогонов и первые 500 отсчётов всех 500 нормальных тестовых прогонов. Протоколы run_*_960 используют отдельные 200/300 длинных нормальных прогонов. Сравнение горизонтов не является чистым сравнением моделей: меняются время наблюдения и калибровочные данные.",
            "",
            "## Ограничения интерпретации",
            "",
            "Контроль по прогонам — маргинальный split-conformal результат при обменности нормальных траекторий и заранее зафиксированной модели. Это не обещание точного эмпирического FPR на одном наборе и не гарантия при произвольном изменении режима. Масштабы ветвей по-прежнему считаются на нормальном обучении; финальные пороги — на отдельных прогонах.",
            "",
            "В существующем протоколе исключены окна, пересекающие введение отказа; минимальная засчитываемая задержка остаётся 72 минуты. История MMD и EWMA работает на той же сетке окон. Более длинные спектральные окна 48/96 и отдельный оперативный протокол здесь не вводились. Это первая проверка пяти идей с фиксированными настройками, а не исчерпывающий поиск архитектур и гиперпараметров.",
            "",
            "Тест TEP уже использовался в предыдущих исследованиях проекта. Выбор кандидата по этим результатам исследовательский; независимое подтверждение требует ранее не использованных прогонов или новых симуляций. Для доверительных интервалов FPR единицей наблюдения служит целый прогон. Высокий AUROC сам по себе не означает хорошее обнаружение при малом бюджете тревог.",
            "",
            "## Воспроизведение",
            "",
            "```powershell",
            "poetry install",
            "uv pip install --python .venv/Scripts/python.exe -r requirements-neural.txt",
            ".venv/Scripts/python.exe scripts/run_lacuna_experiments.py",
            ".venv/Scripts/python.exe scripts/summarize_lacuna_experiments.py",
            "```",
            "",
            "CPU PyTorch вынесен в отдельный requirements-neural.txt, чтобы не менять основной Poetry lock. Версии библиотек записаны в manifest.json. summary.csv содержит все модели и пять протоколов, per_fault.csv — результаты каждого отказа, timing.csv — ориентировочные накопленные времена прерванного/продолженного расчёта, neural_training.json — ход обучения. Локальные веса, оценки и таблицы по отдельным прогонам лежат в игнорируемом data/processed/lacuna_2026-10-30/.",
            "",
            "Ссылки и исходные гипотезы: [исследовательская заметка](../../docs/research/lacuna_ideas_2026-10-30.md).",
            "",
        ]
    )
    (RESULTS / "report.md").write_text(text, encoding="utf-8")
    print(table(display))
    print("\nBASELINE CALIBRATION\n" + table(cal_display))
    print("\nManifest seeds:", manifest["seeds"])


if __name__ == "__main__":
    main()
