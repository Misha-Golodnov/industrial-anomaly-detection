"""Report the supervised track without mixing it with normal-only detection."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path("results/ashrae_2026-11-04/supervised")
    data = pd.read_csv(root / "summary.csv")
    details = pd.read_csv(root / "per_class.csv")
    classes = ["normal", "cf", "eo", "fwc", "fwe", "nc", "rl", "ro"]
    methods = [
        "logistic",
        "rbf_svm",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "temporal_cnn",
    ]
    blocked = data[data.protocol == "blocked"].set_index(["scenario", "method"])
    seen_linear = blocked.loc[("seen_sl1", "logistic"), "macro_f1_mean"]
    seen_cnn = blocked.loc[("seen_sl1", "temporal_cnn"), "macro_f1_mean"]
    transfer_cnn = blocked.loc[("transfer_sl1", "temporal_cnn"), "macro_f1_mean"]
    transfer_gb = blocked.loc[
        ("transfer_sl1", "hist_gradient_boosting"), "macro_f1_mean"
    ]
    lines = [
        "# RP-1043: supervised diagnosis, отдельный трек",
        "",
        "Дата: 4 ноября 2026. [Математика и общий протокол](../../../docs/research/ashrae_rp1043_research_2026-11-04.md).",
        "",
        (
            "Обучены **6 методов × 2 сценария × 4 протокола × 5 seeds = 240 моделей**. "
            "Это восемь классов (normal + семь неисправностей). Fault labels разрешены в обучении; "
            "результаты нельзя напрямую ранжировать вместе с normal-only novelty detection."
        ),
        "",
        (
            "**Разведочная проверка, без заявления SOTA.** Данные — авторская CSV-копия; один "
            "нормальный файл и по одному файлу на fault/severity. В seen SL1 разные блоки одного "
            "файла попадают в обучение и тест: эффект неисправности может быть смешан с особенностями "
            "конкретной записи. Независимые повторные эксперименты отсутствуют. Seeds не заменяют их."
        ),
        "",
        "## Два сценария",
        "",
        "- `seen_sl1`: обучение на fit-блоках SL1; тест на других блоках SL1.",
        (
            "- `transfer_sl1`: обучение только на SL2–SL4 для каждой неисправности; тест на SL1. "
            "Классы неисправностей известны, их слабая выраженность в обучении отсутствует. "
            "Одна normal trajectory разделена без дублирования между severities."
        ),
        "",
        (
            "Общие allowlist, ширина окна, embargo и протоколы совпадают с novelty треком. "
            "Классические модели получают mean+std (52 признака), CNN — полное окно 12×26. "
            "Все scalers обучены только на разрешённых supervised fit-данных; веса классов "
            "определяются только по обучению. Параметры фиксированы без test-based подбора."
        ),
        "",
        "## Временные блоки, SL1",
        "",
        (
            "± — SD по пяти split/initialization seeds, не доверительный интервал. "
            "FPR означает долю normal окон, классифицированных как любая неисправность; "
            "порог в 5% из novelty трека здесь не используется."
        ),
        "",
        "| Метод | Seen SL1: macro-F1 ± SD | FPR, % | SL2–SL4 → SL1: macro-F1 ± SD | FPR, % |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        rows = [
            data[
                (data.protocol == "blocked")
                & (data.scenario == scenario)
                & (data.method == method)
            ].iloc[0]
            for scenario in ("seen_sl1", "transfer_sl1")
        ]
        a, b = rows
        lines.append(
            f"| {method} | {a.macro_f1_mean:.4f} ± {a.macro_f1_std:.4f} | {100 * a.normal_fpr_mean:.2f} | "
            f"{b.macro_f1_mean:.4f} ± {b.macro_f1_std:.4f} | {100 * b.normal_fpr_mean:.2f} |"
        )
    lines += [
        "",
        "## Удержанная температурная полоса",
        "",
        (
            "В ячейке macro-F1 / FPR нормальных окон. Классификатор обучен только на source bands; "
            "target labels в обучение не попадают."
        ),
        "",
    ]
    for scenario in ("seen_sl1", "transfer_sl1"):
        lines += [
            f"### {scenario}",
            "",
            "| Метод | Hold 40 °F | Hold 45 °F | Hold 50 °F |",
            "|---|---:|---:|---:|",
        ]
        for method in methods:
            cells = []
            for protocol in ("loto40", "loto45", "loto50"):
                row = data[
                    (data.protocol == protocol)
                    & (data.scenario == scenario)
                    & (data.method == method)
                ].iloc[0]
                cells.append(
                    f"{row.macro_f1_mean:.3f} / {100 * row.normal_fpr_mean:.1f}%"
                )
            lines.append(f"| {method} | " + " | ".join(cells) + " |")
        lines += [""]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, scenario, method in zip(
        axes, ["seen_sl1", "transfer_sl1"], ["logistic", "temporal_cnn"], strict=True
    ):
        subset = details[
            (details.protocol == "blocked")
            & (details.scenario == scenario)
            & (details.method == method)
        ]
        matrices = []
        for _, group in subset.groupby("seed"):
            matrix = np.array(
                [
                    json.loads(group[group["class"] == label].iloc[0].confusion_row)
                    for label in classes
                ]
            )
            matrices.append(matrix / matrix.sum(axis=1, keepdims=True))
        mean = np.mean(matrices, axis=0)
        ax.imshow(mean, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(8), classes, rotation=45)
        ax.set_yticks(range(8), classes)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title(f"{scenario}: {method}", fontsize=10)
        for i in range(8):
            for j in range(8):
                ax.text(
                    j,
                    i,
                    f"{mean[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if mean[i, j] > 0.5 else "black",
                )
    fig.suptitle(
        "Exploratory RP-1043 diagnosis | Row-normalized confusion, mean of 5 seeds",
        fontsize=11,
    )
    fig.savefig(root / "sl1_confusion.png", dpi=180)
    plt.close(fig)
    lines += [
        "## Выводы",
        "",
        (
            f"На blocked seen SL1 лучший средний macro-F1 у logistic regression ({seen_linear:.4f}); "
            f"temporal CNN близка ({seen_cnn:.4f}). Сложность модели не гарантирует улучшение."
        ),
        "",
        (
            f"При SL2–SL4 → SL1 на blocked лучшая из проверенных — temporal CNN ({transfer_cnn:.4f}), "
            f"затем histogram gradient boosting ({transfer_gb:.4f}). Это предварительный результат; "
            "при переносе температурных условий устойчивого универсального лидера нет."
        ),
        "",
        (
            "Ошибки нормы остаются существенными даже при высоком macro-F1: "
            "fault diagnosis по восьми классам и надёжная выдача тревоги требуют раздельной оценки. "
            "Точные значения, включая все normal FPR, приведены в таблицах выше."
        ),
        "",
        "![Матрицы ошибок SL1](sl1_confusion.png)",
        "",
        "## Воспроизведение",
        "",
        "```powershell",
        ".venv/Scripts/python.exe scripts/run_ashrae_supervised.py",
        ".venv/Scripts/python.exe scripts/summarize_ashrae_supervised.py",
        "```",
        "",
        (
            "[Manifest](manifest.json), [split blocks](split_blocks.json), [per-seed metrics](per_seed.csv), "
            "[per-class recall и confusion rows](per_class.csv), [summary](summary.csv)."
        ),
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "report.md")


if __name__ == "__main__":
    main()
