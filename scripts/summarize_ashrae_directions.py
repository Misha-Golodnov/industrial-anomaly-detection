"""Summaries of all development stages, retaining negative results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path("results/ashrae_directions_2026-11-04")
    base = pd.read_csv(root / "summary.csv")
    precision = pd.read_csv(root / "precision/summary.csv")
    scaling = pd.read_csv(root / "scaling/summary.csv")
    combined = pd.concat(
        [
            base,
            precision[~precision.method.isin(["raw_lw", "raw_oas"])],
            scaling[~scaling.method.isin(["raw_lw", "thermo_lw"])],
        ],
        ignore_index=True,
    )
    first = combined[(combined.severity == 1) & (combined.alpha == 0.05)]
    diagnosis = pd.read_csv(root / "supervised/summary.csv")
    matched = pd.read_csv(root / "matched_windows_summary.csv")
    combined.to_csv(root / "combined_summary.csv", index=False)
    methods = (
        first[(first.window == 48) & (first.protocol == "blocked")]
        .sort_values("auroc_mean", ascending=False)
        .method.tolist()
    )
    lines = [
        "# RP-1043: второй раунд, результаты перспективных направлений",
        "",
        "Дата: 4 ноября 2026. [Литература, математика и анализ ограничений](../../docs/research/ashrae_promising_directions_2026-11-04.md).",
        "",
        (
            "**2240 завершённых обучений:** 680 normal-only, 960 supervised, 360 covariance follow-up и 240 scaling audit "
            "(включая повторные baseline). Это development на уже просмотренной авторской CSV-копии, "
            "не независимое подтверждение SOTA. Пять seeds — split/initialization variability, не независимые чиллеры."
        ),
        "",
        "## Что улучшилось",
        "",
        (
            "- Normal-only: при W48 covariance shrinkage с выбором по normal reference даёт blocked SL1 "
            "AUROC 0.9669, recall 66.80%, FPR 6.51%. Исходный W12 Ledoit–Wolf: AUROC 0.9413, FPR 10.31%. "
            "Порог в обоих случаях имеет номинальный calibration budget 5%, но достигнутые FPR различаются."
        ),
        (
            "- Diagnosis seen SL1: при том же W12 raw+thermo LDA поднимает macro-F1 с 0.9046 до 0.9368 "
            "и уменьшает normal FPR с 16.07% до 13.33%. W48 LDA: macro-F1 0.9499, FPR 9.91%."
        ),
        (
            "- Положительный эффект длинного окна сохраняется в анализе одинаковых calibration/test endpoints. "
            "Это контролирует изменение test support, но fit support и представление окна остаются разными."
        ),
        "",
        "## Что не решено",
        "",
        (
            "Hold 50 °F по-прежнему вызывает много ложных тревог. Seen-fault выигрыш LDA "
            "не переносится на SL2–SL4 → SL1. EOS, короткие спектральные признаки, GP, MEWMA/scan и "
            "простые severity signatures не дали универсального улучшения. Ниже сохранены все варианты."
        ),
        "",
        "## Normal-only: blocked SL1",
        "",
        (
            "В ячейке AUROC / normal FPR. Порог определяется только нормальной калибровкой; "
            "recall и per-fault показатели находятся в CSV. Сравнение W12/W48 по этим строкам "
            "допускает разные test endpoints; соответствующий контроль приведён далее."
        ),
        "",
        "| Метод | W12: AUROC / FPR | W48: AUROC / FPR |",
        "|---|---:|---:|",
    ]
    for method in methods:
        cells = []
        for w in (12, 48):
            row = first[
                (first.window == w)
                & (first.protocol == "blocked")
                & (first.method == method)
            ].iloc[0]
            cells.append(f"{row.auroc_mean:.4f} / {100 * row.normal_fpr_mean:.2f}%")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        (
            "В отдельном unit-scaling audit убран общий floor 0.05 для разномерных физических признаков. "
            "thermo_unit_* использует стандартное отклонение каждой неконстантной координаты. "
            "Это восстановило инвариантность к смене единиц мощности (проверена тестом), "
            "но не улучшило основной результат: W48 thermo_unit_lw AUROC 0.9024 против thermo_lw 0.9154 "
            "и raw covariance CV 0.9669. Все четыре новые абляции показаны выше."
        ),
        "",
        (
            "`cov_cv` означает выбор shrinkage по отложенной норме; "
            "`graphical_cv` — такой же выбор L1 precision; `raw_cov_1e-4` — фиксированная абляция. "
            "Суффикс CV не означает настройку на fault test. `thermo` использует fixed physical map, "
            "`eos` дополнительно предполагает R134a/Fahrenheit/psig."
        ),
        "",
        "## Перенос режима: отдельные новые кандидаты, W48 SL1",
        "",
        "В ячейке AUROC / recall / normal FPR. Низкий FPR не следует читать без recall.",
        "",
        "| Метод | Hold 40 °F | Hold 45 °F | Hold 50 °F |",
        "|---|---:|---:|---:|",
    ]
    selected = [
        "raw_lw",
        "raw_cov_cv",
        "invariant_cov_cv",
        "thermo_poly1",
        "thermo_gp",
        "thermo_spectral_lw",
    ]
    for method in selected:
        cells = []
        for protocol in ("loto40", "loto45", "loto50"):
            row = first[
                (first.window == 48)
                & (first.protocol == protocol)
                & (first.method == method)
            ].iloc[0]
            cells.append(
                f"{row.auroc_mean:.3f} / {100 * row.recall_mean:.1f}% / {100 * row.normal_fpr_mean:.1f}%"
            )
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Проверка одинаковых концов окон",
        "",
        (
            "Calibration endpoints и normal/fault test endpoints совпадают между W12/W48. "
            "Threshold пересчитан отдельно для каждой модели на общей calibration population, "
            "без использования test для выбора порога. Таблица — blocked SL1."
        ),
        "",
        "| Метод | W12 AUROC / FPR | W48 AUROC / FPR |",
        "|---|---:|---:|",
    ]
    for method in matched.method.unique():
        cells = []
        for w in (12, 48):
            row = matched[
                (matched.window == w)
                & (matched.protocol == "blocked")
                & (matched.severity == 1)
                & (matched.method == method)
            ].iloc[0]
            cells.append(f"{row.auroc_mean:.4f} / {100 * row.normal_fpr_mean:.2f}%")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        (
            "[Все matched результаты](matched_windows_summary.csv). Это post-hoc development analysis; "
            "он не создаёт нового независимого теста."
        ),
        "",
    ]

    plot_methods = [
        "raw_lw",
        "raw_cov_cv",
        "raw_cov_1e-4",
        "invariant_cov_cv",
        "thermo_poly1",
        "thermo_gp",
        "thermo_spectral_lw",
        "raw_graphical_cv",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric, title, cmap, minimum in zip(
        axes,
        ["auroc_mean", "normal_fpr_mean"],
        ["SL1 AUROC (higher is better)", "Normal FPR at nominal 5% calibration"],
        ["viridis", "magma_r"],
        [0.5, 0],
        strict=True,
    ):
        values = (
            first[first.window == 48]
            .pivot(index="method", columns="protocol", values=metric)
            .loc[plot_methods, ["blocked", "loto40", "loto45", "loto50"]]
            .to_numpy()
        )
        im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=minimum, vmax=1)
        ax.set_xticks(range(4), ["Blocked", "Hold 40 F", "Hold 45 F", "Hold 50 F"])
        ax.set_yticks(range(len(plot_methods)), plot_methods)
        ax.set_title(title, fontsize=10)
        for i in range(len(plot_methods)):
            for j in range(4):
                white = (
                    values[i, j] < 0.7 if metric == "auroc_mean" else values[i, j] > 0.5
                )
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if white else "black",
                )
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(
        "RP-1043 development | 48-row windows | Mean of five split seeds", fontsize=11
    )
    fig.savefig(root / "novelty_directions.png", dpi=180)
    plt.close(fig)
    lines += [
        "![Новые кандидаты normal-only](novelty_directions.png)",
        "",
        "## Supervised: два разных сценария",
        "",
        (
            "Значения — macro-F1 / normal FPR для blocked. У ordinary classifiers нет "
            "5% rejection threshold. Ray/subspace classifiers используют дополнительную "
            "normal reference/calibration для gate. Это указано в manifest."
        ),
        "",
        "| Метод | W12 seen SL1 | W48 seen SL1 | W12 SL2–4 → SL1 | W48 SL2–4 → SL1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in diagnosis.method.unique():
        cells = []
        for scenario, w in [
            ("seen_sl1", 12),
            ("seen_sl1", 48),
            ("transfer_sl1", 12),
            ("transfer_sl1", 48),
        ]:
            row = diagnosis[
                (diagnosis.window == w)
                & (diagnosis.scenario == scenario)
                & (diagnosis.protocol == "blocked")
                & (diagnosis.method == method)
            ].iloc[0]
            cells.append(f"{row.macro_f1_mean:.4f} / {100 * row.normal_fpr_mean:.2f}%")
        lines.append(f"| {method} | " + " | ".join(cells) + " |")
    lines += [
        "",
        (
            "[Все supervised режимы](supervised/summary.csv), [per-class recall и confusion rows](supervised/per_class.csv). "
            "Предыдущая temporal CNN (W12, blocked transfer macro-F1 0.7509) сильнее новых "
            "кандидатов в переносе severity. Победу LDA в seen SL1 на эту задачу не распространяем."
        ),
        "",
    ]
    details = pd.read_csv(root / "supervised/per_class.csv")
    classes = ["normal", "cf", "eo", "fwc", "fwe", "nc", "rl", "ro"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, method, w in zip(
        axes,
        ["raw_logistic", "raw_thermo_lda", "raw_thermo_lda"],
        [12, 12, 48],
        strict=True,
    ):
        subset = details[
            (details.window == w)
            & (details.method == method)
            & (details.protocol == "blocked")
            & (details.scenario == "seen_sl1")
        ]
        matrices = []
        for _, group in subset.groupby("seed"):
            cm = np.array(
                [
                    json.loads(group[group["class"] == label].iloc[0].confusion_row)
                    for label in classes
                ]
            )
            matrices.append(cm / cm.sum(axis=1, keepdims=True))
        values = np.mean(matrices, axis=0)
        ax.imshow(values, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(8), classes, rotation=45)
        ax.set_yticks(range(8), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{method}, W={w}", fontsize=10)
        for i in range(8):
            for j in range(8):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if values[i, j] > 0.5 else "black",
                )
    fig.suptitle(
        "Seen SL1 diagnosis | Row-normalized confusion | Mean of five split seeds",
        fontsize=11,
    )
    fig.savefig(root / "diagnosis_directions.png", dpi=180)
    plt.close(fig)
    lines += [
        "![Сравнение диагнозов SL1](diagnosis_directions.png)",
        "",
        "## Файлы и воспроизведение",
        "",
        (
            "[Основной manifest](manifest.json), [per-fault normal-only](per_fault.csv), "
            "[covariance manifest](precision/extension_manifest.json), [параметры и отброшенные candidates](precision/selection.csv), "
            "[supervised manifest](supervised/manifest.json). Raw score archives сохранены локально в scores/ "
            "и precision/scores/ и восстанавливаются runner-скриптами."
        ),
        "",
        "```powershell",
        "uv pip install --python .venv/Scripts/python.exe -r requirements-ashrae-extensions.txt",
        ".venv/Scripts/python.exe scripts/run_ashrae_directions.py",
        ".venv/Scripts/python.exe scripts/run_ashrae_diagnosis_directions.py",
        ".venv/Scripts/python.exe scripts/run_ashrae_precision.py",
        ".venv/Scripts/python.exe scripts/run_ashrae_physical_scaling.py",
        ".venv/Scripts/python.exe scripts/analyze_ashrae_matched_windows.py",
        ".venv/Scripts/python.exe scripts/summarize_ashrae_directions.py",
        "```",
        "",
        (
            "По-прежнему доступны одна normal trajectory и по одному файлу на fault/severity. "
            "Fault/file confounding, rounded timestamps, отсутствие независимых нормальных повторов "
            "и неустановленный fault onset ограничивают выводы. Точного detection delay в секундах "
            "и distribution-free runwise guarantee этот раунд не устанавливает."
        ),
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    sup_lines = [
        "# Supervised follow-up RP-1043",
        "",
        "[Общий отчёт с таблицами и графиками](../report.md)",
        "",
        (
            "12 методов, два окна, четыре режима, пять seeds, два сценария: 960 fits. "
            "Seen SL1 и SL2–SL4 → SL1 оцениваются раздельно. Полные значения — в summary.csv."
        ),
        "",
    ]
    for scenario in ("seen_sl1", "transfer_sl1"):
        for w in (12, 48):
            sup_lines += [
                f"## {scenario}, W={w}",
                "",
                "Ячейка: macro-F1 / normal FPR.",
                "",
                "| Метод | Blocked | Hold 40 F | Hold 45 F | Hold 50 F |",
                "|---|---:|---:|---:|---:|",
            ]
            for method in diagnosis.method.unique():
                cells = []
                for protocol in ("blocked", "loto40", "loto45", "loto50"):
                    row = diagnosis[
                        (diagnosis.window == w)
                        & (diagnosis.scenario == scenario)
                        & (diagnosis.protocol == protocol)
                        & (diagnosis.method == method)
                    ].iloc[0]
                    cells.append(
                        f"{row.macro_f1_mean:.4f} / {100 * row.normal_fpr_mean:.2f}%"
                    )
                sup_lines.append(f"| {method} | " + " | ".join(cells) + " |")
            sup_lines += [""]
    sup_lines += [
        "[Manifest](manifest.json), [точные разбиения](split_blocks.json), [per-seed](per_seed.csv), [per-class](per_class.csv).",
        "",
    ]
    (root / "supervised/report.md").write_text("\n".join(sup_lines), encoding="utf-8")
    print(root / "report.md")


if __name__ == "__main__":
    main()
