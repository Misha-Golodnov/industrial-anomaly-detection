"""Generate a concise Russian report and scientific figure from retained metrics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    root = Path("results/ashrae_2026-11-04")
    summary = pd.read_csv(root / "summary.csv")
    first = summary[(summary.severity == 1) & (summary.alpha == 0.05)]
    order = (
        first[first.protocol == "blocked"]
        .sort_values("auroc_mean", ascending=False)
        .method.tolist()
    )
    protocols = ["blocked", "loto40", "loto45", "loto50"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)
    for ax, metric, title, cmap in zip(
        axes,
        ["auroc_mean", "normal_fpr_mean"],
        [
            "SL1 macro AUROC (higher is better)",
            "Normal FPR at nominal 5% calibration (lower is better)",
        ],
        ["viridis", "magma_r"],
        strict=True,
    ):
        values = (
            first.pivot(index="method", columns="protocol", values=metric)
            .loc[order, protocols]
            .to_numpy()
        )
        view = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(4), ["Blocked", "Hold 40 F", "Hold 45 F", "Hold 50 F"])
        ax.set_yticks(range(len(order)), order)
        ax.set_title(title, fontsize=10)
        for i in range(len(order)):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white"
                    if (values[i, j] < 0.5 if cmap == "viridis" else values[i, j] > 0.5)
                    else "black",
                )
        fig.colorbar(view, ax=ax, shrink=0.7)
    fig.suptitle(
        "ASHRAE RP-1043 CSV derivative | 5 split/initialization seeds | Exploratory, one normal trajectory",
        fontsize=11,
    )
    fig.savefig(root / "sl1_comparison.png", dpi=180)
    plt.close(fig)

    text = [
        "# RP-1043: предварительные результаты на CSV-копии",
        "",
        (
            "Дата: 4 ноября 2026. Основной обзор и математическая постановка: "
            "[исследование](../../docs/research/ashrae_rp1043_research_2026-11-04.md)."
        ),
        "",
        (
            "**Это разведочный эксперимент, не подтверждение SOTA.** Одна нормальная траектория, "
            "искусственные временные блоки и повторяющиеся округлённые timestamps ограничивают выводы. "
            "Пять seeds показывают чувствительность к разбиению/инициализации, а не пять независимых испытаний."
        ),
        "",
        (
            "Обучены 12 методов × 4 протокола × 5 seeds = 240 моделей. "
            "Оценены семь типов неисправностей на SL1–SL4. Метки неисправностей не использованы "
            "для обучения, порога или подбора гиперпараметров. Значения ниже — средние по семи отказам и пяти seeds."
        ),
        "",
        "## Слабые неисправности SL1: разделение по блокам",
        "",
        (
            "Порог задан по нормальной калибровке с номинальным оконным уровнем 5%. "
            "Фактический тестовый FPR показан отдельно; методы работают при разных достигнутых FPR. "
            "Поэтому recall нельзя сравнивать как recall при одинаковом фактическом FPR. "
            "± означает стандартное отклонение по seeds, не доверительный интервал."
        ),
        "",
        "| Метод | AUROC ± SD | Recall, % | FPR нормы, % |",
        "|---|---:|---:|---:|",
    ]
    blocked = first[first.protocol == "blocked"].set_index("method")
    for name in order:
        row = blocked.loc[name]
        text.append(
            f"| {name} | {row.auroc_mean:.4f} ± {row.auroc_std:.4f} | {100 * row.recall_mean:.2f} | {100 * row.normal_fpr_mean:.2f} |"
        )
    text += [
        "",
        "## Смена уставки: AUROC / FPR нормы",
        "",
        (
            "Удерживается целая температурная полоса 40, 45 или 50 °F. "
            "Калибровка остаётся в исходном домене. 45 °F — промежуточная уставка, "
            "40/50 °F — крайние; это не проверка всех 27 исходных режимов и не перенос между установками."
        ),
        "",
        "| Метод | Hold 40 °F | Hold 45 °F | Hold 50 °F |",
        "|---|---:|---:|---:|",
    ]
    for name in order:
        cells = []
        for protocol in protocols[1:]:
            row = first[(first.method == name) & (first.protocol == protocol)].iloc[0]
            cells.append(f"{row.auroc_mean:.3f} / {100 * row.normal_fpr_mean:.1f}%")
        text.append(f"| {name} | " + " | ".join(cells) + " |")
    text += [
        "",
        "![SL1: AUROC и ложные тревоги](sl1_comparison.png)",
        "",
        "## Что действительно следует из эксперимента",
        "",
        (
            "1. Mahalanobis с усадкой — сильнейший из проверенных ориентиров по среднему AUROC "
            "во всех четырёх протоколах. На blocked SL1 AUROC = 0.9413, однако FPR = 10.31% "
            "при номинале 5%. Лидерство по AUROC не означает допустимые эксплуатационные тревоги."
        ),
        (
            "2. Проекция касательного пространства улучшает conditional RBF по AUROC в этих протоколах, "
            "но не обгоняет Mahalanobis. На Hold 50 °F обе условные версии имеют FPR = 100%."
        ),
        (
            "3. Компактный conditional flow и NeuTraL MLP не показали универсального преимущества. "
            "Это конкретные небольшие реализации с фиксированными настройками, а не опровержение "
            "соответствующих семейств или воспроизведение статей FDSD/MTGFlow."
        ),
        (
            "4. Hold 50 °F труден для всех: даже Mahalanobis имеет FPR = 56.07%. "
            "Автоматически считать выход за обучающий режим неисправностью нельзя."
        ),
        (
            "5. При бюджете 1% во всех LOTO-разбиениях недостаточно даже калибровочных окон "
            "для конечного рангового порога; возвращён +∞ и recall равен нулю. "
            "При 5% конечный порог всё равно не получает гарантии из-за зависимости/сдвига распределения."
        ),
        "",
        "## Mahalanobis: отдельные неисправности SL1",
        "",
        "| Неисправность | AUROC | Recall при номинальном пороге 5%, % |",
        "|---|---:|---:|",
    ]
    detailed = pd.read_csv(root / "per_fault.csv")
    selected = detailed[
        (detailed.protocol == "blocked")
        & (detailed.method == "mahalanobis")
        & (detailed.severity == 1)
        & (detailed.alpha == 0.05)
    ]
    for fault, row in selected.groupby("fault")[["auroc", "recall"]].mean().iterrows():
        text.append(f"| {fault} | {row.auroc:.4f} | {100 * row.recall:.2f} |")
    text += [
        "",
        (
            "Обозначения: cf — загрязнение конденсатора; eo — избыток масла; "
            "fwc/fwe — уменьшение расхода воды конденсатора/испарителя; nc — неконденсируемый газ; "
            "rl — утечка хладагента; ro — избыток хладагента."
        ),
        "",
        "## Воспроизведение",
        "",
        "```powershell",
        "uv pip install --python .venv/Scripts/python.exe -r requirements-ashrae.txt",
        "uv pip install --python .venv/Scripts/python.exe -r requirements-neural.txt",
        ".venv/Scripts/python.exe scripts/prepare_ashrae_public.py",
        ".venv/Scripts/python.exe scripts/run_ashrae_benchmark.py",
        ".venv/Scripts/python.exe scripts/summarize_ashrae_benchmark.py",
        "```",
        "",
        (
            "[Паспорт источника](data_manifest.json), [аудит CSV](data_audit.csv), "
            "[замороженный протокол](experiment_manifest.json), [все результаты](per_fault.csv), "
            "[агрегация](summary.csv), [калибровка и FPR](normal_metrics.csv), "
            "[время и потери обучения](training.csv)."
        ),
        "",
        (
            "balanced_ap в таблицах вычислен при искусственном априорном классовом балансе 50/50 "
            "для каждой пары normal/fault. Он не оценивает precision при реальной редкости отказов. "
            "Времена в training.csv включают fit и нормальные оценки, но не полный faulty inference."
        ),
        "",
    ]
    (root / "report.md").write_text("\n".join(text), encoding="utf-8")
    # Keep reporting environment available without changing the Poetry lockfile.
    assert np.isfinite(first.auroc_mean).all()
    print(root / "report.md")


if __name__ == "__main__":
    main()
