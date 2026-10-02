"""Shared figures and captions for the Markdown report and presentation."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


DETECTOR_NAMES = {
    "bayesian_runlength": "Байесовский BOCPD",
    "discounted_cusum": "CUSUM с затуханием",
    "robust_innovation": "Робастный скачок",
    "window_glr": "Оконный тест",
}
RISK_NAMES = {
    "constant_prior": "Постоянная вероятность",
    "history_only": "История расходов",
    "history_and_news": "История + новости",
}


def plot_real_cases(panel, predictions, category, test_start, output):
    """Show observed values, missing months and forecasts without interpolation."""
    difficulty = predictions.groupby("territory_id").absolute_error.mean().sort_values(kind="stable")
    positions = [len(difficulty)//2, int(.9*(len(difficulty)-1)), len(difficulty)-1]
    labels = ["Медианная MAE", "90-й перцентиль", "Наибольшая MAE"]
    ci = panel.categories.index(category)
    dates = panel.dates
    navy, green = "#263b44", "#007d66"
    fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.25))
    cases = []
    for ax, position, label in zip(axes, positions, labels):
        tid = int(difficulty.index[position])
        ix = int(np.flatnonzero(panel.ids == tid)[0])
        values = panel.values[ix, ci]
        forecast = predictions[predictions.territory_id == tid].sort_values("target_date")
        ax.axvspan(pd.Timestamp(test_start)-pd.Timedelta(days=15),
                   dates[-1]+pd.Timedelta(days=15), color="#cde9df", alpha=.6, zorder=0)
        for month in dates[~np.isfinite(values)]:
            ax.axvspan(month-pd.Timedelta(days=14), month+pd.Timedelta(days=15),
                       facecolor="#e2e5e4", edgecolor="#a8b2ad", hatch="////",
                       linewidth=0, alpha=.55, zorder=1)
        # NaNs break the line; markers preserve isolated observations.
        ax.plot(dates, values/1000, "o-", color=navy, lw=1.8, ms=3.8, zorder=3)
        ax.plot(pd.to_datetime(forecast.target_date), forecast.prediction/1000,
                "D--", color=green, lw=1.8, ms=5, zorder=4)
        title = "пропуски в истории" if label == labels[-1] and not np.isfinite(values).all() else label[0].lower()+label[1:]
        ax.set_title(f"МО {tid} | {title}", fontsize=11, pad=13)
        ax.set_xlim(dates[0]-pd.Timedelta(days=16), dates[-1]+pd.Timedelta(days=19))
        tick_dates = dates[np.unique(np.r_[np.arange(0, len(dates), 6), len(dates)-1])]
        month_names = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
        ax.set_xticks(tick_dates, [f"{month_names[d.month-1]} {d.year%100:02d}" for d in tick_dates],
                      rotation=28, ha="right", fontsize=9)
        ax.set_ylabel("Расходы, тыс. ₽", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(alpha=.13, zorder=0)
        count = len(forecast)
        observed = int(np.isfinite(values).sum())
        mae = float(forecast.absolute_error.mean())
        caption = f"Факты: {observed}/{len(dates)} мес. | Оценено: {count} мес.\nMAE = {mae:,.0f} ₽".replace(",", " ")
        ax.text(.5, -.30, caption, transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color=navy, linespacing=1.5)
        cases.append({"territory_id": tid, "selection_rule": label,
                      "observations": observed, "history_months": len(dates),
                      "test_months": count, "test_mae_h1": mae})
    legend = [
        Line2D([], [], color=navy, marker="o", lw=1.6, ms=4, label="Факт"),
        Line2D([], [], color=green, marker="D", ls="--", lw=1.6, ms=4, label="Прогноз на 1 месяц"),
        Patch(facecolor="#e2e5e4", edgecolor="#a8b2ad", hatch="////", label="Нет наблюдений"),
        Patch(facecolor="#cde9df", edgecolor="none", label="Период оценки"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, 1.02), ncol=4, frameon=False, fontsize=10)
    fig.subplots_adjust(left=.052, right=.99, top=.83, bottom=.28, wspace=.30)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return cases


def case_explanation(cases):
    """Use the same observed counts and MAE in every output."""
    worst = cases[-1]
    missing = worst["history_months"]-worst["observations"]
    mae = f"{worst['test_mae_h1']:,.0f}".replace(",", "\u00a0")
    text = (f"МО {worst['territory_id']}: пропущено {missing} из {worst['history_months']} месяцев; "
            f"пропуски отмечены штриховкой, отдельные факты - точками. "
            f"Оценено прогнозов: {worst['test_months']}; MAE = {mae} ₽. "
            "Это наибольшая MAE среди МО по доступным прогнозам. ")
    others = [c["test_months"] for c in cases[:-1]]
    if len(set(others)) == 1:
        text += f"У двух других примеров оценено по {others[0]} прогноза."
    else:
        text += "Число оценённых прогнозов указано под каждым графиком."
    return text


def plot_detection(detection, output):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), sharey=True)
    for ax, kind, title in zip(axes, ["step", "ramp", "temporary"],
                               ["Устойчивый скачок", "Плавный сдвиг", "Временный импульс"]):
        group = detection[detection.kind == kind].groupby(["model", "magnitude"]).recall.mean().unstack(0)
        for model in group.columns:
            ax.plot(group.index*100, group[model], "o-", label=DETECTOR_NAMES[model])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Амплитуда изменения, %", fontsize=10)
        ax.set_xticks([5, 10, 20])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Доля обнаруженных событий", fontsize=10)
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
