from __future__ import annotations

from pathlib import Path
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from not_an_llm.analysis.interrupted_time_series import compute_interrupted_time_series
from not_an_llm.analysis.visual_style import (
    DATASET_COLORS,
    MARKERS,
    WHITE,
)
from compare_corpus_trends import (
    DEFAULT_PRIMARY_DATASETS,
    build_manuscript_feature_table,
    load_all_its,
)


DATASETS = (
    ("arxiv_ai_abstracts", "AI"),
    ("arxiv_qbio_abstracts", "Quantitative biology"),
    ("arxiv_stat_abstracts", "Statistics"),
    ("medarxiv_abstracts", "medRxiv"),
)
LABELS = {
    "word_across_per_1k_words": r"$\it{across}$",
    "verb_align_per_1k_words": r"$\it{align}$",
    "em_dash_per_1k_words": "em dash",
    "avg_syllables_per_word": "syll./word",
    "flesch_reading_ease": "FRE",
    "flesch_kincaid_grade": "FKGL",
    "clause_depth": "max tree depth",
    "clause_depth_std": r"tree depth $\sigma$",
    "dependency_entropy": "dep. entropy",
    "hedge_ratio": "hedging",
    "sequential_marker_moreover_per_1k_words": r"$\it{moreover}$",
    "emphasis_marker_crucially_per_1k_words": r"$\it{crucially}$",
}

EXAMPLE_TRAJECTORY_FEATURES = (
    "word_across_per_1k_words",
    "word_insight_per_1k_words",
    "verb_delve_per_1k_words",
    "avg_syllables_per_word",
    "clause_depth",
)
EXAMPLE_TRAJECTORY_LABELS = {
    "word_across_per_1k_words": r"$\it{across}$",
    "word_insight_per_1k_words": r"$\it{insight}$",
    "verb_delve_per_1k_words": r"$\it{delve}$",
    "avg_syllables_per_word": "syllables/word",
    "clause_depth": "max tree depth",
}
EXAMPLE_TOPIC_COLORS = tuple(plt.get_cmap("tab10").colors[:9])


def select_features(analysis_dir: Path) -> tuple[str, ...]:
    stats = load_all_its(analysis_dir, list(DEFAULT_PRIMARY_DATASETS))
    table, _ = build_manuscript_feature_table(stats, list(DEFAULT_PRIMARY_DATASETS))
    selected: list[str] = []
    for _, group in table.groupby("manuscript_group", sort=False):
        selected.extend(group.head(2)["feature"].astype(str))
    return tuple(selected)


def compute_topic_effects(analysis_dir: Path, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for dataset, _ in DATASETS:
        topic_root = analysis_dir / dataset / "topics"
        summary_path = analysis_dir / dataset / "topic_summary.csv"
        labels = {}
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            labels = summary.set_index("topic_id")["topic_label"].astype(str).to_dict()
        for topic_dir in sorted(topic_root.glob("topic_*")):
            try:
                topic_id = int(topic_dir.name.removeprefix("topic_"))
            except ValueError:
                continue
            if topic_id < 0:
                continue
            monthly_path = topic_dir / "trends_by_month.csv"
            if not monthly_path.exists():
                continue
            fitted = compute_interrupted_time_series(pd.read_csv(monthly_path), list(features))
            if fitted.empty:
                continue
            fitted = fitted[fitted["feature"].isin(features)].copy()
            fitted.insert(0, "topic_label", labels.get(topic_id, f"Topic {topic_id}"))
            fitted.insert(0, "topic_id", topic_id)
            fitted.insert(0, "dataset", dataset)
            rows.append(fitted)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def plot(topic_effects: pd.DataFrame, analysis_dir: Path, output_dir: Path, features: tuple[str, ...]) -> None:
    corpus_rows = []
    for dataset, _ in DATASETS:
        stats = pd.read_csv(analysis_dir / dataset / "its_stats.csv")
        stats = stats[stats["feature"].isin(features)].copy()
        stats.insert(0, "dataset", dataset)
        corpus_rows.append(stats)
    corpus_effects = pd.concat(corpus_rows, ignore_index=True)

    all_values = pd.concat([
        topic_effects["standardized_slope_change_per_year"],
        corpus_effects["standardized_slope_change_per_year"],
    ]).pipe(pd.to_numeric, errors="coerce")
    limit = max(1.0, float(np.nanmax(np.abs(all_values))) * 1.08)
    x = np.arange(len(features))
    fig, axes = plt.subplots(2, 2, figsize=(6.0, 3.9), sharex=True, sharey=True)
    for ax, (dataset, title) in zip(axes.flat, DATASETS, strict=True):
        color = DATASET_COLORS[dataset]
        for feature_index, feature in enumerate(features):
            points = topic_effects[
                (topic_effects["dataset"] == dataset) & (topic_effects["feature"] == feature)
            ].sort_values("topic_id")
            values = pd.to_numeric(points["standardized_slope_change_per_year"], errors="coerce").dropna()
            offsets = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.array([0.0])
            ax.scatter(
                feature_index + offsets, values, s=11, facecolors=WHITE,
                edgecolors=color, linewidths=0.75, zorder=2,
            )
            corpus = corpus_effects[
                (corpus_effects["dataset"] == dataset) & (corpus_effects["feature"] == feature)
            ]
            if not corpus.empty:
                ax.scatter(
                    feature_index,
                    float(corpus.iloc[0]["standardized_slope_change_per_year"]),
                    s=27, marker="D", color=color, edgecolor="black", linewidth=0.35, zorder=3,
                )
        ax.axhline(0, color="black", linewidth=0.65)
        ax.set_title(title, fontsize=8.5, color="black", pad=2)
        ax.set_ylim(-limit, limit)
        ax.grid(axis="y", alpha=0.22, linewidth=0.5)
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
    for ax in axes[1]:
        ax.set_xticks(x, [LABELS.get(feature, feature) for feature in features], rotation=38, ha="right", fontsize=6)
    fig.supylabel(r"Standardized $\Delta$ slope/year", fontsize=7)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.19, top=0.96, wspace=0.08, hspace=0.16)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "topic_feature_heterogeneity.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.025)
    plt.close(fig)

    legend_fig, legend_ax = plt.subplots(figsize=(2.45, 0.28))
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], marker="o", markerfacecolor=WHITE, markeredgecolor="black",
                   linestyle="none", markersize=3.8, label="Topic"),
            Line2D([0], [0], marker="D", markerfacecolor="black", markeredgecolor="black",
                   linestyle="none", markersize=4.2, label="Corpus level"),
        ],
        loc="center", ncol=2, frameon=False, fontsize=6.5,
        handletextpad=0.35, columnspacing=1.1,
    )
    legend_fig.savefig(
        output_dir / "topic_feature_heterogeneity_legend.png",
        dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.01,
    )
    plt.close(legend_fig)


def plot_heatmap(topic_effects: pd.DataFrame, output_dir: Path, features: tuple[str, ...]) -> None:
    """Four compact within-corpus topic-by-feature heatmaps."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from not_an_llm.analysis.visual_style import ORCHID_GREEN_DIVERGING_CMAP

    values = pd.to_numeric(topic_effects["standardized_slope_change_per_year"], errors="coerce")
    limit = max(1.0, float(np.nanmax(np.abs(values))))
    fig, axes = plt.subplots(2, 2, figsize=(5.35, 3.55))
    for ax, (dataset, title) in zip(axes.flat, DATASETS, strict=True):
        subset = topic_effects[topic_effects["dataset"] == dataset].copy()
        matrix = subset.pivot_table(
            index="topic_id", columns="feature",
            values="standardized_slope_change_per_year", aggfunc="first",
        ).reindex(columns=features).sort_index()
        ax.imshow(
            matrix.to_numpy(dtype=float), aspect="auto",
            cmap=ORCHID_GREEN_DIVERGING_CMAP, vmin=-limit, vmax=limit,
        )
        ax.set_title(title, fontsize=8.5, color="black", pad=2)
        ax.set_yticks(range(len(matrix)), [f"T{int(topic)}" for topic in matrix.index])
        ax.tick_params(axis="y", labelsize=5.8, length=0)
        ax.set_xticks(range(len(features)))
        if ax in axes[1]:
            ax.set_xticklabels([LABELS.get(feature, feature) for feature in features], rotation=42, ha="right", fontsize=5.6)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        for row in range(len(matrix)):
            for column in range(len(features)):
                value = matrix.iat[row, column]
                if np.isfinite(value):
                    ax.text(column, row, f"{value:.1f}", ha="center", va="center", fontsize=5.4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    fig.subplots_adjust(left=0.058, right=0.995, bottom=0.205, top=0.96, wspace=0.105, hspace=0.115)
    path = output_dir / "topic_feature_heterogeneity_heatmap.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.02)
    plt.close(fig)

    legend_fig, legend_ax = plt.subplots(figsize=(0.48, 2.1))
    legend_ax.axis("off")
    colorbar = legend_fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=-limit, vmax=limit), cmap=ORCHID_GREEN_DIVERGING_CMAP),
        ax=legend_ax, orientation="vertical", fraction=0.72, pad=0.0,
    )
    colorbar.set_label(r"Standardized $\Delta$ slope/year", fontsize=6, labelpad=3)
    colorbar.ax.tick_params(labelsize=5.5, length=2)
    legend_fig.savefig(
        output_dir / "topic_feature_heterogeneity_heatmap_legend.png",
        dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.01,
    )
    plt.close(legend_fig)


def plot_topic_example_trajectories(analysis_dir: Path, output_dir: Path) -> None:
    """Plot the five main example trajectories by topic in each corpus."""
    features = EXAMPLE_TRAJECTORY_FEATURES
    fig, axes = plt.subplots(
        len(DATASETS), len(features), figsize=(7.5, 6.0),
        sharex=True, squeeze=False,
    )
    legend_payload: list[tuple[str, list[Line2D]]] = []

    for row, (dataset, dataset_label) in enumerate(DATASETS):
        labels_path = analysis_dir / dataset / "topic_labels.csv"
        labels_frame = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame()
        labels = (
            labels_frame.set_index("topic_id")["topic_label"].astype(str).to_dict()
            if not labels_frame.empty else {}
        )
        topic_ids = sorted(topic_id for topic_id in labels if int(topic_id) >= 0)
        handles: list[Line2D] = []

        for topic_id in topic_ids:
            yearly_path = (
                analysis_dir / dataset / "topics" / f"topic_{int(topic_id)}" /
                "trends_by_year.csv"
            )
            if not yearly_path.exists():
                continue
            yearly = pd.read_csv(yearly_path)
            years = pd.to_numeric(yearly["year"], errors="coerce")
            color = EXAMPLE_TOPIC_COLORS[int(topic_id) % len(EXAMPLE_TOPIC_COLORS)]
            marker = MARKERS[int(topic_id) % len(MARKERS)]
            for column, feature in enumerate(features):
                value_column = f"{feature}_yearly_mean"
                if value_column not in yearly:
                    continue
                values = pd.to_numeric(yearly[value_column], errors="coerce")
                axes[row, column].plot(
                    years, values, color=color, marker=marker, lw=0.75,
                    markersize=2.0, markeredgewidth=0.25,
                )
            short_label = ", ".join(str(labels.get(topic_id, "")).split(", ")[:5])
            legend_label = textwrap.fill(
                f"T{int(topic_id)}: {short_label}",
                width=30,
                subsequent_indent="    ",
            )
            handles.append(Line2D(
                [0], [0], color=color, marker=marker, lw=0.9, markersize=3.0,
                label=legend_label,
            ))

        legend_payload.append((dataset_label, handles))
        for column, feature in enumerate(features):
            ax = axes[row, column]
            ax.axvline(2022.9, color="0.25", lw=0.6, ls="--")
            ax.grid(alpha=0.22, linewidth=0.45)
            ax.set_xlim(2015, 2026.15)
            ax.set_xticks([2016, 2019, 2022, 2025])
            ax.tick_params(axis="both", labelsize=5.2, length=2, pad=1.3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
            if row == 0:
                ax.set_title(EXAMPLE_TRAJECTORY_LABELS[feature], fontsize=7.0, pad=3)
            if column == 0:
                ax.set_ylabel(dataset_label, fontsize=6.5, labelpad=5)
            if row == len(DATASETS) - 1:
                ax.set_xlabel("Year", fontsize=5.8)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            else:
                ax.tick_params(axis="x", labelbottom=False)

    fig.subplots_adjust(
        left=0.075, right=0.995, bottom=0.09, top=0.96,
        wspace=0.38, hspace=0.25,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "topic_example_trajectories.png"
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor=WHITE, pad_inches=0.02)
    plt.close(fig)

    corpus_topic_text: list[list[str]] = []
    for _, handles in legend_payload:
        labels_for_corpus: list[str] = []
        for topic_id, handle in enumerate(handles):
            raw_label = " ".join(handle.get_label().split())
            prefix = f"T{topic_id}: "
            if raw_label.startswith(prefix):
                raw_label = raw_label[len(prefix):]
            labels_for_corpus.append(textwrap.fill(raw_label, width=25))
        corpus_topic_text.append(labels_for_corpus)

    row_line_counts = [
        max(corpus_topic_text[column][topic_id].count("\n") + 1 for column in range(4))
        for topic_id in range(9)
    ]
    height_ratios = [0.58] + [0.46 * count for count in row_line_counts]
    legend_fig = plt.figure(figsize=(6.25, 2.55))
    grid = legend_fig.add_gridspec(
        10, 5,
        width_ratios=[0.38, 1.0, 1.08, 1.08, 1.18],
        height_ratios=height_ratios,
        left=0.004, right=0.998, bottom=0.01, top=0.99,
        wspace=0.035, hspace=0.0,
    )

    headers = ["Topic"] + [label for label, _ in legend_payload]
    for column, header in enumerate(headers):
        ax = legend_fig.add_subplot(grid[0, column])
        ax.axis("off")
        ax.text(0.5, 0.5, header, ha="center", va="center", fontsize=7.6)

    for topic_id in range(9):
        marker_ax = legend_fig.add_subplot(grid[topic_id + 1, 0])
        marker_ax.axis("off")
        color = EXAMPLE_TOPIC_COLORS[topic_id]
        marker = MARKERS[topic_id % len(MARKERS)]
        marker_ax.plot([0.03, 0.30], [0.72, 0.72], color=color, lw=0.9,
                       marker=marker, markersize=3.2, markevery=[1])
        marker_ax.text(0.40, 0.72, f"T{topic_id}", ha="left", va="center", fontsize=6.3)
        marker_ax.set_xlim(0, 1)
        marker_ax.set_ylim(0, 1)

        for corpus_column in range(4):
            text_ax = legend_fig.add_subplot(grid[topic_id + 1, corpus_column + 1])
            text_ax.axis("off")
            text_ax.text(
                0.0, 0.94, corpus_topic_text[corpus_column][topic_id],
                ha="left", va="top", fontsize=6.3, linespacing=1.05,
            )
    legend_fig.savefig(
        output_dir / "topic_example_trajectories_legend.png",
        dpi=400, bbox_inches="tight", facecolor=WHITE, pad_inches=0.02,
    )
    plt.close(legend_fig)


def main() -> None:
    analysis_dir = ROOT / "data" / "analysis"
    output_dir = ROOT / "data" / "visuals" / "corpus_comparison"
    features = select_features(analysis_dir)
    effects = compute_topic_effects(analysis_dir, features)
    if effects.empty:
        raise SystemExit("No topic-level ITS estimates could be computed.")
    effects.to_csv(analysis_dir / "corpus_comparison" / "topic_feature_heterogeneity.csv", index=False)
    plot_heatmap(effects, output_dir, features)
    plot_topic_example_trajectories(analysis_dir, output_dir)
    (analysis_dir / "corpus_comparison" / "topic_feature_heterogeneity_selection.txt").write_text(
        "\n".join(features) + "\n", encoding="utf-8"
    )
    print(output_dir / "topic_feature_heterogeneity_heatmap.png")


if __name__ == "__main__":
    main()
