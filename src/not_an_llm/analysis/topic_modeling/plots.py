from __future__ import annotations

from pathlib import Path
import textwrap
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from not_an_llm.analysis.trends import TrendAnalyzer


TOPIC_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]

TOPIC_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p", "8", "H", "d"]


def format_xticks(ax):
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def topic_color_map(topic_ids) -> dict[int, str]:
    ordered = sorted(int(topic_id) for topic_id in topic_ids)
    return {
        topic_id: TOPIC_COLORS[index % len(TOPIC_COLORS)]
        for index, topic_id in enumerate(ordered)
    }


def save_legend_only(ax, output_path: Path, ncol: int = 1, wrap_width: int = 44):
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    wrapped_labels = [
        "\n".join(textwrap.wrap(label, width=wrap_width, break_long_words=False)) or label
        for label in labels
    ]
    line_count = sum(max(1, label.count("\n") + 1) for label in wrapped_labels)
    max_line_len = max(
        (len(line) for label in wrapped_labels for line in label.splitlines()),
        default=wrap_width,
    )
    fig_width = max(5.0, min(8.5, 1.6 + max_line_len * 0.085))
    fig_height = max(2.0, 0.34 * line_count + 0.2)

    fig_legend = plt.figure(figsize=(fig_width, fig_height))
    legend = fig_legend.legend(
        handles,
        wrapped_labels,
        loc="center",
        frameon=False,
        ncol=ncol,
        fontsize=8,
        handlelength=1.6,
        labelspacing=0.8,
    )

    fig_legend.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())
    pad_inches = 0.18
    required_width = bbox.width + pad_inches * 2
    required_height = bbox.height + pad_inches * 2
    if required_width > fig_width or required_height > fig_height:
        fig_width = max(fig_width, required_width)
        fig_height = max(fig_height, required_height)
        fig_legend.set_size_inches(fig_width, fig_height)
        fig_legend.canvas.draw()

    fig_legend.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig_legend)


def save_topic_prevalence(
    enriched: pd.DataFrame,
    plot_dir: Path,
    analysis_dir: Path,
    input_stem: str,
    topic_labels: dict[int, str],
) -> list[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if "year" not in enriched.columns:
        raise ValueError("Topic prevalence requires a 'year' column.")

    df = enriched.copy()
    if "month_ts" not in df.columns and "year" in df.columns and "month" in df.columns:
        df["month_ts"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )

    df["count"] = 1
    yearly = df.groupby(["year", "topic_id"], as_index=False)["count"].sum()
    totals = df.groupby("year", as_index=False)["count"].sum().rename(columns={"count": "total"})
    yearly = yearly.merge(totals, on="year")
    yearly["pct"] = yearly["count"] / yearly["total"] * 100
    yearly["topic_label"] = yearly["topic_id"].map(topic_labels)

    yearly_csv = analysis_dir / f"{input_stem}_topic_prevalence_yearly.csv"
    yearly.to_csv(yearly_csv, index=False)
    paths.append(yearly_csv)

    topic_order = (
        yearly.groupby("topic_id")["count"]
        .sum()
        .sort_values(ascending=False)
        .index.tolist()
    )
    color_map = topic_color_map(topic_order)

    yearly_count_pivot = yearly.pivot(index="year", columns="topic_id", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(4, 3))
    bottom = np.zeros(len(yearly_count_pivot.index))
    for topic_id in topic_order:
        if topic_id not in yearly_count_pivot.columns:
            continue
        values = yearly_count_pivot[topic_id].to_numpy()
        label = topic_labels.get(int(topic_id), f"topic_{topic_id}")
        ax.bar(
            yearly_count_pivot.index,
            values,
            bottom=bottom,
            label=label,
            color=color_map[int(topic_id)],
            alpha=0.9,
        )
        bottom += values
    ax.set_xlabel("Year")
    ax.set_ylabel("Abstract count")
    legend = ax.legend(loc="best", fontsize=8)
    save_legend_only(ax, plot_dir / "topic_evolution_stacked_counts_legend.png")
    legend.remove()
    ax.grid(alpha=0.3, axis="y")
    format_xticks(ax)
    stacked_counts_path = plot_dir / "topic_evolution_stacked_counts_yearly.png"
    fig.tight_layout()
    fig.savefig(stacked_counts_path, dpi=150)
    plt.close(fig)
    paths.append(stacked_counts_path)

    yearly_pivot = yearly.pivot(index="year", columns="topic_id", values="pct").fillna(0.0)

    fig, ax = plt.subplots(figsize=(4, 3))
    for marker_index, topic_id in enumerate(topic_order):
        if topic_id not in yearly_pivot.columns:
            continue
        label = topic_labels.get(int(topic_id), f"topic_{topic_id}")
        ax.plot(
            yearly_pivot.index,
            yearly_pivot[topic_id],
            marker=TOPIC_MARKERS[marker_index % len(TOPIC_MARKERS)],
            markersize=4,
            color=color_map[int(topic_id)],
            label=label,
            alpha=0.9,
            linewidth=1,
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("Topic prevalence (%)")
    legend = ax.legend(loc="best", fontsize=8)
    save_legend_only(ax, plot_dir / "topic_prevalence_legend.png")
    legend.remove()
    ax.grid(alpha=0.3)
    format_xticks(ax)
    trend_path = plot_dir / "topic_prevalence_yearly.png"
    fig.tight_layout()
    fig.savefig(trend_path, dpi=150)
    plt.close(fig)
    paths.append(trend_path)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.stackplot(
        yearly_pivot.index,
        *[yearly_pivot[topic_id] for topic_id in topic_order if topic_id in yearly_pivot.columns],
        labels=[topic_labels.get(int(topic_id), f"topic_{topic_id}") for topic_id in topic_order if topic_id in yearly_pivot.columns],
        colors=[color_map[int(topic_id)] for topic_id in topic_order if topic_id in yearly_pivot.columns],
        alpha=0.9,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Topic prevalence (%)")
    legend = ax.legend(loc="best", fontsize=8)
    save_legend_only(ax, plot_dir / "topic_evolution_stacked_legend.png")
    legend.remove()
    ax.grid(alpha=0.3, axis="y")
    format_xticks(ax)
    stacked_path = plot_dir / "topic_evolution_stacked_yearly.png"
    fig.tight_layout()
    fig.savefig(stacked_path, dpi=150)
    plt.close(fig)
    paths.append(stacked_path)

    if "month_ts" in df.columns:
        monthly = df.groupby(["month_ts", "topic_id"], as_index=False)["count"].sum()
        month_totals = df.groupby("month_ts", as_index=False)["count"].sum().rename(columns={"count": "total"})
        monthly = monthly.merge(month_totals, on="month_ts")
        monthly["pct"] = monthly["count"] / monthly["total"] * 100
        monthly["topic_label"] = monthly["topic_id"].map(topic_labels)

        monthly_csv = analysis_dir / f"{input_stem}_topic_prevalence_monthly.csv"
        monthly.to_csv(monthly_csv, index=False)
        paths.append(monthly_csv)

        monthly_pivot = monthly.pivot(index="month_ts", columns="topic_id", values="pct").fillna(0.0)
        fig, ax = plt.subplots(figsize=(4, 3))
        for marker_index, topic_id in enumerate(topic_order):
            if topic_id not in monthly_pivot.columns:
                continue
            label = topic_labels.get(int(topic_id), f"topic_{topic_id}")
            ax.plot(
                pd.to_datetime(monthly_pivot.index),
                monthly_pivot[topic_id],
                marker=TOPIC_MARKERS[marker_index % len(TOPIC_MARKERS)],
                color=color_map[int(topic_id)],
                label=label,
                linewidth=1,
            )
        ax.set_xlabel("Month")
        ax.set_ylabel("Topic prevalence (%)")
        legend = ax.legend(loc="best", fontsize=8)
        save_legend_only(ax, plot_dir / "topic_prevalence_monthly_legend.png")
        legend.remove()
        ax.grid(alpha=0.3)
        format_xticks(ax)
        monthly_path = plot_dir / "topic_prevalence_monthly.png"
        fig.tight_layout()
        fig.savefig(monthly_path, dpi=150)
        plt.close(fig)
        paths.append(monthly_path)

    return paths


def save_topic_trend_plots(
    enriched: pd.DataFrame,
    plot_dir: Path,
    topic_labels: dict[int, str],
    events: dict[str, str],
) -> list[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if "year" not in enriched.columns:
        return paths

    feature_columns = [
        col
        for col in enriched.columns
        if col.endswith("_per_1k_words") and pd.api.types.is_numeric_dtype(enriched[col])
    ]
    if not feature_columns:
        return paths

    grouped = enriched.groupby(["year", "topic_id"])[feature_columns].mean().reset_index()
    grouped["topic_label"] = grouped["topic_id"].map(topic_labels)
    event_dates = {name: pd.to_datetime(value) for name, value in events.items()}
    topic_order = (
        enriched["topic_id"]
        .value_counts()
        .sort_values(ascending=False)
        .index.tolist()
    )
    color_map = topic_color_map(topic_order)

    for feature in feature_columns:
        fig, ax = plt.subplots(figsize=(4, 3))
        for marker_index, topic_id in enumerate(topic_order):
            topic_data = grouped[grouped["topic_id"] == topic_id]
            if topic_data.empty:
                continue
            topic_label = topic_labels.get(topic_id, f"topic_{topic_id}")
            ax.plot(
                topic_data["year"],
                topic_data[feature],
                marker=TOPIC_MARKERS[marker_index % len(TOPIC_MARKERS)],
                color=color_map[int(topic_id)],
                linewidth=1,
                label=topic_label,
                alpha=0.9,
            )

        for event_name, event_date in event_dates.items():
            if event_date.year in grouped["year"].values:
                ax.axvline(x=event_date.year, color="red", linestyle="--", alpha=0.7, label=event_name)

        ax.set_xlabel("Year")
        ax.set_ylabel(feature.replace("_per_1k_words", "").replace("_", " ").replace("Word", "").title())
        legend = ax.legend(loc="best", fontsize=8)
        save_legend_only(ax, plot_dir / f"{feature}_by_topic_legend.png")
        legend.remove()
        ax.grid(alpha=0.3)
        format_xticks(ax)

        plot_path = plot_dir / f"{feature}_by_topic.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        paths.append(plot_path)

    return paths


def save_topic_cluster_plot(embeddings_2d: pd.DataFrame | None, plot_dir: Path) -> Path | None:
    if embeddings_2d is None or embeddings_2d.empty:
        return None

    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))

    topic_sizes = embeddings_2d["topic_id"].value_counts()
    topics = [topic for topic in topic_sizes.index.tolist() if topic != -1]
    topics = sorted(topics, key=lambda topic: topic_sizes[topic], reverse=True)
    color_map = topic_color_map(topics)

    noise_data = embeddings_2d[embeddings_2d["topic_id"] == -1]
    if not noise_data.empty:
        ax.scatter(noise_data["x"], noise_data["y"], c="lightgray", alpha=0.35, s=10, label="noise")

    for topic_id in topics:
        topic_data = embeddings_2d[embeddings_2d["topic_id"] == topic_id]
        if topic_data.empty:
            continue
        topic_label = topic_data["topic_label"].iloc[0]
        ax.scatter(
            topic_data["x"],
            topic_data["y"],
            c=[color_map[topic_id]],
            alpha=0.7,
            s=15,
            label=f"{topic_id}: {topic_label}",
            edgecolors="black",
            linewidth=0.2,
        )

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    legend = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    save_legend_only(ax, plot_dir / "topic_clusters_legend.png")
    legend.remove()
    ax.grid(alpha=0.3)

    plot_path = plot_dir / "topic_clusters.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def save_per_topic_trend_outputs(
    enriched: pd.DataFrame,
    plot_dir: Path,
    analysis_topic_base: Path,
    trend_analyzer: "TrendAnalyzer",
    group_specs: dict[str, dict[str, object]],
    summary_features: set[str],
    events: dict[str, str],
) -> list[Path]:
    paths: list[Path] = []
    unique_labels = sorted(enriched["topic_id"].dropna().unique())

    for topic_id in unique_labels:
        plot_topic_dir = plot_dir / f"topic_{topic_id}"
        plot_topic_dir.mkdir(parents=True, exist_ok=True)
        analysis_topic_dir = analysis_topic_base / f"topic_{topic_id}"
        analysis_topic_dir.mkdir(parents=True, exist_ok=True)

        topic_subset = enriched[enriched["topic_id"] == topic_id]
        if topic_subset.empty:
            continue

        topic_yearly = trend_analyzer.aggregate_yearly(topic_subset)
        topic_monthly = trend_analyzer.aggregate_monthly(topic_subset)
        topic_yearly.to_csv(analysis_topic_dir / "trends_by_year.csv", index=False)
        topic_monthly.to_csv(analysis_topic_dir / "trends_by_month.csv", index=False)

        stats_path = analysis_topic_dir / "feature_stats.csv"
        trend_analyzer.compute_all_stats(topic_yearly).to_csv(stats_path, index=False)
        paths.append(stats_path)

        paths.extend(
            trend_analyzer.save_plots(
                topic_yearly,
                topic_monthly,
                plot_topic_dir,
                events,
                exclude_features=summary_features,
            )
        )
        if hasattr(trend_analyzer, "save_grouped_feature_diffs"):
            paths.extend(trend_analyzer.save_grouped_feature_diffs(topic_yearly, output_dir=plot_topic_dir).values())
        paths.extend(
            trend_analyzer.save_grouped_word_plots(
                yearly=topic_yearly,
                monthly=topic_monthly,
                output_dir=plot_topic_dir,
                group_specs=group_specs,
                events=events,
                smoothing_window=3,
            )
        )
        paths.append(
            trend_analyzer.save_word_prefix_stack_plot(
                yearly=topic_yearly,
                monthly=topic_monthly,
                output_dir=plot_topic_dir,
                events=events,
                smoothing_window=3,
            )
        )
        paths.extend(
            trend_analyzer.save_dependency_distribution_plot(
                df=topic_subset,
                output_dir=plot_topic_dir,
                events=events,
            )
        )
        paths.append(
            trend_analyzer.save_stacked_word_plots(
                yearly=topic_yearly,
                monthly=topic_monthly,
                output_dir=plot_topic_dir,
                events=events,
                smoothing_window=3,
                exclude_features=summary_features,
            )
        )

    return paths
