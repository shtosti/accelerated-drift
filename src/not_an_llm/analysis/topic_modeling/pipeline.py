from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from not_an_llm.analysis.topic_modeling.model import assign_topics, save_merge_candidates, save_topic_labels
from not_an_llm.analysis.topic_modeling.plots import (
    save_per_topic_trend_outputs,
    save_topic_cluster_plot,
    save_topic_prevalence,
    save_topic_trend_plots,
)
from not_an_llm.analysis.topic_modeling.selection import filter_topics
from not_an_llm.analysis.trends import TrendAnalyzer
from not_an_llm.config import AppConfig


logger = logging.getLogger(__name__)


def run_topic_modeling(
    enriched: pd.DataFrame,
    config: AppConfig,
    analysis_dir: Path,
) -> tuple[pd.DataFrame, dict[int, str], pd.DataFrame | None, list[Path]]:
    if not config.analysis.topic_modeling_enabled:
        return enriched, {}, None, []

    paths: list[Path] = []
    input_stem = config.analysis.preprocessed_jsonl.stem
    result = assign_topics(enriched, config)

    labels_path = save_topic_labels(result.topic_labels, analysis_dir / f"{input_stem}_topic_labels.csv")
    paths.append(labels_path)

    selection = filter_topics(
        result.enriched,
        result.topic_labels,
        result.embeddings_2d,
        min_share=config.analysis.topic_modeling_min_topic_share,
        min_count=config.analysis.topic_modeling_min_topic_count,
        merge_candidates=result.merge_candidates,
        merge_under_threshold=config.analysis.topic_modeling_merge_under_threshold,
        max_final_topics=config.analysis.topic_modeling_max_final_topics,
        final_label_top_terms=config.analysis.topic_modeling_top_n_terms,
    )
    initial_selection_path = analysis_dir / f"{input_stem}_initial_topic_selection.csv"
    selection.initial_summary.to_csv(initial_selection_path, index=False)
    paths.append(initial_selection_path)

    merge_plan_path = analysis_dir / f"{input_stem}_topic_merge_plan.csv"
    selection.merge_plan.to_csv(merge_plan_path, index=False)
    paths.append(merge_plan_path)

    final_labels_path = save_topic_labels(
        selection.topic_labels,
        analysis_dir / f"{input_stem}_final_topic_labels.csv",
    )
    paths.append(final_labels_path)

    selection_path = analysis_dir / f"{input_stem}_topic_selection.csv"
    selection.summary.to_csv(selection_path, index=False)
    paths.append(selection_path)

    merge_candidates = _annotate_merge_candidates(result.merge_candidates, selection.summary)
    merge_path = save_merge_candidates(
        merge_candidates,
        analysis_dir / f"{input_stem}_topic_merge_candidates.csv",
    )
    if merge_path is not None:
        paths.append(merge_path)

    topic_labels = {
        int(topic): selection.enriched.loc[selection.enriched["topic_id"] == int(topic), "topic_label"].iloc[0]
        for topic in selection.enriched["topic_id"].unique()
    }
    return selection.enriched, topic_labels, selection.embeddings_2d, paths


def run_topic_analysis(
    enriched: pd.DataFrame,
    config: AppConfig,
    plot_dir: Path,
    trend_analyzer: TrendAnalyzer,
    group_specs: dict[str, dict[str, object]],
    summary_features: set[str],
    events: dict[str, str],
    embeddings_2d: pd.DataFrame | None = None,
) -> list[Path]:
    logger.info("Starting topic analysis...")
    if not config.analysis.topic_modeling_enabled:
        logger.info("Topic modeling disabled, skipping topic analysis")
        return []

    if "topic_id" not in enriched.columns:
        logger.error("topic_id column missing from enriched data, skipping topic analysis")
        return []

    logger.info("Running topic analysis for %d selected topics", enriched["topic_id"].nunique())

    topic_plot_dir = plot_dir / "topics"
    topic_plot_dir.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(enriched["topic_id"].dropna().unique())
    topic_labels = {
        int(topic): enriched.loc[enriched["topic_id"] == int(topic), "topic_label"].iloc[0]
        for topic in unique_labels
    }

    analysis_dir = Path(config.data_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_topic_base = analysis_dir / f"{config.analysis.preprocessed_jsonl.stem}_topics"
    analysis_topic_base.mkdir(parents=True, exist_ok=True)

    paths = save_topic_prevalence(
        enriched,
        topic_plot_dir,
        analysis_dir,
        config.analysis.preprocessed_jsonl.stem,
        topic_labels,
    )
    paths.extend(save_topic_trend_plots(enriched, topic_plot_dir, topic_labels, events))

    cluster_plot_path = save_topic_cluster_plot(embeddings_2d, topic_plot_dir)
    if cluster_plot_path:
        paths.append(cluster_plot_path)

    paths.extend(
        save_per_topic_trend_outputs(
            enriched=enriched,
            plot_dir=topic_plot_dir,
            analysis_topic_base=analysis_topic_base,
            trend_analyzer=trend_analyzer,
            group_specs=group_specs,
            summary_features=summary_features,
            events=events,
        )
    )

    return paths


def _annotate_merge_candidates(
    merge_candidates: pd.DataFrame | None,
    topic_selection: pd.DataFrame,
) -> pd.DataFrame | None:
    if merge_candidates is None or merge_candidates.empty or topic_selection.empty:
        return merge_candidates

    summary = topic_selection.set_index("topic_id")
    annotated = merge_candidates.copy()
    for column, prefix in (
        ("Child_Left_ID", "child_left"),
        ("Child_Right_ID", "child_right"),
        ("Parent_ID", "parent"),
    ):
        if column not in annotated.columns:
            continue
        annotated[f"{prefix}_abstract_count"] = annotated[column].map(summary["abstract_count"])
        annotated[f"{prefix}_abstract_share"] = annotated[column].map(summary["abstract_share"])
        annotated[f"{prefix}_selected"] = annotated[column].map(summary["selected"])

    child_selected_columns = [
        column
        for column in ("child_left_selected", "child_right_selected")
        if column in annotated.columns
    ]
    if child_selected_columns:
        annotated["merge_priority"] = annotated[child_selected_columns].apply(
            lambda row: "review_under_threshold_child" if not all(row.fillna(False)) else "review_semantic_merge",
            axis=1,
        )

    return annotated
