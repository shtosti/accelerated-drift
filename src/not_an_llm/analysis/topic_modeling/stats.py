from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_topic_modeling_stats(
    output_path: Path,
    *,
    initial_enriched: pd.DataFrame,
    final_enriched: pd.DataFrame,
    initial_summary: pd.DataFrame,
    final_summary: pd.DataFrame,
    merge_plan: pd.DataFrame,
) -> Path:
    total_abstracts = len(initial_enriched)
    kept_abstracts = len(final_enriched)
    dropped_abstracts = total_abstracts - kept_abstracts
    initial_topics = initial_summary["topic_id"].nunique() if "topic_id" in initial_summary.columns else 0
    final_topics = final_summary["topic_id"].nunique() if "topic_id" in final_summary.columns else 0
    under_threshold_initial_topics = (
        int((~initial_summary["selected"]).sum())
        if "selected" in initial_summary.columns
        else 0
    )

    rows: list[dict[str, object]] = [
        _stats_row("dataset", "all", None, None, "total_abstracts", total_abstracts),
        _stats_row("dataset", "all", None, None, "initial_topics", initial_topics),
        _stats_row("dataset", "all", None, None, "final_topics", final_topics),
        _stats_row("dataset", "all", None, None, "kept_abstracts_after_topic_modeling", kept_abstracts),
        _stats_row("dataset", "all", None, None, "dropped_abstracts_under_threshold", dropped_abstracts),
        _stats_row("dataset", "all", None, None, "kept_abstract_share", kept_abstracts / total_abstracts if total_abstracts else 0.0),
        _stats_row("dataset", "all", None, None, "dropped_abstract_share", dropped_abstracts / total_abstracts if total_abstracts else 0.0),
        _stats_row("dataset", "all", None, None, "under_threshold_initial_topics", under_threshold_initial_topics),
        _stats_row("merge", "all", None, None, "merge_steps", len(merge_plan)),
    ]

    rows.extend(_topic_stats_rows(initial_summary, "initial"))
    rows.extend(_topic_stats_rows(final_summary, "final"))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def _stats_row(
    scope: str,
    stage: str,
    topic_id: int | None,
    topic_label: str | None,
    metric: str,
    value: object,
) -> dict[str, object]:
    return {
        "scope": scope,
        "stage": stage,
        "topic_id": topic_id,
        "topic_label": topic_label,
        "metric": metric,
        "value": value,
    }


def _topic_stats_rows(summary: pd.DataFrame, stage: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if summary.empty:
        return rows

    for topic in summary.itertuples(index=False):
        topic_id = int(topic.topic_id)
        topic_label = str(topic.topic_label)
        rows.append(_stats_row("topic", stage, topic_id, topic_label, "abstract_count", topic.abstract_count))
        rows.append(_stats_row("topic", stage, topic_id, topic_label, "abstract_share", topic.abstract_share))
        rows.append(_stats_row("topic", stage, topic_id, topic_label, "selected", topic.selected))
    return rows
