from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_topic_modeling_stats(
    output_path: Path,
    *,
    enriched: pd.DataFrame,
    summary: pd.DataFrame,
) -> Path:
    total_abstracts = len(enriched)
    if "topic_id" in summary.columns:
        non_outlier_summary = summary[summary["topic_id"] != -1]
        outlier_summary = summary[summary["topic_id"] == -1]
        topics = non_outlier_summary["topic_id"].nunique()
        outlier_count = int(outlier_summary["abstract_count"].sum()) if not outlier_summary.empty else 0
        outlier_share = outlier_count / total_abstracts if total_abstracts else 0.0
    else:
        topics = 0
        outlier_count = 0
        outlier_share = 0.0

    rows: list[dict[str, object]] = [
        _stats_row("dataset", "all", None, None, "total_abstracts", total_abstracts),
        _stats_row("dataset", "all", None, None, "topics", topics),
        _stats_row("dataset", "all", -1, "-1 (outliers)", "outlier_abstract_count", outlier_count),
        _stats_row("dataset", "all", -1, "-1 (outliers)", "outlier_abstract_share", outlier_share),
    ]

    rows.extend(_topic_stats_rows(summary, "model"))

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
    return rows
