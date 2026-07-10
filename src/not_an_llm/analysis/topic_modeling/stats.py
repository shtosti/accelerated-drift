from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_topic_modeling_stats(
    output_path: Path,
    *,
    enriched: pd.DataFrame,
    summary: pd.DataFrame,
    text_source: str = "title_abstract",
) -> Path:
    text_label = "title" if text_source == "title" else "abstract"
    count_col = f"{text_label}_count"
    share_col = f"{text_label}_share"
    total_documents = len(enriched)
    if "topic_id" in summary.columns:
        non_outlier_summary = summary[summary["topic_id"] != -1]
        outlier_summary = summary[summary["topic_id"] == -1]
        topics = non_outlier_summary["topic_id"].nunique()
        outlier_count = int(outlier_summary[count_col].sum()) if not outlier_summary.empty else 0
        outlier_share = outlier_count / total_documents if total_documents else 0.0
    else:
        topics = 0
        outlier_count = 0
        outlier_share = 0.0

    rows: list[dict[str, object]] = [
        _stats_row("dataset", "all", None, None, f"total_{text_label}s", total_documents),
        _stats_row("dataset", "all", None, None, "topics", topics),
        _stats_row("dataset", "all", -1, "-1 (outliers)", f"outlier_{text_label}_count", outlier_count),
        _stats_row("dataset", "all", -1, "-1 (outliers)", f"outlier_{text_label}_share", outlier_share),
    ]

    rows.extend(_topic_stats_rows(summary, "model", count_col, share_col))

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


def _topic_stats_rows(
    summary: pd.DataFrame,
    stage: str,
    count_col: str,
    share_col: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if summary.empty:
        return rows

    for topic in summary.itertuples(index=False):
        topic_id = int(topic.topic_id)
        topic_label = str(topic.topic_label)
        rows.append(_stats_row("topic", stage, topic_id, topic_label, count_col, getattr(topic, count_col)))
        rows.append(_stats_row("topic", stage, topic_id, topic_label, share_col, getattr(topic, share_col)))
    return rows
