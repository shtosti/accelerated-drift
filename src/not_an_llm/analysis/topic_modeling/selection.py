from __future__ import annotations

from dataclasses import dataclass
import logging

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TopicSelectionResult:
    enriched: pd.DataFrame
    topic_labels: dict[int, str]
    embeddings_2d: pd.DataFrame | None
    summary: pd.DataFrame


def filter_topics(
    enriched: pd.DataFrame,
    topic_labels: dict[int, str],
    embeddings_2d: pd.DataFrame | None,
    *,
    min_share: float,
    min_count: int,
) -> TopicSelectionResult:
    if "topic_id" not in enriched.columns:
        return TopicSelectionResult(enriched, topic_labels, embeddings_2d, pd.DataFrame())

    topic_counts = enriched["topic_id"].value_counts().rename_axis("topic_id").reset_index(name="abstract_count")
    total = len(enriched)
    topic_counts["abstract_share"] = topic_counts["abstract_count"] / total if total else 0.0
    topic_counts["topic_label"] = topic_counts["topic_id"].map(topic_labels)
    topic_counts["passes_min_share"] = topic_counts["abstract_share"] >= min_share
    topic_counts["passes_min_count"] = topic_counts["abstract_count"] >= min_count
    topic_counts["selected"] = topic_counts["passes_min_share"] & topic_counts["passes_min_count"]

    selected_topics = set(topic_counts.loc[topic_counts["selected"], "topic_id"].astype(int))
    logger.info(
        "Selected %d/%d topics with min_share=%.4f and min_count=%d",
        len(selected_topics),
        len(topic_counts),
        min_share,
        min_count,
    )

    selected = enriched[enriched["topic_id"].isin(selected_topics)].copy()
    selected_labels = {int(topic_id): label for topic_id, label in topic_labels.items() if int(topic_id) in selected_topics}

    if embeddings_2d is not None:
        embeddings_2d = embeddings_2d[embeddings_2d["topic_id"].isin(selected_topics)].copy()

    return TopicSelectionResult(selected, selected_labels, embeddings_2d, topic_counts)
