from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import logging
import math
import re
from typing import Iterable

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TopicSelectionResult:
    enriched: pd.DataFrame
    topic_labels: dict[int, str]
    embeddings_2d: pd.DataFrame | None
    summary: pd.DataFrame
    initial_summary: pd.DataFrame
    merge_plan: pd.DataFrame


def filter_topics(
    enriched: pd.DataFrame,
    topic_labels: dict[int, str],
    embeddings_2d: pd.DataFrame | None,
    *,
    min_share: float,
    min_count: int,
    merge_candidates: pd.DataFrame | None = None,
    merge_under_threshold: bool = True,
    max_final_topics: int = 0,
    final_label_top_terms: int = 8,
) -> TopicSelectionResult:
    if "topic_id" not in enriched.columns:
        empty = pd.DataFrame()
        return TopicSelectionResult(enriched, topic_labels, embeddings_2d, empty, empty, empty)

    initial_summary = _topic_summary(enriched, topic_labels, min_share, min_count)
    if merge_under_threshold or max_final_topics > 0:
        final_topic_map, final_topic_labels, merge_plan = _build_merge_plan(
            initial_summary=initial_summary,
            topic_labels=topic_labels,
            merge_candidates=merge_candidates,
            min_share=min_share,
            min_count=min_count,
            max_final_topics=max_final_topics,
            merge_under_threshold=merge_under_threshold,
        )
        selected = _apply_merge_plan(enriched, final_topic_map, final_topic_labels)
        final_topic_labels = _recompute_final_topic_labels(
            selected,
            final_topic_labels,
            top_terms=final_label_top_terms,
        )
        selected["topic_label"] = selected["topic_id"].map(final_topic_labels)
    else:
        merge_plan = pd.DataFrame()
        selected_topics = set(initial_summary.loc[initial_summary["selected"], "topic_id"].astype(int))
        selected = enriched[enriched["topic_id"].isin(selected_topics)].copy()
        final_topic_map = {int(topic_id): int(topic_id) for topic_id in selected_topics}
        final_topic_labels = {
            int(topic_id): label
            for topic_id, label in topic_labels.items()
            if int(topic_id) in selected_topics
        }
        final_topic_labels = _recompute_final_topic_labels(
            selected,
            final_topic_labels,
            top_terms=final_label_top_terms,
        )
        selected["topic_label"] = selected["topic_id"].map(final_topic_labels)

    final_summary = _topic_summary(selected, final_topic_labels, min_share, min_count)

    if embeddings_2d is not None:
        embeddings_2d = _apply_embedding_merge_plan(embeddings_2d, final_topic_map, final_topic_labels)

    logger.info(
        "Topic selection finished with %d final topics from %d initial topics",
        final_summary["topic_id"].nunique() if "topic_id" in final_summary.columns else 0,
        initial_summary["topic_id"].nunique() if "topic_id" in initial_summary.columns else 0,
    )

    return TopicSelectionResult(
        selected,
        final_topic_labels,
        embeddings_2d,
        final_summary,
        initial_summary,
        merge_plan,
    )


def _topic_summary(
    enriched: pd.DataFrame,
    topic_labels: dict[int, str],
    min_share: float,
    min_count: int,
) -> pd.DataFrame:
    topic_counts = enriched["topic_id"].value_counts().rename_axis("topic_id").reset_index(name="abstract_count")
    total = len(enriched)
    topic_counts["abstract_share"] = topic_counts["abstract_count"] / total if total else 0.0
    topic_counts["topic_label"] = topic_counts["topic_id"].map(topic_labels)
    topic_counts["passes_min_share"] = topic_counts["abstract_share"] >= min_share
    topic_counts["passes_min_count"] = topic_counts["abstract_count"] >= min_count
    topic_counts["selected"] = topic_counts["passes_min_share"] & topic_counts["passes_min_count"]
    return topic_counts.sort_values(["abstract_count", "topic_id"], ascending=[False, True]).reset_index(drop=True)


def _build_merge_plan(
    initial_summary: pd.DataFrame,
    topic_labels: dict[int, str],
    merge_candidates: pd.DataFrame | None,
    min_share: float,
    min_count: int,
    max_final_topics: int,
    merge_under_threshold: bool,
) -> tuple[dict[int, int], dict[int, str], pd.DataFrame]:
    total = int(initial_summary["abstract_count"].sum())
    counts = {
        int(row.topic_id): int(row.abstract_count)
        for row in initial_summary.itertuples(index=False)
    }
    active: dict[int, set[int]] = {topic_id: {topic_id} for topic_id in counts}
    events: list[dict[str, object]] = []

    hierarchy = _prepare_hierarchy(merge_candidates, counts)

    if merge_under_threshold:
        _merge_by_hierarchy(
            active=active,
            counts=counts,
            total=total,
            hierarchy=hierarchy,
            events=events,
            min_share=min_share,
            min_count=min_count,
            reason="under_threshold_hierarchy",
            only_if_under_threshold=True,
        )
        _merge_by_size_fallback(
            active=active,
            counts=counts,
            total=total,
            events=events,
            min_share=min_share,
            min_count=min_count,
            max_final_topics=0,
            reason="under_threshold_size_fallback",
            only_if_under_threshold=True,
        )

    if max_final_topics > 0:
        _merge_by_hierarchy(
            active=active,
            counts=counts,
            total=total,
            hierarchy=hierarchy,
            events=events,
            min_share=min_share,
            min_count=min_count,
            reason="max_final_topics_hierarchy",
            only_if_under_threshold=False,
            max_final_topics=max_final_topics,
        )
        _merge_by_size_fallback(
            active=active,
            counts=counts,
            total=total,
            events=events,
            min_share=min_share,
            min_count=min_count,
            max_final_topics=max_final_topics,
            reason="max_final_topics_size_fallback",
            only_if_under_threshold=False,
        )

    final_clusters = sorted(
        active.values(),
        key=lambda members: (-_cluster_count(members, counts), min(members)),
    )
    final_topic_map: dict[int, int] = {}
    final_topic_labels: dict[int, str] = {}
    for final_topic_id, members in enumerate(final_clusters):
        for source_topic_id in members:
            final_topic_map[int(source_topic_id)] = final_topic_id
        final_topic_labels[final_topic_id] = _merged_label(members, counts, topic_labels)

    missing_topics = sorted(set(counts) - set(final_topic_map))
    if missing_topics:
        raise RuntimeError(f"Topic merge planning lost source topics: {missing_topics}")

    merge_plan = _merge_events_to_frame(events, final_topic_map, counts, total)
    return final_topic_map, final_topic_labels, merge_plan


def _prepare_hierarchy(
    merge_candidates: pd.DataFrame | None,
    counts: dict[int, int],
) -> list[dict[str, object]]:
    if merge_candidates is None or merge_candidates.empty:
        return []

    rows = merge_candidates.copy()
    if "Distance" in rows.columns:
        rows = rows.sort_values("Distance", ascending=True)

    node_members: dict[int, set[int]] = {topic_id: {topic_id} for topic_id in counts}
    pending = rows.to_dict("records")
    changed = True
    while changed:
        changed = False
        remaining = []
        for row in pending:
            parent = _as_int(row.get("Parent_ID"))
            left = _as_int(row.get("Child_Left_ID"))
            right = _as_int(row.get("Child_Right_ID"))
            if parent is None or left is None or right is None:
                continue
            if left in node_members and right in node_members:
                node_members[parent] = set(node_members[left]) | set(node_members[right])
                changed = True
            else:
                remaining.append(row)
        pending = remaining

    hierarchy: list[dict[str, object]] = []
    for row in rows.to_dict("records"):
        parent = _as_int(row.get("Parent_ID"))
        if parent is None or parent not in node_members:
            continue
        hierarchy.append(
            {
                "parent_id": parent,
                "members": node_members[parent],
                "distance": row.get("Distance"),
            }
        )
    return hierarchy


def _merge_by_hierarchy(
    *,
    active: dict[int, set[int]],
    counts: dict[int, int],
    total: int,
    hierarchy: list[dict[str, object]],
    events: list[dict[str, object]],
    min_share: float,
    min_count: int,
    reason: str,
    only_if_under_threshold: bool,
    max_final_topics: int = 0,
) -> None:
    if not hierarchy:
        return

    made_progress = True
    while made_progress:
        made_progress = False
        if max_final_topics > 0 and len(active) <= max_final_topics:
            return
        for row in hierarchy:
            parent_members = set(row["members"])
            cluster_ids = [
                cluster_id
                for cluster_id, members in active.items()
                if members and members.issubset(parent_members)
            ]
            if len(cluster_ids) < 2:
                continue
            if only_if_under_threshold and not any(
                _cluster_is_under_threshold(active[cluster_id], counts, total, min_share, min_count)
                for cluster_id in cluster_ids
            ):
                continue

            new_cluster_id = _next_cluster_id(active)
            merged_members = set().union(*(active[cluster_id] for cluster_id in cluster_ids))
            _record_merge_event(
                events,
                reason,
                cluster_ids,
                merged_members,
                counts,
                total,
                row.get("distance"),
                hierarchy_parent_id=row.get("parent_id"),
            )
            for cluster_id in cluster_ids:
                active.pop(cluster_id, None)
            active[new_cluster_id] = merged_members
            made_progress = True
            break


def _merge_by_size_fallback(
    *,
    active: dict[int, set[int]],
    counts: dict[int, int],
    total: int,
    events: list[dict[str, object]],
    min_share: float,
    min_count: int,
    max_final_topics: int,
    reason: str,
    only_if_under_threshold: bool,
) -> None:
    next_cluster_id = max(active.keys(), default=0) + 1
    while len(active) > 1:
        if max_final_topics > 0 and len(active) <= max_final_topics:
            return

        ordered = sorted(active, key=lambda cluster_id: _cluster_count(active[cluster_id], counts))
        source_id = next(
            (
                cluster_id
                for cluster_id in ordered
                if not only_if_under_threshold
                or _cluster_is_under_threshold(active[cluster_id], counts, total, min_share, min_count)
            ),
            None,
        )
        if source_id is None:
            return

        target_id = max(
            (cluster_id for cluster_id in active if cluster_id != source_id),
            key=lambda cluster_id: _cluster_count(active[cluster_id], counts),
        )
        merged_members = set(active[source_id]) | set(active[target_id])
        _record_merge_event(events, reason, [source_id, target_id], merged_members, counts, total, None)
        active.pop(source_id, None)
        active.pop(target_id, None)
        active[next_cluster_id] = merged_members
        next_cluster_id += 1


def _record_merge_event(
    events: list[dict[str, object]],
    reason: str,
    cluster_ids: Iterable[int],
    merged_members: set[int],
    counts: dict[int, int],
    total: int,
    distance: object,
    hierarchy_parent_id: object | None = None,
) -> None:
    source_topic_ids = sorted(merged_members)
    merged_count = _cluster_count(merged_members, counts)
    events.append(
        {
            "merge_step": len(events) + 1,
            "reason": reason,
            "source_cluster_ids": ";".join(str(cluster_id) for cluster_id in cluster_ids),
            "source_topic_ids": ";".join(str(topic_id) for topic_id in source_topic_ids),
            "merged_abstract_count": merged_count,
            "merged_abstract_share": merged_count / total if total else 0.0,
            "hierarchy_distance": distance,
            "hierarchy_parent_id": hierarchy_parent_id,
        }
    )


def _next_cluster_id(active: dict[int, set[int]]) -> int:
    return max(active.keys(), default=-1) + 1


def _merge_events_to_frame(
    events: list[dict[str, object]],
    final_topic_map: dict[int, int],
    counts: dict[int, int],
    total: int,
) -> pd.DataFrame:
    rows = []
    for event in events:
        source_topic_ids = [int(topic_id) for topic_id in str(event["source_topic_ids"]).split(";") if topic_id]
        final_topic_ids = sorted({final_topic_map[topic_id] for topic_id in source_topic_ids})
        row = dict(event)
        row["final_topic_ids"] = ";".join(str(topic_id) for topic_id in final_topic_ids)
        row["source_topic_count"] = len(source_topic_ids)
        row["source_abstract_counts"] = ";".join(str(counts[topic_id]) for topic_id in source_topic_ids)
        row["source_abstract_shares"] = ";".join(
            f"{counts[topic_id] / total:.6f}" if total else "0.000000"
            for topic_id in source_topic_ids
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_merge_plan(
    enriched: pd.DataFrame,
    final_topic_map: dict[int, int],
    final_topic_labels: dict[int, str],
) -> pd.DataFrame:
    selected = enriched.copy()
    selected["original_topic_id"] = selected["topic_id"]
    selected["original_topic_label"] = selected["topic_label"]
    selected["topic_id"] = selected["original_topic_id"].map(lambda topic_id: final_topic_map[int(topic_id)])
    selected["topic_label"] = selected["topic_id"].map(final_topic_labels)
    return selected


def _apply_embedding_merge_plan(
    embeddings_2d: pd.DataFrame,
    final_topic_map: dict[int, int],
    final_topic_labels: dict[int, str],
) -> pd.DataFrame:
    merged = embeddings_2d.copy()
    merged["original_topic_id"] = merged["topic_id"]
    merged["original_topic_label"] = merged["topic_label"]
    merged["topic_id"] = merged["original_topic_id"].map(lambda topic_id: final_topic_map[int(topic_id)])
    merged["topic_label"] = merged["topic_id"].map(final_topic_labels)
    return merged


def _recompute_final_topic_labels(
    enriched: pd.DataFrame,
    fallback_labels: dict[int, str],
    *,
    top_terms: int = 8,
) -> dict[int, str]:
    text_column = _label_text_column(enriched)
    if text_column is None or enriched.empty:
        return fallback_labels

    docs_by_topic = (
        enriched.assign(_label_text=enriched[text_column].fillna("").astype(str))
        .groupby("topic_id")["_label_text"]
        .apply(lambda values: " ".join(value for value in values if value.strip()))
    )
    if docs_by_topic.empty:
        return fallback_labels

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
        )
        matrix = vectorizer.fit_transform(docs_by_topic.tolist())
    except ImportError:
        return {
            int(topic_id): _simple_top_terms(text, top_terms)
            or fallback_labels.get(int(topic_id), f"topic_{topic_id}")
            for topic_id, text in docs_by_topic.items()
        }
    except ValueError:
        return fallback_labels

    terms = vectorizer.get_feature_names_out()
    labels: dict[int, str] = {}
    for row_index, topic_id in enumerate(docs_by_topic.index):
        row = matrix.getrow(row_index)
        if row.nnz == 0:
            labels[int(topic_id)] = fallback_labels.get(int(topic_id), f"topic_{topic_id}")
            continue
        top_indices = row.toarray().ravel().argsort()[::-1][:top_terms]
        top_words = _dedupe_terms([terms[index] for index in top_indices if row[0, index] > 0], top_terms)
        labels[int(topic_id)] = ", ".join(top_words) or fallback_labels.get(int(topic_id), f"topic_{topic_id}")

    return labels


def _label_text_column(enriched: pd.DataFrame) -> str | None:
    for column in ("text_lemma", "text_clean", "text_raw", "abstract"):
        if column in enriched.columns:
            return column
    return None


def _simple_top_terms(text: str, top_terms: int) -> str:
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "in", "is", "it", "its", "of", "on", "or", "that", "the", "their",
        "this", "to", "was", "were", "with", "we",
    }
    tokens = [
        token
        for token in re.findall(r"[a-z][a-z0-9_]+", text.lower())
        if token not in stop_words and len(token) > 2
    ]
    if not tokens:
        return ""

    unigrams = Counter(tokens)
    bigrams = Counter(
        f"{left} {right}"
        for left, right in zip(tokens, tokens[1:])
        if left != right
    )
    combined = Counter()
    combined.update(unigrams)
    combined.update({term: count * 2 for term, count in bigrams.items()})

    return ", ".join(_dedupe_terms([term for term, _ in combined.most_common()], top_terms))


def _dedupe_terms(terms: list[str], top_terms: int) -> list[str]:
    selected: list[str] = []
    used_words: set[str] = set()
    for term in terms:
        words = set(term.split())
        if len(selected) >= top_terms:
            break
        if words & used_words:
            continue
        selected.append(term)
        used_words.update(words)
    return selected


def _cluster_count(members: set[int], counts: dict[int, int]) -> int:
    return sum(counts[topic_id] for topic_id in members)


def _cluster_is_under_threshold(
    members: set[int],
    counts: dict[int, int],
    total: int,
    min_share: float,
    min_count: int,
) -> bool:
    count = _cluster_count(members, counts)
    share = count / total if total else 0.0
    return count < min_count or share < min_share


def _merged_label(
    members: set[int],
    counts: dict[int, int],
    topic_labels: dict[int, str],
    max_labels: int = 3,
) -> str:
    ordered = sorted(members, key=lambda topic_id: (-counts[topic_id], topic_id))
    labels = [topic_labels.get(topic_id, f"topic_{topic_id}") for topic_id in ordered[:max_labels]]
    if len(ordered) > max_labels:
        labels.append("...")
    return " / ".join(labels)


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
