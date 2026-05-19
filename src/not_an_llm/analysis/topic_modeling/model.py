from __future__ import annotations

from dataclasses import dataclass
import logging
import multiprocessing
from pathlib import Path
import time
import traceback
from typing import Any

import pandas as pd

from not_an_llm.config import AppConfig


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TopicModelingResult:
    enriched: pd.DataFrame
    topic_labels: dict[int, str]
    embeddings_2d: pd.DataFrame | None
    merge_candidates: pd.DataFrame | None


def assign_topics(enriched: pd.DataFrame, config: AppConfig) -> TopicModelingResult:
    if not config.analysis.topic_modeling_enabled:
        return TopicModelingResult(enriched, {}, None, None)

    if "text_clean" not in enriched.columns:
        raise ValueError("Topic modeling requires a 'text_clean' column.")

    texts = enriched["text_clean"].fillna("").astype(str).tolist()
    top_terms = config.analysis.topic_modeling_top_n_terms

    if not config.analysis.topic_modeling_use_bertopic:
        logger.warning("Only BERTopic topic modeling is implemented; using fallback topic assignment.")
        return _fallback_topics(enriched, texts)

    try:
        return _assign_bertopic_topics(enriched, config, texts, top_terms)
    except Exception as exc:
        logger.error("Topic modeling failed: %s", exc)
        logger.error("Full traceback:\n%s", traceback.format_exc())
        logger.warning("Using fallback topic assignment")
        return _fallback_topics(enriched, texts)


def _assign_bertopic_topics(
    enriched: pd.DataFrame,
    config: AppConfig,
    texts: list[str],
    top_terms: int,
) -> TopicModelingResult:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    import hdbscan
    import torch
    import umap

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = config.analysis.topic_modeling_num_workers or multiprocessing.cpu_count()
    umap_n_jobs = min(num_workers, 16)

    min_cluster_size = max(10, int(len(texts) * config.analysis.topic_modeling_min_cluster_ratio))
    min_samples = max(5, min_cluster_size // 2)

    logger.info(
        "Using device: %s for topic modeling (%d documents, %d requested topics)",
        device,
        len(texts),
    )
    logger.info("Using %d threads for UMAP, %d workers for HDBSCAN", umap_n_jobs, num_workers)

    start_time = time.time()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    logger.info("Encoding document embeddings...")
    embeddings = embedding_model.encode(
        texts,
        batch_size=config.analysis.topic_modeling_embedding_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    vectorizer_model = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
    )
    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
        n_jobs=umap_n_jobs,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=16,
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="english",
        nr_topics=config.analysis.topic_modeling_max_final_topics,
        top_n_words=top_terms,
        min_topic_size=min_cluster_size,
        calculate_probabilities=False,
        verbose=False,
    )

    topic_model.fit_transform(texts, embeddings=embeddings)
    raw_topics = topic_model.topics_
    logger.info("Topic modeling completed in %.2f seconds", time.time() - start_time)

    topic_id_map = _build_topic_id_map(raw_topics)
    labels = [topic_id_map.get(int(topic), int(topic)) for topic in raw_topics]
    topic_labels = _extract_topic_labels(topic_model, vectorizer_model, topic_id_map, top_terms)

    actual_topics = sorted(topic_labels)
    logger.info("Created %d topics", len(actual_topics))
 
    enriched = enriched.copy()
    enriched["topic_id"] = labels
    enriched["topic_label"] = [topic_labels.get(int(topic), "noise") for topic in labels]

    embeddings_2d = _extract_2d_embeddings(topic_model, labels, topic_labels)
    merge_candidates = _extract_hierarchical_merge_candidates(topic_model, texts, topic_id_map, topic_labels)

    return TopicModelingResult(enriched, topic_labels, embeddings_2d, merge_candidates)


def _build_topic_id_map(raw_topics: list[int]) -> dict[int, int]:
    original_topics = sorted(set(int(topic) for topic in raw_topics))
    topic_id_map = {topic_id: topic_id for topic_id in original_topics if topic_id != -1}
    if -1 in original_topics:
        next_topic_id = max(topic_id_map.keys(), default=-1) + 1
        topic_id_map[-1] = next_topic_id
    return topic_id_map


def _extract_topic_labels(
    topic_model: Any,
    vectorizer_model: Any,
    topic_id_map: dict[int, int],
    top_terms: int,
) -> dict[int, str]:
    stop_words = set(vectorizer_model.get_stop_words() or [])
    topic_labels: dict[int, str] = {}

    for source_topic_id, target_topic_id in topic_id_map.items():
        raw_topic = topic_model.get_topic(source_topic_id) or []
        terms = [term for term, _ in raw_topic if term.lower() not in stop_words]
        if not terms:
            terms = [term for term, _ in raw_topic]
        topic_labels[target_topic_id] = ", ".join(terms[:top_terms]) or f"topic_{target_topic_id}"

    return topic_labels


def _extract_2d_embeddings(
    topic_model: Any,
    labels: list[int],
    topic_labels: dict[int, str],
) -> pd.DataFrame | None:
    try:
        if not hasattr(topic_model, "umap_model") or not hasattr(topic_model.umap_model, "embedding_"):
            return None
        embeddings_2d = pd.DataFrame(topic_model.umap_model.embedding_, columns=["x", "y"])
        embeddings_2d["topic_id"] = labels
        embeddings_2d["topic_label"] = [topic_labels.get(int(topic), "noise") for topic in labels]
        return embeddings_2d
    except Exception as exc:
        logger.warning("Could not extract 2D embeddings: %s", exc)
        return None


def _extract_hierarchical_merge_candidates(
    topic_model: Any,
    texts: list[str],
    topic_id_map: dict[int, int],
    topic_labels: dict[int, str],
) -> pd.DataFrame | None:
    try:
        hierarchy = topic_model.hierarchical_topics(texts)
    except Exception as exc:
        logger.warning("Could not compute hierarchical topic merge candidates: %s", exc)
        return None

    if hierarchy is None or hierarchy.empty:
        return None

    candidates = hierarchy.copy()
    for column in ("Parent_ID", "Child_Left_ID", "Child_Right_ID"):
        if column in candidates.columns:
            candidates[column] = candidates[column].map(lambda value: topic_id_map.get(value, value))

    label_columns = {
        "Child_Left_ID": "child_left_label",
        "Child_Right_ID": "child_right_label",
        "Parent_ID": "parent_label",
    }
    for source_column, label_column in label_columns.items():
        if source_column in candidates.columns:
            candidates[label_column] = candidates[source_column].map(topic_labels)

    return candidates


def _fallback_topics(enriched: pd.DataFrame, texts: list[str]) -> TopicModelingResult:
    fallback = enriched.copy()
    fallback["topic_id"] = [0] * len(texts)
    fallback["topic_label"] = "fallback_topic"
    return TopicModelingResult(fallback, {0: "fallback_topic"}, None, None)


def save_topic_labels(topic_labels: dict[int, str], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "topic_id": list(topic_labels.keys()),
            "topic_label": list(topic_labels.values()),
        }
    ).to_csv(output_path, index=False)
    return output_path


def save_merge_candidates(merge_candidates: pd.DataFrame | None, output_path: Path) -> Path | None:
    if merge_candidates is None or merge_candidates.empty:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merge_candidates.to_csv(output_path, index=False)
    return output_path
