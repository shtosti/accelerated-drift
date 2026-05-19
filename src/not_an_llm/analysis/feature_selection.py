from __future__ import annotations

from typing import Iterable

import pandas as pd

from not_an_llm.config import AppConfig


CANONICAL_BASE_FEATURES = {
    "clause_depth",
    "clause_depth_std",
    "dependency_entropy",
    "dependency_length",
    "dependency_length_std",
    "coordination_count",
    "coordination_per_sentence_std",
    "sentence_depth_std",
    "list_of_three",
    "list_of_three_per_1k_words",
    "avg_words_per_sentence",
    "avg_syllables_per_word",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "dale_chall",
    "smog_index",
    "automated_readability_index",
    "gunning_fog",
    "hedge_ratio",
    "certainty_ratio",
}

EXCLUDED_ANALYSIS_FEATURES = {
    "word_count",
    "sentence_count",
    "paper_count",
    "marker_density",
    "coordination_density",
    "clause_depth_per_sentence",
    "dependency_entropy_normalized",
    "dependency_length_norm",
    "sentence_depth_cv",
    "coordination_count_per_1k_words",
}


def resolve_feature_columns(config: AppConfig, *frames: pd.DataFrame) -> list[str]:
    """Resolve configured feature names against raw or aggregated dataframes."""

    available = _available_feature_columns(frames)
    requested = [item.strip() for item in config.analysis.features if item.strip()]

    if not requested or requested == ["all"]:
        return [column for column in available if is_canonical_analysis_feature(column)]

    return [
        column
        for column in requested
        if column in available and is_canonical_analysis_feature(column)
    ]


def is_canonical_analysis_feature(column_name: str) -> bool:
    if column_name in EXCLUDED_ANALYSIS_FEATURES:
        return False

    if column_name in CANONICAL_BASE_FEATURES:
        return True

    if column_name.endswith("_total_per_1k_words"):
        return True

    if column_name.endswith("_per_1k_words") and not column_name.endswith("_count_per_1k_words"):
        return True

    if column_name.startswith(("verb_", "adjective_", "word_", "phrase_")):
        return True

    if column_name.startswith(
        (
            "sequential_marker_",
            "causal_marker_",
            "contrast_marker_",
            "emphasis_marker_",
            "summary_marker_",
        )
    ):
        return True

    return False


def build_marker_group_specs(config: AppConfig) -> tuple[dict[str, dict[str, object]], set[str]]:
    group_specs: dict[str, dict[str, object]] = {}
    summary_features: set[str] = set()

    group_definitions = {
        "marker_words": (
            "Marker words",
            "word",
            config.analysis.llm_marker_words,
            "marker_words_total_per_1k_words",
            "marker_words_total",
        ),
        "marker_verbs": (
            "Marker verbs",
            "verb",
            config.analysis.llm_marker_verbs,
            "marker_verbs_total_per_1k_words",
            "marker_verbs_total",
        ),
        "marker_adjectives": (
            "Marker adjectives",
            "adjective",
            config.analysis.llm_marker_adjectives,
            "marker_adjectives_total_per_1k_words",
            "marker_adjectives_total",
        ),
        "marker_phrases": (
            "Marker phrases",
            "phrase",
            config.analysis.llm_marker_phrases,
            "marker_phrases_total_per_1k_words",
            "marker_phrases_total",
        ),
        "sequential_markers": (
            "Sequential markers",
            "sequential_marker",
            config.analysis.sequential_markers,
            "sequential_markers_total_per_1k_words",
            "sequential_markers_total",
        ),
        "causal_markers": (
            "Causal markers",
            "causal_marker",
            config.analysis.causal_markers,
            "causal_markers_total_per_1k_words",
            "causal_markers_total",
        ),
        "contrast_markers": (
            "Contrast markers",
            "contrast_marker",
            config.analysis.contrast_markers,
            "contrast_markers_total_per_1k_words",
            "contrast_markers_total",
        ),
        "emphasis_markers": (
            "Emphasis markers",
            "emphasis_marker",
            config.analysis.emphasis_markers,
            "emphasis_markers_total_per_1k_words",
            "emphasis_markers_total",
        ),
        "summary_markers": (
            "Summary markers",
            "summary_marker",
            config.analysis.summary_markers,
            "summary_markers_total_per_1k_words",
            "summary_markers_total",
        ),
    }

    for group_name, (label, _prefix, _terms, rate_feature, count_feature) in group_definitions.items():
        group_specs[group_name] = {
            "label": label,
            "rate_feature": rate_feature,
            "count_feature": count_feature,
        }
        summary_features.add(rate_feature)
        summary_features.add(count_feature)

    summary_features.add("marker_density")
    return group_specs, summary_features


def _available_feature_columns(frames: Iterable[pd.DataFrame]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        if frame is None or frame.empty:
            continue
        for column in frame.columns:
            feature = _feature_name_from_column(column)
            if feature is None or feature in seen:
                continue
            if _column_has_numeric_values(frame[column]):
                columns.append(feature)
                seen.add(feature)
    return columns


def _feature_name_from_column(column: str) -> str | None:
    for suffix in ("_yearly_mean", "_monthly_mean"):
        if column.endswith(suffix):
            return column.removesuffix(suffix)
    return column


def _column_has_numeric_values(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    return not pd.to_numeric(series, errors="coerce").dropna().empty
