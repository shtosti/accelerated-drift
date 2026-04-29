from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


# =========================
# CONFIG MODELS
# =========================

@dataclass(slots=True)
class CollectionConfig:
    source: str
    queries: list[str]

    arxiv_collection_mode: str
    medarxiv_collection_mode: str
    biorxiv_collection_mode: str

    samples_per_month: int
    year_min: int
    year_max: int
    page_size: int

    # FIXED: explicit outputs (no dict)
    arxiv_output_jsonl: Path
    arxiv_monthly_output_jsonl: Path
    semantic_scholar_output_jsonl: Path
    medarxiv_output_jsonl: Path
    biorxiv_output_jsonl: Path

    fields: list[str]

    min_request_interval_seconds: float
    max_retries: int
    initial_backoff_seconds: float
    max_backoff_seconds: float
    backoff_jitter_seconds: float


@dataclass(slots=True)
class AnalysisConfig:
    enabled: bool
    features: list[str]
    spacy_features: list[str]
    include_readability: bool
    readability_metrics: list[str]
    syntactic_features: dict[str, str]

    llm_marker_words: list[str]
    llm_marker_verbs: list[str]
    llm_marker_adjectives: list[str]
    llm_marker_phrases: list[str]

    sequential_markers: list[str]
    causal_markers: list[str]
    contrast_markers: list[str]
    emphasis_markers: list[str]
    summary_markers: list[str]

    llm_marker_sentence_patterns: dict[str, str]

    enable_list_of_three_marker: bool
    list_of_three_pattern: str
    llm_marker_word_matching: str

    hedge_terms: list[str]
    certainty_terms: list[str]

    preprocessed_jsonl: Path
    feature_dataset_jsonl: Path
    trends_csv: Path
    monthly_trends_csv: Path
    trends_plot_dir: Path


@dataclass(slots=True)
class ExternalConfig:
    dataset: str
    hc3_subset: str
    hc3_subsets: list[str]
    hc3_split: str
    hc3_pair_output_jsonl: Path
    hc3_feature_output_jsonl: Path
    hc3_comparison_csv: Path
    hc3_comparison_plot: Path
    mage_split: str
    mage_max_pairs: int
    mage_domains: list[str]
    mage_include_sources: list[str]
    mage_exclude_sources: list[str]
    mage_pair_output_jsonl: Path
    mage_feature_output_jsonl: Path
    mage_comparison_csv: Path
    mage_comparison_plot: Path


@dataclass(slots=True)
class AppConfig:
    project_name: str
    data_dir: Path
    collection: CollectionConfig
    analysis: AnalysisConfig
    external: ExternalConfig | None


# =========================
# LOADER
# =========================

def load_config(config_path: str | Path = "config.toml") -> AppConfig:
    path = Path(config_path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    project = _require_dict(raw, "project")
    collection = _require_dict(raw, "collection")
    analysis = _require_dict(raw, "analysis")
    external = raw.get("external") if isinstance(raw.get("external"), dict) else None

    source = _load_collection_source(collection)

    collection_config = CollectionConfig(
        source=source,
        queries=_load_collection_queries(collection, source),

        arxiv_collection_mode=_load_arxiv_collection_mode(collection),
        medarxiv_collection_mode=_load_medarxiv_collection_mode(collection),
        biorxiv_collection_mode=_load_biorxiv_collection_mode(collection),

        samples_per_month=int(collection.get("samples_per_month", 5)),
        year_min=int(collection["year_min"]),
        year_max=int(collection["year_max"]),
        page_size=int(collection["page_size"]),

        arxiv_output_jsonl=Path(collection["arxiv_output_jsonl"]),
        arxiv_monthly_output_jsonl=Path(collection["arxiv_monthly_output_jsonl"]),
        semantic_scholar_output_jsonl=Path(collection["semantic_scholar_output_jsonl"]),
        medarxiv_output_jsonl=Path(collection["medarxiv_output_jsonl"]),
        biorxiv_output_jsonl=_load_optional_collection_path(
            collection,
            "biorxiv_output_jsonl",
            "bioarxiv_output_jsonl",
        ),

        fields=[str(x) for x in collection["fields"]],

        min_request_interval_seconds=float(collection.get("min_request_interval_seconds", 1.0)),
        max_retries=int(collection.get("max_retries", 5)),
        initial_backoff_seconds=float(collection.get("initial_backoff_seconds", 1.0)),
        max_backoff_seconds=float(collection.get("max_backoff_seconds", 30.0)),
        backoff_jitter_seconds=float(collection.get("backoff_jitter_seconds", 0.25)),
    )

    (
        default_preprocessed,
        default_feature_dataset,
        default_trends_csv,
        default_monthly_trends_csv,
        default_trends_plot_dir,
    ) = _default_analysis_paths(Path("data/raw/" + source + ".jsonl"))

    return AppConfig(
        project_name=str(project["name"]),
        data_dir=Path(project["data_dir"]),
        collection=collection_config,

        analysis=AnalysisConfig(
            enabled=bool(analysis["enabled"]),
            features=[str(x) for x in analysis["features"]],
            spacy_features=_load_query_list(analysis.get("spacy_features", [])),
            include_readability=bool(analysis.get("include_readability", True)),
            readability_metrics=_load_readability_metrics(analysis),
            syntactic_features=_load_syntactic_features(analysis),

            llm_marker_words=_load_query_list(analysis.get("markers", [])),
            llm_marker_verbs=_load_query_list(analysis.get("verbs", [])),
            llm_marker_adjectives=_load_query_list(analysis.get("adjectives", [])),
            llm_marker_phrases=_load_query_list(analysis.get("phrases", [])),

            sequential_markers=_load_query_list(analysis.get("sequential_markers", [])),
            causal_markers=_load_query_list(analysis.get("causal_markers", [])),
            contrast_markers=_load_query_list(analysis.get("contrast_markers", [])),
            emphasis_markers=_load_query_list(analysis.get("emphasis_markers", [])),
            summary_markers=_load_query_list(analysis.get("summary_markers", [])),

            llm_marker_sentence_patterns={},

            enable_list_of_three_marker=True,
            list_of_three_pattern=r"",
            llm_marker_word_matching="lemma",

            hedge_terms=_load_query_list(analysis.get("hedge_terms", [])),
            certainty_terms=_load_query_list(analysis.get("certainty_terms", [])),

            preprocessed_jsonl=default_preprocessed,
            feature_dataset_jsonl=default_feature_dataset,
            trends_csv=default_trends_csv,
            monthly_trends_csv=default_monthly_trends_csv,
            trends_plot_dir=default_trends_plot_dir,
        ),

        external=None,
    )


# =========================
# HELPERS (unchanged)
# =========================

def _require_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in raw or not isinstance(raw[key], dict):
        raise ValueError(f"Missing or invalid section: {key}")
    return raw[key]


def _load_query_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(x).strip() for x in raw if str(x).strip()]


def _load_collection_source(collection: dict[str, Any]) -> str:
    return str(collection.get("source", "semantic_scholar")).lower()


def _load_optional_collection_path(collection: dict[str, Any], *keys: str) -> Path:
    for key in keys:
        value = collection.get(key)
        if value is not None:
            return Path(value)
    raise KeyError(keys[0])


def _load_arxiv_collection_mode(c: dict[str, Any]) -> str:
    return str(c.get("arxiv_collection_mode", "full")).lower()


def _load_medarxiv_collection_mode(c: dict[str, Any]) -> str:
    return str(c.get("medarxiv_collection_mode", "full")).lower()


def _load_biorxiv_collection_mode(c: dict[str, Any]) -> str:
    return str(c.get("biorxiv_collection_mode", "full")).lower()


def _load_collection_queries(collection: dict[str, Any], source: str) -> list[str]:
    key = f"{source}_queries"
    return _load_query_list(collection.get(key, ["*"]))


def _load_syntactic_features(analysis: dict[str, Any]) -> dict[str, str]:
    raw = analysis.get("syntactic_features", {})
    if not isinstance(raw, dict):
        return {"em_dash": "--", "semicolon": ";"}
    return {str(k): str(v) for k, v in raw.items()}


def _load_readability_metrics(analysis: dict[str, Any]) -> list[str]:
    return _load_query_list(analysis.get("readability_metrics", []))


def _default_analysis_paths(path: Path):
    return (
        Path("data/processed/" + path.name),
        Path("data/analyzed/" + path.name),
        Path("data/analysis/" + path.stem + "_year.csv"),
        Path("data/analysis/" + path.stem + "_month.csv"),
        Path("data/analysis/plots/" + path.stem),
    )