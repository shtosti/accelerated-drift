from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(slots=True)
class CollectionConfig:
    source: str
    queries: list[str]
    arxiv_collection_mode: str
    samples_per_month: int
    year_min: int
    year_max: int
    page_size: int
    output_jsonl: Path
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
    include_readability: bool
    preprocessed_jsonl: Path
    feature_dataset_jsonl: Path
    trends_csv: Path
    trends_plot_dir: Path


@dataclass(slots=True)
class AppConfig:
    project_name: str
    data_dir: Path
    collection: CollectionConfig
    analysis: AnalysisConfig



def load_config(config_path: str | Path = "config.toml") -> AppConfig:
    path = Path(config_path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    project = _require_dict(raw, "project")
    collection = _require_dict(raw, "collection")
    analysis = _require_dict(raw, "analysis")
    source = _load_collection_source(collection)
    arxiv_collection_mode = _load_arxiv_collection_mode(collection)
    queries = _load_collection_queries(collection, source)
    output_jsonl = _load_output_jsonl(collection, source, arxiv_collection_mode)
    default_preprocessed, default_feature_dataset, default_trends_csv, default_trends_plot_dir = (
        _default_analysis_paths(output_jsonl)
    )

    return AppConfig(
        project_name=str(project["name"]),
        data_dir=Path(project["data_dir"]),
        collection=CollectionConfig(
            source=source,
            queries=queries,
            arxiv_collection_mode=arxiv_collection_mode,
            samples_per_month=int(collection.get("samples_per_month", 5)),
            year_min=int(collection["year_min"]),
            year_max=int(collection["year_max"]),
            page_size=int(collection["page_size"]),
            output_jsonl=output_jsonl,
            fields=[str(item) for item in collection["fields"]],
            min_request_interval_seconds=float(collection.get("min_request_interval_seconds", 1.0)),
            max_retries=int(collection.get("max_retries", 5)),
            initial_backoff_seconds=float(collection.get("initial_backoff_seconds", 1.0)),
            max_backoff_seconds=float(collection.get("max_backoff_seconds", 30.0)),
            backoff_jitter_seconds=float(collection.get("backoff_jitter_seconds", 0.25)),
        ),
        analysis=AnalysisConfig(
            enabled=bool(analysis["enabled"]),
            features=[str(item) for item in analysis["features"]],
            include_readability=bool(analysis.get("include_readability", True)),
            preprocessed_jsonl=_load_optional_path(analysis.get("preprocessed_jsonl"), default_preprocessed),
            feature_dataset_jsonl=_load_optional_path(analysis.get("feature_dataset_jsonl"), default_feature_dataset),
            trends_csv=_load_optional_path(analysis.get("trends_csv"), default_trends_csv),
            trends_plot_dir=_load_optional_path(analysis.get("trends_plot_dir"), default_trends_plot_dir),
        ),
    )



def _require_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid section: {key}")
    return value


def _load_collection_queries(collection: dict[str, Any], source: str) -> list[str]:
    if source == "arxiv":
        queries = _load_query_list(collection.get("arxiv_queries"))
        if queries:
            return queries

    if source == "semantic_scholar":
        queries = _load_query_list(collection.get("semantic_scholar_queries"))
        if queries:
            return queries

    # Backward compatible fallbacks.
    generic_queries = _load_query_list(collection.get("queries"))
    if generic_queries:
        return generic_queries

    raw_query = collection.get("query")
    if isinstance(raw_query, str) and raw_query.strip():
        return [raw_query.strip()]

    raise ValueError(
        "collection must define source-specific queries (arxiv_queries or semantic_scholar_queries), "
        "or fallback queries/query"
    )


def _load_query_list(raw_queries: Any) -> list[str]:
    if not isinstance(raw_queries, list):
        return []

    queries = [str(item).strip() for item in raw_queries if str(item).strip()]
    return queries


def _load_collection_source(collection: dict[str, Any]) -> str:
    raw_source = str(collection.get("source", "semantic_scholar")).strip().lower()
    allowed_sources = {"semantic_scholar", "arxiv"}
    if raw_source not in allowed_sources:
        raise ValueError(
            "collection.source must be one of: semantic_scholar, arxiv"
        )
    return raw_source


def _load_arxiv_collection_mode(collection: dict[str, Any]) -> str:
    raw_mode = str(collection.get("arxiv_collection_mode", "full")).strip().lower()
    allowed_modes = {"full", "monthly"}
    if raw_mode not in allowed_modes:
        raise ValueError("collection.arxiv_collection_mode must be one of: full, monthly")
    return raw_mode


def _load_output_jsonl(collection: dict[str, Any], source: str, arxiv_collection_mode: str) -> Path:
    if source == "arxiv":
        if arxiv_collection_mode == "monthly":
            source_specific_path = collection.get("arxiv_monthly_output_jsonl")
            if isinstance(source_specific_path, str) and source_specific_path.strip():
                return Path(source_specific_path.strip())

            source_specific_path = collection.get("arxiv_output_jsonl")
            if isinstance(source_specific_path, str) and source_specific_path.strip():
                return Path(source_specific_path.strip())
        else:
            source_specific_path = collection.get("arxiv_output_jsonl")
            if isinstance(source_specific_path, str) and source_specific_path.strip():
                return Path(source_specific_path.strip())

            source_specific_path = collection.get("arxiv_monthly_output_jsonl")
            if isinstance(source_specific_path, str) and source_specific_path.strip():
                return Path(source_specific_path.strip())

    if source == "semantic_scholar":
        source_specific_path = collection.get("semantic_scholar_output_jsonl")
        if isinstance(source_specific_path, str) and source_specific_path.strip():
            return Path(source_specific_path.strip())

    fallback_path = collection.get("output_jsonl")
    if isinstance(fallback_path, str) and fallback_path.strip():
        return Path(fallback_path.strip())

    raise ValueError(
        "collection must define source-specific output path "
        "('arxiv_monthly_output_jsonl'/'arxiv_output_jsonl' for arxiv, "
        "'semantic_scholar_output_jsonl' for semantic_scholar) or fallback 'output_jsonl'"
    )


def _load_optional_path(raw_value: Any, default: Path) -> Path:
    if isinstance(raw_value, str) and raw_value.strip():
        return Path(raw_value.strip())
    return default


def _default_analysis_paths(raw_output_path: Path) -> tuple[Path, Path, Path, Path]:
    suffix = raw_output_path.suffix or ".jsonl"
    preprocessed = Path("data/processed") / raw_output_path.name
    feature_dataset = Path("data/processed") / f"{raw_output_path.stem}_features{suffix}"
    trends_csv = Path("data/analysis") / f"{raw_output_path.stem}_feature_trends_by_year.csv"
    trends_plot_dir = Path("data/analysis/plots") / raw_output_path.stem
    return preprocessed, feature_dataset, trends_csv, trends_plot_dir
