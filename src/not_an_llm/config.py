from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib


@dataclass(slots=True)
class CollectionConfig:
    queries: list[str]
    year_min: int
    year_max: int
    max_results: int
    page_size: int
    output_jsonl: Path
    fields: list[str]
    min_request_interval_seconds: float
    max_retries: int
    initial_backoff_seconds: float
    max_backoff_seconds: float
    backoff_jitter_seconds: float


@dataclass(slots=True)
class ExperimentConfig:
    llm_introduction_year: int
    pre_window_years: int
    post_window_years: int


@dataclass(slots=True)
class AnalysisConfig:
    enabled: bool
    features: list[str]


@dataclass(slots=True)
class AppConfig:
    project_name: str
    data_dir: Path
    collection: CollectionConfig
    experiment: ExperimentConfig
    analysis: AnalysisConfig



def load_config(config_path: str | Path = "config.toml") -> AppConfig:
    path = Path(config_path)
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    project = _require_dict(raw, "project")
    collection = _require_dict(raw, "collection")
    experiment = _require_dict(raw, "experiment")
    analysis = _require_dict(raw, "analysis")
    queries = _load_collection_queries(collection)

    return AppConfig(
        project_name=str(project["name"]),
        data_dir=Path(project["data_dir"]),
        collection=CollectionConfig(
            queries=queries,
            year_min=int(collection["year_min"]),
            year_max=int(collection["year_max"]),
            max_results=int(collection["max_results"]),
            page_size=int(collection["page_size"]),
            output_jsonl=Path(collection["output_jsonl"]),
            fields=[str(item) for item in collection["fields"]],
            min_request_interval_seconds=float(collection.get("min_request_interval_seconds", 1.0)),
            max_retries=int(collection.get("max_retries", 5)),
            initial_backoff_seconds=float(collection.get("initial_backoff_seconds", 1.0)),
            max_backoff_seconds=float(collection.get("max_backoff_seconds", 30.0)),
            backoff_jitter_seconds=float(collection.get("backoff_jitter_seconds", 0.25)),
        ),
        experiment=ExperimentConfig(
            llm_introduction_year=int(experiment["llm_introduction_year"]),
            pre_window_years=int(experiment["pre_window_years"]),
            post_window_years=int(experiment["post_window_years"]),
        ),
        analysis=AnalysisConfig(
            enabled=bool(analysis["enabled"]),
            features=[str(item) for item in analysis["features"]],
        ),
    )



def _require_dict(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid section: {key}")
    return value


def _load_collection_queries(collection: dict[str, Any]) -> list[str]:
    raw_queries = collection.get("queries")
    if isinstance(raw_queries, list):
        queries = [str(item).strip() for item in raw_queries if str(item).strip()]
        if queries:
            return queries

    raw_query = collection.get("query")
    if isinstance(raw_query, str) and raw_query.strip():
        return [raw_query.strip()]

    raise ValueError("collection must define either a non-empty 'queries' list or 'query' string")
