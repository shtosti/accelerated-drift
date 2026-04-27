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
    medarxiv_collection_mode: str
    bioarxiv_collection_mode: str
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
    marker_config = analysis.get("markers") if isinstance(analysis.get("markers"), dict) else {}
    lexicon_config = analysis.get("lexicon") if isinstance(analysis.get("lexicon"), dict) else {}
    source = _load_collection_source(collection)
    arxiv_collection_mode = _load_arxiv_collection_mode(collection)
    medarxiv_collection_mode = _load_medarxiv_collection_mode(collection)
    bioarxiv_collection_mode = _load_bioarxiv_collection_mode(collection)
    queries = _load_collection_queries(collection, source)
    output_jsonl = _load_output_jsonl(collection, source, arxiv_collection_mode)
    (
        default_preprocessed,
        default_feature_dataset,
        default_trends_csv,
        default_monthly_trends_csv,
        default_trends_plot_dir,
    ) = (
        _default_analysis_paths(output_jsonl)
    )

    return AppConfig(
        project_name=str(project["name"]),
        data_dir=Path(project["data_dir"]),
        collection=CollectionConfig(
            source=source,
            queries=queries,
            arxiv_collection_mode=arxiv_collection_mode,
            medarxiv_collection_mode=medarxiv_collection_mode,
            bioarxiv_collection_mode=bioarxiv_collection_mode,
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
            spacy_features=_load_query_list(analysis.get("spacy_features", [])),
            include_readability=bool(analysis.get("include_readability", True)),
            readability_metrics=_load_readability_metrics(analysis),
            syntactic_features=_load_syntactic_features(analysis),
            llm_marker_words=_load_lexicon_terms(
                lexicon_config,
                marker_config,
                analysis,
                "markers",
                ["unparalleled", "invaluable", "delve"],
                "llm_marker_words",
            ),
            llm_marker_verbs=_load_lexicon_terms(
                lexicon_config,
                marker_config,
                analysis,
                "verbs",
                ["delve", "underscore", "showcase", "enhance", "exhibit", "garner", "align"],
                "llm_marker_verbs",
            ),
            llm_marker_adjectives=_load_lexicon_terms(
                lexicon_config,
                marker_config,
                analysis,
                "adjectives",
                ["crucial", "pivotal", "comprehensive", "intricate", "potential"],
                "llm_marker_adjectives",
            ),
            llm_marker_phrases=_load_lexicon_terms(
                lexicon_config,
                marker_config,
                analysis,
                "phrases",
                ["meticulously delve", "intricate web", "comprehensive chapter", "deep dive", "intricate interplay", "essential insight"],
                "llm_marker_phrases",
            ),
            sequential_markers=_load_lexicon_terms(
                lexicon_config,
                marker_config,
                analysis,
                "sequential_markers",
                ["additionally", "furthermore", "moreover", "subsequently", "further"],
                "sequential_markers",
            ),
            causal_markers=_load_lexicon_terms(
                lexicon_config,
                marker_config,
                analysis,
                "causal_markers",
                ["hence", "thus", "consequently", "accordingly", "thereby"],
                "causal_markers",
            ),
            contrast_markers=_load_lexicon_terms(
                lexicon_config,
                marker_config,
                analysis,
                "contrast_markers",
                ["however", "nonetheless", "nevertheless", "conversely", "alternatively"],
                "contrast_markers",
            ),
            emphasis_markers=_load_lexicon_terms(
                lexicon_config,
                marker_config,
                analysis,
                "emphasis_markers",
                ["notably", "crucially", "remarkably", "particularly", "importantly"],
                "emphasis_markers",
            ),
            summary_markers=_load_lexicon_terms(
                lexicon_config,
                marker_config,
                analysis,
                "summary_markers",
                ["overall", "collectively", "ultimately", "in summary", "taken together"],
                "summary_markers",
            ),
            llm_marker_sentence_patterns=_load_marker_pattern_map(
                marker_config.get("llm_marker_sentence_patterns")
                or analysis.get("llm_marker_sentence_patterns")
                or {
                    "not_only_also": r"\bnot\s+only\b[^.!?;:\n]{0,240}\b(?:but\s+)?also\b"
                }
            ),
            enable_list_of_three_marker=bool(
                marker_config.get(
                    "enable_list_of_three_marker",
                    analysis.get("enable_list_of_three_marker", True),
                )
            ),
            list_of_three_pattern=str(
                marker_config.get(
                    "list_of_three_pattern",
                    analysis.get(
                        "list_of_three_pattern",
                        (
                            r"\b[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,3}\s*,\s*"
                            r"[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,3}\s*,\s*"
                            r"(?:and|or)\s+[a-z][a-z'-]*(?:\s+[a-z][a-z'-]*){0,3}\b"
                        ),
                    ),
                )
            ).strip(),
            llm_marker_word_matching=_load_marker_word_matching(analysis, marker_config),
            hedge_terms=_load_query_list(
                lexicon_config.get(
                    "hedge_terms",
                    analysis.get("hedge_terms", ["may", "might", "could", "suggest", "indicate"]),
                )
            ),
            certainty_terms=_load_query_list(
                lexicon_config.get(
                    "certainty_terms",
                    analysis.get("certainty_terms", ["demonstrate", "prove", "show", "confirm"]),
                )
            ),
            preprocessed_jsonl=_load_optional_path(analysis.get("preprocessed_jsonl"), default_preprocessed),
            feature_dataset_jsonl=_load_optional_path(analysis.get("feature_dataset_jsonl"), default_feature_dataset),
            trends_csv=_load_optional_path(analysis.get("trends_csv"), default_trends_csv),
            monthly_trends_csv=_load_optional_path(
                analysis.get("monthly_trends_csv"), default_monthly_trends_csv
            ),
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

    if source == "medarxiv":
        queries = _load_query_list(collection.get("medarxiv_queries"))
        if queries:
            return queries
        # medRxiv API is date-window based, so a wildcard query means "collect all".
        return ["*"]

    if source == "bioarxiv":
        queries = _load_query_list(collection.get("bioarxiv_queries"))
        if queries:
            return queries
        # bioRxiv API is date-window based, so a wildcard query means "collect all".
        return ["*"]

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


def _load_lexicon_terms(
    lexicon_config: dict[str, Any],
    legacy_config: dict[str, Any],
    analysis: dict[str, Any],
    key: str,
    default: list[str],
    legacy_key: str | None = None,
) -> list[str]:
    legacy_name = legacy_key or key
    return _load_query_list(
        lexicon_config.get(
            key,
            legacy_config.get(
                legacy_name,
                analysis.get(legacy_name, default),
            ),
        )
    )


def _load_marker_pattern_map(raw_patterns: Any) -> dict[str, str]:
    if not isinstance(raw_patterns, dict):
        return {}

    patterns: dict[str, str] = {}
    for key, value in raw_patterns.items():
        name = str(key).strip()
        pattern = str(value).strip()
        if not name or not pattern:
            continue
        patterns[name] = pattern
    return patterns


def _load_collection_source(collection: dict[str, Any]) -> str:
    raw_source = str(collection.get("source", "semantic_scholar")).strip().lower()
    allowed_sources = {"semantic_scholar", "arxiv", "medarxiv", "bioarxiv"}
    if raw_source not in allowed_sources:
        raise ValueError(
            "collection.source must be one of: semantic_scholar, arxiv, medarxiv, bioarxiv"
        )
    return raw_source


def _load_arxiv_collection_mode(collection: dict[str, Any]) -> str:
    raw_mode = str(collection.get("arxiv_collection_mode", "full")).strip().lower()
    allowed_modes = {"full", "monthly"}
    if raw_mode not in allowed_modes:
        raise ValueError("collection.arxiv_collection_mode must be one of: full, monthly")
    return raw_mode


def _load_medarxiv_collection_mode(collection: dict[str, Any]) -> str:
    raw_mode = str(collection.get("medarxiv_collection_mode", "monthly")).strip().lower()
    allowed_modes = {"full", "monthly"}
    if raw_mode not in allowed_modes:
        raise ValueError("collection.medarxiv_collection_mode must be one of: full, monthly")
    return raw_mode


def _load_bioarxiv_collection_mode(collection: dict[str, Any]) -> str:
    raw_mode = str(collection.get("bioarxiv_collection_mode", "monthly")).strip().lower()
    allowed_modes = {"full", "monthly"}
    if raw_mode not in allowed_modes:
        raise ValueError("collection.bioarxiv_collection_mode must be one of: full, monthly")
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

    if source == "medarxiv":
        source_specific_path = collection.get("medarxiv_output_jsonl")
        if isinstance(source_specific_path, str) and source_specific_path.strip():
            return Path(source_specific_path.strip())

    if source == "bioarxiv":
        source_specific_path = collection.get("bioarxiv_output_jsonl")
        if isinstance(source_specific_path, str) and source_specific_path.strip():
            return Path(source_specific_path.strip())

    fallback_path = collection.get("output_jsonl")
    if isinstance(fallback_path, str) and fallback_path.strip():
        return Path(fallback_path.strip())

    raise ValueError(
        "collection must define source-specific output path "
        "('arxiv_monthly_output_jsonl'/'arxiv_output_jsonl' for arxiv, "
        "'semantic_scholar_output_jsonl' for semantic_scholar, "
        "'medarxiv_output_jsonl' for medarxiv, "
        "'bioarxiv_output_jsonl' for bioarxiv) or fallback 'output_jsonl'"
    )


def _load_optional_path(raw_value: Any, default: Path) -> Path:
    if isinstance(raw_value, str) and raw_value.strip():
        return Path(raw_value.strip())
    return default


def _load_marker_word_matching(analysis: dict[str, Any], marker_config: dict[str, Any]) -> str:
    mode = str(
        marker_config.get(
            "llm_marker_word_matching",
            analysis.get("llm_marker_word_matching", "exact"),
        )
    ).strip().lower()
    allowed = {"exact", "lemma"}
    if mode not in allowed:
        raise ValueError("analysis.llm_marker_word_matching must be one of: exact, lemma")
    return mode


def _load_syntactic_features(analysis: dict[str, Any]) -> dict[str, str]:
    raw = analysis.get("syntactic_features")
    if raw is None:
        return {
            "em_dash": r"--",
            "semicolon": r";",
        }

    if not isinstance(raw, dict):
        raise ValueError("analysis.syntactic_features must be a table of feature_name = regex_pattern")

    features: dict[str, str] = {}
    for key, value in raw.items():
        name = str(key).strip()
        pattern = str(value).strip()
        if not name or not pattern:
            continue
        features[name] = pattern

    if not features:
        raise ValueError("analysis.syntactic_features cannot be empty when provided")
    return features


def _load_readability_metrics(analysis: dict[str, Any]) -> list[str]:
    defaults = [
        "avg_words_per_sentence",
        "avg_syllables_per_word",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "dale_chall",
    ]
    metrics = _load_query_list(analysis.get("readability_metrics", defaults))
    if not metrics:
        return defaults

    allowed = {
        "avg_words_per_sentence",
        "avg_syllables_per_word",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "dale_chall",
        "gunning_fog",
        "smog_index",
    }
    invalid = [metric for metric in metrics if metric not in allowed]
    if invalid:
        raise ValueError(
            "analysis.readability_metrics contains unsupported metrics: " + ", ".join(invalid)
        )
    return metrics


def _default_analysis_paths(raw_output_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    suffix = raw_output_path.suffix or ".jsonl"
    preprocessed = Path("data/processed") / raw_output_path.name
    feature_dataset = Path("data/analyzed") / f"{raw_output_path.stem}{suffix}"
    trends_csv = Path("data/analysis") / f"{raw_output_path.stem}_feature_trends_by_year.csv"
    monthly_trends_csv = Path("data/analysis") / f"{raw_output_path.stem}_feature_trends_by_month.csv"
    trends_plot_dir = Path("data/analysis/plots") / raw_output_path.stem
    return preprocessed, feature_dataset, trends_csv, monthly_trends_csv, trends_plot_dir
