from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import spacy
import pandas as pd

from not_an_llm.analysis.feature_extractor import FeatureExtractor, _slugify
from not_an_llm.analysis.readability import ReadabilityAnalyzer
from not_an_llm.analysis.trends import TrendAnalyzer
from not_an_llm.config import AppConfig


@dataclass(slots=True)
class AnalysisArtifacts:
    feature_dataset_jsonl: Path
    trends_csv: Path
    monthly_trends_csv: Path
    trends_plot_paths: list[Path]


# =========================================================
# PATH RESOLUTION (AUTO BY SOURCE)
# =========================================================
def _resolve_analysis_paths(config: AppConfig):
    source = config.collection.source
    base_dir = Path(config.data_dir) / "analysis"

    base_dir.mkdir(parents=True, exist_ok=True)

    def _maybe(path_value: str, default: Path) -> Path:
        return Path(path_value) if path_value else default

    feature_dataset = _maybe(
        config.analysis.feature_dataset_jsonl,
        base_dir / f"{source}.jsonl",
    )

    trends_csv = _maybe(
        config.analysis.trends_csv,
        base_dir / f"{source}_trends_by_year.csv",
    )

    monthly_csv = _maybe(
        config.analysis.monthly_trends_csv,
        base_dir / f"{source}_trends_by_month.csv",
    )

    plot_dir = _maybe(
        config.analysis.trends_plot_dir,
        base_dir / f"{source}_plots",
    )

    return feature_dataset, trends_csv, monthly_csv, plot_dir


def _build_marker_group_specs(config: AppConfig) -> tuple[dict[str, dict[str, object]], set[str]]:
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
    }

    for group_name, (label, prefix, terms, rate_feature, count_feature) in group_definitions.items():
        group_specs[group_name] = {
            "label": label,
            "rate_feature": rate_feature,
            "count_feature": count_feature,
        }
        summary_features.add(rate_feature)
        summary_features.add(count_feature)

    summary_features.add("marker_density")
    return group_specs, summary_features


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_analysis(config: AppConfig) -> AnalysisArtifacts:
    input_path = config.analysis.preprocessed_jsonl
    if not input_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {input_path}. Run preprocess first."
        )

    # -------------------------
    # RESOLVE OUTPUT PATHS
    # -------------------------
    (
        enriched_output_path,
        trends_csv,
        monthly_trends_csv,
        plot_dir,
    ) = _resolve_analysis_paths(config)

    print("Saving feature dataset to:", enriched_output_path)
    print("Saving yearly trends to:", trends_csv)
    print("Saving monthly trends to:", monthly_trends_csv)
    print("Saving plots to:", plot_dir)

    # =========================
    # LOAD SPACY
    # =========================
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["ner", "textcat"]
    )
    nlp.max_length = 2_000_000

    feature_extractor = FeatureExtractor(
        nlp=nlp,
        syntactic_features=config.analysis.syntactic_features,
        marker_words=config.analysis.llm_marker_words,
        marker_verbs=config.analysis.llm_marker_verbs,
        marker_adjectives=config.analysis.llm_marker_adjectives,
        marker_phrases=config.analysis.llm_marker_phrases,
        enable_list_of_three_marker=config.analysis.enable_list_of_three_marker,
        marker_word_matching=config.analysis.llm_marker_word_matching,
        hedges=config.analysis.hedge_terms,
        certainty_terms=config.analysis.certainty_terms,
    )

    readability = None
    if config.analysis.include_readability:
        readability = ReadabilityAnalyzer(metrics=config.analysis.readability_metrics)

    # =========================
    # STREAM DATA IN CHUNKS
    # =========================
    chunks = pd.read_json(input_path, lines=True, chunksize=2000)

    enriched_output_path.parent.mkdir(parents=True, exist_ok=True)

    if enriched_output_path.exists():
        enriched_output_path.unlink()

    first_chunk = True
    collected_frames = []

    for chunk in chunks:
        enriched_chunk = feature_extractor.transform(chunk)

        if readability:
            enriched_chunk = readability.transform(enriched_chunk)

        enriched_chunk.to_json(
            enriched_output_path,
            orient="records",
            lines=True,
            force_ascii=False,
            mode="w" if first_chunk else "a"
        )
        first_chunk = False

        collected_frames.append(enriched_chunk)

    # =========================
    # CONCAT FOR ANALYSIS
    # =========================
    enriched = pd.concat(collected_frames, ignore_index=True)

    feature_columns = _resolve_feature_columns(config, enriched)

    trend_analyzer = TrendAnalyzer(feature_columns)
    marker_group_specs, summary_features = _build_marker_group_specs(config)

    yearly = trend_analyzer.aggregate_yearly(enriched)
    monthly = trend_analyzer.aggregate_monthly(enriched)

    # =========================
    # SAVE TABLES
    # =========================
    trends_csv.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(trends_csv, index=False)

    monthly_trends_csv.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(monthly_trends_csv, index=False)

    # =========================
    # PLOTS
    # =========================
    llm_events = {
        "ChatGPT release": "2022-11-30",
        # "Delve study": "2024-01-15",
    }

    trend_plots = trend_analyzer.save_plots(
        yearly,
        monthly,
        plot_dir,
        llm_events,
        exclude_features=summary_features,
    )

    trend_plots.extend(
        trend_analyzer.save_grouped_word_plots(
            yearly=yearly,
            monthly=monthly,
            output_dir=plot_dir,
            group_specs=marker_group_specs,
            events=llm_events,
            smoothing_window=3,
            exclude_features=summary_features,
        )
    )

    dep_plot_path = trend_analyzer.save_dependency_distribution_plot(
        df=enriched,
        output_dir=plot_dir,
        events=llm_events,
    )

    trend_plots.append(dep_plot_path)

    # -------------------------
    # EVENT-STACKED PLOT
    # -------------------------

    stacked_plot_path = trend_analyzer.save_stacked_word_plots(
        yearly=yearly,
        monthly=monthly,
        output_dir=plot_dir,
        events=llm_events,
        smoothing_window=3,
        exclude_features=summary_features,
    )

    trend_plots.append(stacked_plot_path)

    return AnalysisArtifacts(
        feature_dataset_jsonl=enriched_output_path,
        trends_csv=trends_csv,
        monthly_trends_csv=monthly_trends_csv,
        trends_plot_paths=trend_plots,
    )


# =========================================================
# FEATURE SELECTION
# =========================================================
def _resolve_feature_columns(config: AppConfig, frame: pd.DataFrame) -> list[str]:
    metadata_exclusions = {
        "year",
        "paperId",
        "citationCount",
        "influentialCitationCount",
        "isOpenAccess",
    }

    numeric_cols = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and column not in metadata_exclusions
    ]

    requested = [item.strip() for item in config.analysis.features if item.strip()]
    if not requested or requested == ["all"]:
        return [column for column in numeric_cols if not _is_metadata_like(column)]

    return [column for column in requested if column in frame.columns]


def _is_metadata_like(column_name: str) -> bool:
    lowered = column_name.lower()
    blocked_tokens = {
        "citation",
        "openaccess",
        "paperid",
        "publication",
        "externalid",
        "fields_of_study",
        "fieldsofstudy",
        "authors",
        "venue",
        "journal",
        "url",
        "tldr",
    }

    normalized = lowered.replace("_", "")
    return any(token in lowered or token in normalized for token in blocked_tokens)