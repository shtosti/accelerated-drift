from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import spacy
import pandas as pd

from not_an_llm.analysis.feature_extractor import FeatureExtractor
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
        yearly, monthly, plot_dir, llm_events
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