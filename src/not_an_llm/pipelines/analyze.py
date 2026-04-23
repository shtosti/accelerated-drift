from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import spacy

from not_an_llm.analysis.trends import TrendAnalyzer
from not_an_llm.analysis.unified_features import UnifiedLinguisticFeatures
from not_an_llm.config import AppConfig


@dataclass(slots=True)
class AnalysisArtifacts:
    feature_dataset_jsonl: Path
    trends_csv: Path
    trends_plot_paths: list[Path]


def run_analysis(config: AppConfig) -> AnalysisArtifacts:
    input_path = config.analysis.preprocessed_jsonl
    if not input_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {input_path}. Run preprocess first."
        )

    frame = pd.read_json(input_path, lines=True)
    
    # Load spaCy pipeline for single-pass feature extraction
    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except Exception:
        nlp = spacy.blank("en")
    
    # Single-pass unified feature extraction
    unified = UnifiedLinguisticFeatures(
        syntactic_features=config.analysis.syntactic_features,
        marker_phrases=config.analysis.llm_marker_phrases,
        marker_words=config.analysis.llm_marker_words,
        marker_word_matching=config.analysis.llm_marker_word_matching,
        hedge_terms=config.analysis.hedge_terms,
        certainty_terms=config.analysis.certainty_terms,
        readability_metrics=config.analysis.readability_metrics if config.analysis.include_readability else [],
    )
    
    # Single unified pass: tokenize + extract all features
    texts = frame["text_clean"].fillna("").astype(str).tolist()
    docs = list(nlp.pipe(texts, batch_size=128))
    
    feature_rows = []
    for text, doc in zip(texts, docs):
        feature_rows.append(unified.extract(text, doc))
    
    features_df = pd.DataFrame(feature_rows)
    enriched = pd.concat([frame.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)


    enriched_output_path = config.analysis.feature_dataset_jsonl
    enriched_output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_json(enriched_output_path, orient="records", lines=True, force_ascii=False)

    feature_columns = _resolve_feature_columns(config, enriched)
    trend_analyzer = TrendAnalyzer(feature_columns)
    yearly = trend_analyzer.aggregate_yearly(enriched)

    trends_csv = config.analysis.trends_csv
    trends_csv.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(trends_csv, index=False)

    trend_plots = trend_analyzer.save_plots(yearly, config.analysis.trends_plot_dir)

    return AnalysisArtifacts(
        feature_dataset_jsonl=enriched_output_path,
        trends_csv=trends_csv,
        trends_plot_paths=trend_plots,
    )


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
        if pd.api.types.is_numeric_dtype(frame[column]) and column not in metadata_exclusions
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
