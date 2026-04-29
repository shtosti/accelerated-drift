from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy

from not_an_llm.analysis.feature_extractor import FeatureExtractor
from not_an_llm.analysis.readability import ReadabilityAnalyzer
from not_an_llm.analysis.trends import save_grouped_difference_plot
from not_an_llm.config import AppConfig


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExternalAnalysisArtifacts:
    feature_dataset_jsonl: Path
    comparison_csv: Path
    comparison_plot: Path


def _cohens_d(human: pd.Series, ai: pd.Series) -> float:
    x = pd.to_numeric(human, errors="coerce").dropna()
    y = pd.to_numeric(ai, errors="coerce").dropna()

    if len(x) < 2 or len(y) < 2:
        return 0.0

    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return 0.0
    return float((y.mean() - x.mean()) / np.sqrt(pooled))


def _expand_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, row in frame.iterrows():
        base = {
            "pair_id": row.get("pair_id"),
            "subset": row.get("subset"),
            "split": row.get("split"),
            "question": row.get("question"),
        }

        human = row.get("human") or {}
        ai = row.get("ai") or {}

        rows.append(
            {
                **base,
                "source_group": "human",
                "text_raw": str(human.get("text_raw", "")),
                "text_clean": str(human.get("text_clean", "")),
                "text_lemma": str(human.get("text_lemma", "")),
                "word_count": pd.to_numeric(human.get("word_count", 0), errors="coerce"),
                "sentence_count": pd.to_numeric(human.get("sentence_count", 0), errors="coerce"),
            }
        )

        rows.append(
            {
                **base,
                "source_group": "ai",
                "text_raw": str(ai.get("text_raw", "")),
                "text_clean": str(ai.get("text_clean", "")),
                "text_lemma": str(ai.get("text_lemma", "")),
                "word_count": pd.to_numeric(ai.get("word_count", 0), errors="coerce"),
                "sentence_count": pd.to_numeric(ai.get("sentence_count", 0), errors="coerce"),
            }
        )

    expanded = pd.DataFrame(rows)
    expanded["word_count"] = expanded["word_count"].fillna(0).astype(int)
    expanded["sentence_count"] = expanded["sentence_count"].fillna(0).astype(int)
    return expanded


def _build_comparison_table(enriched: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    human_df = enriched[enriched["source_group"] == "human"]
    ai_df = enriched[enriched["source_group"] == "ai"]

    for feature in feature_columns:
        if feature not in enriched.columns:
            continue

        values = pd.to_numeric(enriched[feature], errors="coerce")
        if values.dropna().empty:
            continue

        h = pd.to_numeric(human_df[feature], errors="coerce")
        a = pd.to_numeric(ai_df[feature], errors="coerce")

        human_mean = float(h.mean()) if not h.dropna().empty else np.nan
        ai_mean = float(a.mean()) if not a.dropna().empty else np.nan

        diff = ai_mean - human_mean if pd.notna(human_mean) and pd.notna(ai_mean) else np.nan
        ratio = ai_mean / human_mean if pd.notna(human_mean) and human_mean not in (0, 0.0) else np.nan

        rows.append(
            {
                "feature": feature,
                "human_mean": human_mean,
                "human_std": float(h.std()) if not h.dropna().empty else np.nan,
                "ai_mean": ai_mean,
                "ai_std": float(a.std()) if not a.dropna().empty else np.nan,
                "diff_ai_minus_human": diff,
                "ratio_ai_over_human": ratio,
                "cohens_d": _cohens_d(h, a),
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values("diff_ai_minus_human", ascending=False, key=lambda s: s.abs())


def _is_canonical_analysis_feature(column_name: str) -> bool:
    excluded = {
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
        "list_of_three_per_1k_words",
    }

    if column_name in excluded:
        return False

    if column_name in {
        "clause_depth",
        "dependency_entropy",
        "dependency_length",
        "coordination_count",
        "sentence_depth_std",
        "list_of_three",
    }:
        return True

    if column_name.endswith("_total_per_1k_words"):
        return True

    if column_name.endswith("_per_1k_words") and not column_name.endswith("_count_per_1k_words"):
        return True

    return False


def _save_top_diff_plot(comparison: pd.DataFrame, output_path: Path, top_n: int = 20) -> Path:
    logger.info("Generating grouped external diff plot...")
    saved_path = save_grouped_difference_plot(
        comparison,
        output_path=output_path,
        feature_column="feature",
        diff_column="diff_ai_minus_human",
        title="Top feature shifts: AI - Human",
        xlabel="Mean difference",
        top_n=top_n,
    )
    logger.info("Saved grouped external diff plot to %s", saved_path)
    return saved_path


def run_external_analysis(
    config: AppConfig,
    *,
    input_jsonl: str | Path,
    feature_dataset_jsonl: str | Path,
    comparison_csv: str | Path,
    comparison_plot: str | Path,
) -> ExternalAnalysisArtifacts:
    input_path = Path(input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"External pair dataset not found at {input_path}")

    pairs = pd.read_json(input_path, lines=True)
    expanded = _expand_pairs(pairs)

    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    nlp.max_length = 2_000_000

    feature_extractor = FeatureExtractor(
        nlp=nlp,
        syntactic_features=config.analysis.syntactic_features,
        marker_words=config.analysis.llm_marker_words,
        marker_verbs=config.analysis.llm_marker_verbs,
        marker_adjectives=config.analysis.llm_marker_adjectives,
        marker_phrases=config.analysis.llm_marker_phrases,
        sequential_markers=config.analysis.sequential_markers,
        causal_markers=config.analysis.causal_markers,
        contrast_markers=config.analysis.contrast_markers,
        emphasis_markers=config.analysis.emphasis_markers,
        summary_markers=config.analysis.summary_markers,
        enable_list_of_three_marker=config.analysis.enable_list_of_three_marker,
        marker_word_matching=config.analysis.llm_marker_word_matching,
        hedges=config.analysis.hedge_terms,
        certainty_terms=config.analysis.certainty_terms,
    )

    enriched = feature_extractor.transform(expanded)

    if config.analysis.include_readability:
        readability = ReadabilityAnalyzer(metrics=config.analysis.readability_metrics)
        enriched = readability.transform(enriched)

    feature_output_path = Path(feature_dataset_jsonl)
    feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_json(feature_output_path, orient="records", lines=True, force_ascii=False)

    numeric_features = [
        c for c in enriched.columns
        if pd.api.types.is_numeric_dtype(enriched[c])
        and c not in {"pair_id", "word_count", "sentence_count"}
        and _is_canonical_analysis_feature(c)
    ]

    comparison = _build_comparison_table(enriched, numeric_features)
    comparison_path = Path(comparison_csv)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)

    plot_path = _save_top_diff_plot(comparison, Path(comparison_plot))

    return ExternalAnalysisArtifacts(
        feature_dataset_jsonl=feature_output_path,
        comparison_csv=comparison_path,
        comparison_plot=plot_path,
    )


def run_configured_external_analysis(config: AppConfig) -> ExternalAnalysisArtifacts:
    if config.external is None:
        raise ValueError("Missing [external] section in config; use config_external.toml for external pipelines.")

    external = config.external
    if external.dataset == "hc3":
        return run_external_analysis(
            config,
            input_jsonl=external.hc3_pair_output_jsonl,
            feature_dataset_jsonl=external.hc3_feature_output_jsonl,
            comparison_csv=external.hc3_comparison_csv,
            comparison_plot=external.hc3_comparison_plot,
        )

    if external.dataset == "mage":
        return run_external_analysis(
            config,
            input_jsonl=external.mage_pair_output_jsonl,
            feature_dataset_jsonl=external.mage_feature_output_jsonl,
            comparison_csv=external.mage_comparison_csv,
            comparison_plot=external.mage_comparison_plot,
        )

    raise ValueError(f"Unsupported external dataset: {external.dataset}")
