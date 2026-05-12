from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import spacy
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

from not_an_llm.analysis.feature_extractor import FeatureExtractor, _slugify
from not_an_llm.analysis.readability import ReadabilityAnalyzer
from not_an_llm.analysis.trends import TrendAnalyzer
from not_an_llm.config import AppConfig

from .feature_groups import FEATURE_GROUPS
from .label_map import LABEL_MAP


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnalysisArtifacts:
    feature_dataset_jsonl: Path
    trends_csv: Path
    monthly_trends_csv: Path
    trends_plot_paths: list[Path]


# =========================================================
# PATH RESOLUTION
# =========================================================
def _resolve_analysis_paths(config: AppConfig):
    # Derive output names from the preprocessed input file basename to distinguish mini/full runs
    input_stem = config.analysis.preprocessed_jsonl.stem
    base_dir = Path(config.data_dir) / "analysis"

    base_dir.mkdir(parents=True, exist_ok=True)

    def _maybe(path_value: str, default: Path) -> Path:
        return Path(path_value) if path_value else default

    feature_dataset = _maybe(
        config.analysis.feature_dataset_jsonl,
        base_dir / f"{input_stem}.jsonl",
    )

    trends_csv = _maybe(
        config.analysis.trends_csv,
        base_dir / f"{input_stem}_trends_by_year.csv",
    )

    monthly_csv = _maybe(
        config.analysis.monthly_trends_csv,
        base_dir / f"{input_stem}_trends_by_month.csv",
    )

    plot_dir = _maybe(
        config.analysis.trends_plot_dir,
        base_dir / f"{input_stem}_plots",
    )

    return feature_dataset, trends_csv, monthly_csv, plot_dir


# =========================================================
# MARKER GROUPS
# =========================================================
def _build_marker_group_specs(config: AppConfig):
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
# WORKER (PARALLEL CHUNK PROCESSING)
# =========================================================
def _process_chunk_worker(args):
    chunk, config = args

    import spacy
    from not_an_llm.analysis.feature_extractor import FeatureExtractor
    from not_an_llm.analysis.readability import ReadabilityAnalyzer

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

    readability = None
    if config.analysis.include_readability:
        readability = ReadabilityAnalyzer(metrics=config.analysis.readability_metrics)

    enriched = feature_extractor.transform(chunk)

    if readability:
        enriched = readability.transform(enriched)

    return enriched


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_analysis(config: AppConfig) -> AnalysisArtifacts:
    input_path = config.analysis.preprocessed_jsonl
    if not input_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {input_path}. Run preprocess first."
        )

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
    # CHUNK LOADING
    # =========================
    chunks = pd.read_json(input_path, lines=True, chunksize=2000)
    chunks_list = list(chunks)

    enriched_output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if enriched_output_path.exists():
        enriched_output_path.unlink()

    # =========================
    # PARALLEL EXECUTION
    # =========================
    num_workers = 4
    logger.info(f"Using {num_workers} workers")

    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, enriched_chunk in enumerate(
            executor.map(_process_chunk_worker, [(c, config) for c in chunks_list])
        ):
            enriched_chunk.to_json(
                enriched_output_path,
                orient="records",
                lines=True,
                force_ascii=False,
                mode="w" if i == 0 else "a",
            )
            results.append(enriched_chunk)

    # =========================
    # CONCAT
    # =========================
    enriched = pd.concat(results, ignore_index=True)
    for col in enriched.columns:
        if col.endswith("_per_1k_words"):
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(0.0)

    feature_columns = _resolve_feature_columns(config, enriched)

    trend_analyzer = TrendAnalyzer(feature_columns)
    marker_group_specs, summary_features = _build_marker_group_specs(config)

    yearly = trend_analyzer.aggregate_yearly(enriched)

    # =========================
    # STATISTICAL ANALYSIS
    # =========================
    logger.info("Computing statistical summaries...")

    stats_df = trend_analyzer.compute_all_stats(yearly)

    stats_path = plot_dir / "feature_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    logger.info("Saved statistical summary to %s", stats_path)

    # =========================
    # PRE/POST DIFF PLOTS
    # =========================
    if config.analysis.generate_plots:
        logger.info("Generating grouped pre/post diff plot...")
        diff_plot_path = trend_analyzer.save_pre_post_diff_plot(
            yearly,
            output_dir=plot_dir,
        )
        logger.info("Saved grouped pre/post diff plot to %s", diff_plot_path)

        logger.info("Generating per-group pre/post diff plots...")
        if hasattr(trend_analyzer, "save_grouped_feature_diffs"):
            grouped_diff_plots = trend_analyzer.save_grouped_feature_diffs(
                yearly,
                output_dir=plot_dir,
            )
            logger.info("Saved %d per-group diff plots", len(grouped_diff_plots))
        else:
            # Backwards-compatible fallback: compute grouped diffs inline and save plots
            logger.warning("TrendAnalyzer lacks 'save_grouped_feature_diffs'; using fallback implementation.")
            from not_an_llm.analysis.trends import save_grouped_difference_plot

            grouped_diff_plots = {}
            pre = yearly[yearly["year"] <= 2022]
            post = yearly[yearly["year"] >= 2023]

            for group_name, members in FEATURE_GROUPS.items():
                rows = []
                valid_cols = []
                for m in members:
                    col = f"{m}_yearly_mean"
                    if col not in yearly.columns:
                        continue
                    valid_cols.append(col)
                    pre_vals = pd.to_numeric(pre[col], errors="coerce")
                    post_vals = pd.to_numeric(post[col], errors="coerce")
                    if pre_vals.dropna().empty or post_vals.dropna().empty:
                        continue
                    pre_mean = float(pre_vals.mean())
                    post_mean = float(post_vals.mean())
                    scale = abs(pre_mean) + abs(post_mean)
                    pct_change = 0.0 if scale == 0.0 else 200.0 * (post_mean - pre_mean) / scale
                    rows.append({"feature": m, "diff": pct_change})

                if not rows:
                    continue

                df = pd.DataFrame(rows)

                df = df.sort_values("diff")
                out_path = plot_dir / f"{group_name}_diff.png"
                save_grouped_difference_plot(
                    df,
                    out_path,
                    "feature",
                    "diff",
                    "Percent change (%)",
                    LABEL_MAP,
                )
                grouped_diff_plots[group_name] = out_path
            logger.info("Saved %d per-group diff plots (fallback)", len(grouped_diff_plots))

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
    trend_plots = []
    if config.analysis.generate_plots:
        llm_events = {
            "ChatGPT release": "2022-11-30",
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
            )
        )

        dep_plot_path = trend_analyzer.save_dependency_distribution_plot(
            df=enriched,
            output_dir=plot_dir,
            events=llm_events,
        )

        trend_plots.append(dep_plot_path)

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
# FEATURE SELECTION (UNCHANGED)
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
        return [column for column in numeric_cols if _is_canonical_analysis_feature(column)]

    return [
        column
        for column in requested
        if column in frame.columns and _is_canonical_analysis_feature(column)
    ]


def _is_canonical_analysis_feature(column_name: str) -> bool:
    excluded = {
        "word_count",
        "sentence_count",
        "paper_count",
        "marker_density",
        # "coordination_density",
        # "clause_depth_per_sentence",
        # "dependency_entropy_normalized",
        # "dependency_length_norm",
        # "sentence_depth_cv",
        # "coordination_count_per_1k_words",
        # "list_of_three_per_1k_words",
    }

    if column_name in excluded:
        return False

    if column_name in {
        "clause_depth",
        "clause_depth_std",
        "dependency_entropy",
        "dependency_length",
        "dependency_length_std",
        "coordination_count",
        "coordination_per_sentence_std",
        "sentence_depth_std",
        "list_of_three",
        "avg_words_per_sentence",
        "avg_syllables_per_word",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "dale_chall",
        "hedge_ratio",
        "certainty_ratio",
    }:
        return True

    if column_name.endswith("_total_per_1k_words"):
        return True

    if column_name.endswith("_per_1k_words") and not column_name.endswith("_count_per_1k_words"):
        return True
    
    if column_name.startswith("verb_") or column_name.startswith("adjective_") or column_name.startswith("word_") or column_name.startswith("phrase_"):
        return True

    return False


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