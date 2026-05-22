from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from not_an_llm.analysis.topic_modeling import run_topic_analysis, run_topic_modeling
from not_an_llm.analysis.topic_modeling.comparison import (
    compare_topic_distributions_and_features,
    save_selected_its_features,
    select_top_its_features,
)
from not_an_llm.analysis.trends import TrendAnalyzer, is_group_total_feature
from not_an_llm.analysis.feature_groups import FEATURE_GROUPS
from not_an_llm.analysis.feature_selection import build_marker_group_specs, resolve_feature_columns
from not_an_llm.analysis.interrupted_time_series import (
    compute_interrupted_time_series,
    compute_placebo_interrupted_time_series,
    save_its_slope_change_plot,
    save_its_standardized_grouped_slope_change_plots,
    save_its_standardized_slope_change_plot,
)
from not_an_llm.analysis.label_map import LABEL_MAP
from not_an_llm.config import AppConfig


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
    analysis_dir = Path(config.data_dir) / "analysis"
    visuals_dir = Path(config.data_dir) / "visuals"

    analysis_dir.mkdir(parents=True, exist_ok=True)

    def _maybe(path_value: Path, default: Path) -> Path:
        # Use default if path_value is empty or matches the old analysis/plots default
        if not path_value or str(path_value).startswith("data/visuals"):
            return default
        return path_value

    feature_dataset = _maybe(
        config.analysis.feature_dataset_jsonl,
        analysis_dir / f"{input_stem}.jsonl",
    )

    trends_csv = _maybe(
        config.analysis.trends_csv,
        analysis_dir / f"{input_stem}_trends_by_year.csv",
    )

    monthly_csv = _maybe(
        config.analysis.monthly_trends_csv,
        analysis_dir / f"{input_stem}_trends_by_month.csv",
    )

    plot_dir = _maybe(
        config.analysis.trends_plot_dir,
        visuals_dir / input_stem,
    )

    return feature_dataset, trends_csv, monthly_csv, plot_dir


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
    import multiprocessing

    num_workers = config.analysis.topic_modeling_num_workers or multiprocessing.cpu_count()
    num_workers = max(1, min(num_workers, multiprocessing.cpu_count()))
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

    analysis_dir = Path(config.data_dir) / "analysis"
    enriched, topic_labels, embeddings_2d, topic_modeling_paths = run_topic_modeling(
        enriched=enriched,
        config=config,
        analysis_dir=analysis_dir,
    )
    enriched.to_json(
        enriched_output_path,
        orient="records",
        lines=True,
        force_ascii=False,
        mode="w",
    )

    feature_columns = resolve_feature_columns(config, enriched)

    trend_analyzer = TrendAnalyzer(feature_columns)
    marker_group_specs, summary_features = build_marker_group_specs(config)

    yearly = trend_analyzer.aggregate_yearly(enriched)
    monthly = trend_analyzer.aggregate_monthly(enriched)

    # =========================
    # STATISTICAL ANALYSIS
    # =========================
    logger.info("Computing monthly interrupted time-series statistics...")
    input_stem = config.analysis.preprocessed_jsonl.stem
    its_stats = compute_interrupted_time_series(monthly, feature_columns)
    its_stats_path = analysis_dir / f"{input_stem}_its_stats.csv"
    its_stats.to_csv(its_stats_path, index=False)
    logger.info("Saved monthly interrupted time-series statistics to %s", its_stats_path)

    placebo_stats = compute_placebo_interrupted_time_series(monthly, feature_columns)
    placebo_stats_path = analysis_dir / f"{input_stem}_its_placebo_stats.csv"
    placebo_stats.to_csv(placebo_stats_path, index=False)
    logger.info("Saved placebo interrupted time-series statistics to %s", placebo_stats_path)

    # =========================
    # PRE/POST DIFF PLOTS
    # =========================
    if config.analysis.generate_plots:
        logger.info("Generating grouped pre/post diff plot...")
        diff_plot_path = trend_analyzer.save_pre_post_diff_plot(
            yearly,
            output_dir=plot_dir,
        )
        if diff_plot_path is not None:
            logger.info("Saved grouped pre/post diff plot to %s", diff_plot_path)
        else:
            logger.info("Skipped grouped pre/post diff plot because no finite rows were available")

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
                    if is_group_total_feature(m):
                        continue

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
                    pct_change = (
                        float("nan")
                        if pre_mean == 0.0
                        else 100.0 * (post_mean - pre_mean) / abs(pre_mean)
                    )
                    rows.append({"feature": m, "diff": pct_change})

                if not rows:
                    continue

                df = pd.DataFrame(rows).dropna(subset=["diff"])
                if df.empty:
                    continue

                df = df.sort_values("diff")
                out_path = plot_dir / f"{group_name}_diff.png"
                saved_path = save_grouped_difference_plot(
                    df,
                    out_path,
                    "feature",
                    "diff",
                    "Percent change from pre-period baseline (%)",
                    LABEL_MAP,
                )
                if saved_path is not None:
                    grouped_diff_plots[group_name] = saved_path
            logger.info("Saved %d per-group diff plots (fallback)", len(grouped_diff_plots))

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
    }

    trend_plots = list(topic_modeling_paths)
    if config.analysis.generate_plots:
        its_plot_path = save_its_slope_change_plot(
            its_stats,
            plot_dir / "its_slope_changes.png",
            label_map=LABEL_MAP,
        )
        standardized_its_plot_path = save_its_standardized_slope_change_plot(
            its_stats,
            plot_dir / "its_slope_changes_standardized.png",
            label_map=LABEL_MAP,
        )
        standardized_grouped_its_plot_paths = save_its_standardized_grouped_slope_change_plots(
            its_stats,
            plot_dir / "its_slope_changes_standardized_groups",
            label_map=LABEL_MAP,
        )

        trend_plots.extend(
            trend_analyzer.save_plots(
                yearly,
                monthly,
                plot_dir,
                llm_events,
                exclude_features=summary_features,
            )
        )
        if its_plot_path is not None:
            trend_plots.append(its_plot_path)
        if standardized_its_plot_path is not None:
            trend_plots.append(standardized_its_plot_path)
        trend_plots.extend(standardized_grouped_its_plot_paths)

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

        word_stack_plot_path = trend_analyzer.save_word_prefix_stack_plot(
            yearly=yearly,
            monthly=monthly,
            output_dir=plot_dir,
            events=llm_events,
            smoothing_window=3,
        )
        trend_plots.append(word_stack_plot_path)

        dep_plot_paths = trend_analyzer.save_dependency_distribution_plot(
            df=enriched,
            output_dir=plot_dir,
            events=llm_events,
        )

        trend_plots.extend(dep_plot_paths)

        stacked_plot_path = trend_analyzer.save_stacked_word_plots(
            yearly=yearly,
            monthly=monthly,
            output_dir=plot_dir,
            events=llm_events,
            smoothing_window=3,
            exclude_features=summary_features,
        )

        trend_plots.append(stacked_plot_path)

    if config.analysis.topic_modeling_enabled:
        topic_paths = run_topic_analysis(
            enriched=enriched,
            config=config,
            plot_dir=plot_dir,
            trend_analyzer=trend_analyzer,
            group_specs=marker_group_specs,
            summary_features=summary_features,
            events=llm_events,
            embeddings_2d=embeddings_2d,
        )
        trend_plots.extend(topic_paths)
        trend_plots.extend(_maybe_run_cross_domain_topic_comparison(analysis_dir))

    return AnalysisArtifacts(
        feature_dataset_jsonl=enriched_output_path,
        trends_csv=trends_csv,
        monthly_trends_csv=monthly_trends_csv,
        trends_plot_paths=trend_plots,
    )


def _maybe_run_cross_domain_topic_comparison(analysis_dir: Path) -> list[Path]:
    domains = ("arxiv", "medarxiv")
    required_paths = []
    for domain in domains:
        required_paths.extend(
            [
                analysis_dir / f"{domain}_its_stats.csv",
                analysis_dir / f"{domain}_topic_summary.csv",
                analysis_dir / f"{domain}_topic_prevalence_yearly.csv",
                analysis_dir / f"{domain}_topics",
            ]
        )

    missing = [path for path in required_paths if not path.exists()]
    if missing:
        logger.info(
            "Skipping cross-domain topic comparison because not all arXiv/medRxiv topic artifacts exist yet."
        )
        return []

    output_dir = analysis_dir / "topic_comparison"
    features, selected = select_top_its_features(
        analysis_dir=analysis_dir,
        domains=domains,
        top_n=20,
        q_threshold=0.05,
    )
    selected_path = save_selected_its_features(
        selected,
        output_dir / "selected_top_its_features.csv",
    )
    artifacts = compare_topic_distributions_and_features(
        analysis_dir=analysis_dir,
        output_dir=output_dir,
        domains=domains,
        features=features,
        min_topic_share=0.005,
        make_heatmap=True,
    )

    paths = [
        selected_path,
        artifacts.distribution_csv,
        artifacts.feature_strength_csv,
        artifacts.its_stats_csv,
        artifacts.combined_csv,
    ]
    if artifacts.percent_change_heatmap_path:
        paths.append(artifacts.percent_change_heatmap_path)
    if artifacts.standardized_its_heatmap_path:
        paths.append(artifacts.standardized_its_heatmap_path)

    logger.info("Saved cross-domain topic comparison outputs to %s", output_dir)
    return paths

