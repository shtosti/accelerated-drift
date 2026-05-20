from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import pandas as pd

from not_an_llm.analysis.trends import TrendAnalyzer
from not_an_llm.analysis.feature_groups import FEATURE_GROUPS
from not_an_llm.analysis.feature_selection import build_marker_group_specs, resolve_feature_columns
from not_an_llm.analysis.interrupted_time_series import (
    compute_interrupted_time_series,
    save_its_slope_change_plot,
)
from not_an_llm.analysis.label_map import LABEL_MAP
from not_an_llm.config import AppConfig


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VisualizationArtifacts:
    trends_plot_paths: list[Path]


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_visualization(config: AppConfig) -> VisualizationArtifacts:
    feature_dataset_path = config.analysis.feature_dataset_jsonl
    trends_csv_path = config.analysis.trends_csv
    monthly_trends_csv_path = config.analysis.monthly_trends_csv
    plot_dir = config.analysis.trends_plot_dir

    if not feature_dataset_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at {feature_dataset_path}. Run analyze first."
        )

    if not trends_csv_path.exists():
        raise FileNotFoundError(
            f"Trends CSV not found at {trends_csv_path}. Run analyze first."
        )

    if not monthly_trends_csv_path.exists():
        raise FileNotFoundError(
            f"Monthly trends CSV not found at {monthly_trends_csv_path}. Run analyze first."
        )

    print("Loading feature dataset from:", feature_dataset_path)
    print("Loading yearly trends from:", trends_csv_path)
    print("Loading monthly trends from:", monthly_trends_csv_path)
    print("Saving plots to:", plot_dir)

    # =========================
    # LOAD DATA
    # =========================
    enriched = _load_dependency_distribution(feature_dataset_path)
    yearly = pd.read_csv(trends_csv_path)
    monthly = pd.read_csv(monthly_trends_csv_path)

    # Ensure year/month columns exist in monthly trend data.
    if "month_ts" in monthly.columns and ("year" not in monthly.columns or "month" not in monthly.columns):
        monthly["month_ts"] = pd.to_datetime(monthly["month_ts"], errors="coerce")
        monthly["year"] = monthly["month_ts"].dt.year
        monthly["month"] = monthly["month_ts"].dt.month

    if "year" in yearly.columns:
        yearly["year"] = yearly["year"].astype(int)

    if "year" in monthly.columns:
        monthly["year"] = monthly["year"].astype(int)

    if "month" in monthly.columns:
        monthly["month"] = monthly["month"].astype(int)

    plot_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = resolve_feature_columns(config, yearly, monthly, enriched)

    trend_analyzer = TrendAnalyzer(feature_columns)
    marker_group_specs, summary_features = build_marker_group_specs(config)

    # =========================
    # LOAD PRIMARY STATISTICS
    # =========================
    input_stem = config.analysis.preprocessed_jsonl.stem
    analysis_dir = Path(config.data_dir) / "analysis"
    its_stats_path = analysis_dir / f"{input_stem}_its_stats.csv"
    if its_stats_path.exists():
        its_stats = pd.read_csv(its_stats_path)
        logger.info("Loaded monthly interrupted time-series statistics from %s", its_stats_path)
    else:
        logger.warning(
            "Interrupted time-series statistics not found at %s; computing them in memory for plotting only.",
            its_stats_path,
        )
        its_stats = compute_interrupted_time_series(monthly, feature_columns)

    # =========================
    # PRE/POST DIFF PLOTS
    # =========================
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
                "change (%)",
                LABEL_MAP,
            )
            if saved_path is not None:
                grouped_diff_plots[group_name] = saved_path
        logger.info("Saved %d per-group diff plots (fallback)", len(grouped_diff_plots))

    # =========================
    # PLOTS
    # =========================
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

    its_plot_path = save_its_slope_change_plot(
        its_stats,
        plot_dir / "its_slope_changes.png",
        label_map=LABEL_MAP,
    )
    if its_plot_path is not None:
        trend_plots.append(its_plot_path)

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

    verb_stack_plot_path = trend_analyzer.save_verb_prefix_stack_plot(
        yearly=yearly,
        monthly=monthly,
        output_dir=plot_dir,
        events=llm_events,
        smoothing_window=3,
    )
    trend_plots.append(verb_stack_plot_path)

    adjective_stack_plot_path = trend_analyzer.save_adjective_prefix_stack_plot(
        yearly=yearly,
        monthly=monthly,
        output_dir=plot_dir,
        events=llm_events,
        smoothing_window=3,
    )
    trend_plots.append(adjective_stack_plot_path)

    readability_stack_plot_path = trend_analyzer.save_readability_stack_plot(
        yearly=yearly,
        monthly=monthly,
        output_dir=plot_dir,
        events=llm_events,
        smoothing_window=3,
    )
    trend_plots.append(readability_stack_plot_path)

    punctuation_stack_plot_path = trend_analyzer.save_punctuation_stack_plot(
        yearly=yearly,
        monthly=monthly,
        output_dir=plot_dir,
        events=llm_events,
        smoothing_window=3,
    )
    trend_plots.append(punctuation_stack_plot_path)

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

    return VisualizationArtifacts(
        trends_plot_paths=trend_plots,
    )

def _load_dependency_distribution(path: Path) -> pd.DataFrame:
    """Load only the fields needed for dependency distribution plotting.

    The full analyzed JSONL includes large text columns that are unnecessary
    for visualization. This avoids blowing up memory by reading only year and
    dependency_distribution.
    """
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "year" in record and "dependency_distribution" in record:
                rows.append(
                    {
                        "year": record["year"],
                        "dependency_distribution": record["dependency_distribution"],
                    }
                )

    return pd.DataFrame(rows)
