from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from not_an_llm.analysis.label_map import pretty_feature_label
from not_an_llm.analysis.visual_style import (
    CATEGORICAL_COLORS,
    CORPUS_LINESTYLES,
    DATASET_COLORS,
    DATASET_HATCHES,
    DATASET_MARKERS,
    DARK_GREY,
    DECREASE_COLOR,
    INCREASE_COLOR,
    ORCHID_GREEN_DIVERGING_CMAP,
    WHITE,
    sign_hatch,
)


DEFAULT_PRIMARY_DATASETS = (
    "arxiv_ai_abstracts",
    "arxiv_qbio_abstracts",
    "arxiv_stat_abstracts",
    "medarxiv_abstracts",
)

MANUSCRIPT_ABSTRACT_FEATURES = (
    "word_across_per_1k_words",
    "marker_words_total_per_1k_words",
    "verb_align_per_1k_words",
    "em_dash_per_1k_words",
    "avg_syllables_per_word",
    "flesch_kincaid_grade",
    "flesch_reading_ease",
    "clause_depth",
    "clause_depth_std",
    "dependency_entropy",
    "hedge_ratio",
    "certainty_ratio",
)
MANUSCRIPT_FEATURE_LABELS = {
    "word_across_per_1k_words": r"$\it{across}$",
    "marker_words_total_per_1k_words": "marker words",
    "verb_align_per_1k_words": r"$\it{align}$",
    "em_dash_per_1k_words": "em dash",
    "avg_syllables_per_word": "syllables/word",
    "flesch_kincaid_grade": "FKGL",
    "flesch_reading_ease": "FRE",
    "clause_depth": "max tree depth",
    "clause_depth_std": r"tree depth $\sigma$",
    "dependency_entropy": "dep. entropy",
    "hedge_ratio": "hedging",
    "certainty_ratio": "certainty",
}
READABILITY_FEATURES = (
    "avg_syllables_per_word",
    "dale_chall",
    "automated_readability_index",
    "flesch_kincaid_grade",
    "gunning_fog",
    "smog_index",
    "flesch_reading_ease",
)
READABILITY_FEATURE_LABELS = {
    "avg_syllables_per_word": "syllables/word",
    "dale_chall": "Dale-Chall",
    "automated_readability_index": "ARI",
    "flesch_kincaid_grade": "FKGL",
    "gunning_fog": "Gunning Fog",
    "smog_index": "SMOG",
    "flesch_reading_ease": "FRE",
}
MANUSCRIPT_TABLE_GROUPS = {
    "Lexical markers": {"marker_words", "verbs", "adjectives", "phrases"},
    "Punctuation": {"punctuation"},
    "Readability": {"readability"},
    "Syntax": {"syntax"},
    "Rhetorical stance": {
        "causal_markers", "contrast_markers", "discourse_marker_totals",
        "emphasis_markers", "sequential_markers", "summary_markers", "other",
    },
}
FAMILY_AGGREGATE_FEATURES = {
    "marker_words_total_per_1k_words",
    "marker_verbs_total_per_1k_words",
    "marker_adjectives_total_per_1k_words",
    "marker_phrases_total_per_1k_words",
}
SYNTAX_FEATURE_HINTS = (
    "dependency",
    "clause",
    "sentence_depth",
    "coordination",
    "list_of_three",
    "avg_words_per_sentence",
)
SHORT_DATASET_LABELS = {
    "arxiv_ai": "AI",
    "arxiv_qbio": "qbio",
    "arxiv_stat": "stat",
    "medarxiv": "medRxiv",
    "arxiv_ai_abstracts": "AI",
    "arxiv_qbio_abstracts": "q-bio",
    "arxiv_stat_abstracts": "stat",
    "medarxiv_abstracts": "medRxiv",
    "arxiv_ai_titles": "AI title",
    "arxiv_qbio_titles": "q-bio title",
    "arxiv_stat_titles": "stat title",
    "medarxiv_titles": "medRxiv title",
}
COMPARISON_HEATMAP_CMAP = ORCHID_GREEN_DIVERGING_CMAP
DEPENDENCY_TREND_COLORS = CATEGORICAL_COLORS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ITS slope-change profiles across corpora."
    )
    parser.add_argument("--analysis-dir", default="data/analysis")
    parser.add_argument("--visuals-dir", default="data/visuals")
    parser.add_argument("--output-name", default="corpus_comparison")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to compare. Defaults to non-mini title and abstract folders with its_stats.csv.",
    )
    parser.add_argument(
        "--primary-datasets",
        nargs="+",
        default=list(DEFAULT_PRIMARY_DATASETS),
        help="Datasets emphasized in compact bar/scatter plots.",
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    analysis_output_dir = analysis_dir / args.output_name
    visuals_output_dir = Path(args.visuals_dir) / args.output_name
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    visuals_output_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or discover_datasets(analysis_dir)
    primary_datasets = [name for name in args.primary_datasets if name in datasets]

    stats = load_all_its(analysis_dir, datasets)
    if stats.empty:
        raise SystemExit("No ITS statistics found for the requested datasets.")

    matrix = build_slope_matrix(stats)
    matrix.to_csv(analysis_output_dir / "standardized_slope_matrix.csv", index=False)

    table_rows, table_audit = build_manuscript_feature_table(stats, primary_datasets)
    table_rows.to_csv(analysis_output_dir / "manuscript_feature_table.csv", index=False)
    table_audit.to_csv(analysis_output_dir / "manuscript_feature_selection_audit.csv", index=False)
    save_manuscript_feature_table_latex(
        table_rows,
        primary_datasets,
        analysis_output_dir / "manuscript_feature_table.tex",
    )

    pairwise = pairwise_correlations(matrix, datasets)
    pairwise.to_csv(analysis_output_dir / "corpus_pairwise_correlations.csv", index=False)

    group_corr = feature_group_correlations(stats, datasets)
    group_corr.to_csv(analysis_output_dir / "feature_group_correlations.csv", index=False)

    summary = summarize_dataset_similarity(pairwise)
    summary.to_csv(analysis_output_dir / "corpus_similarity_summary.csv", index=False)

    disagreements = top_feature_disagreements(matrix, datasets)
    disagreements.to_csv(analysis_output_dir / "top_feature_disagreements.csv", index=False)

    syntax = syntax_dependency_subset(stats)
    syntax.to_csv(analysis_output_dir / "syntax_dependency_slope_comparison.csv", index=False)
    syntax_corr = pairwise_unit_correlations(
        syntax,
        unit_column="feature",
        value_column="standardized_slope_change_per_year",
        n_column="n_features",
    )
    syntax_corr.to_csv(
        analysis_output_dir / "syntax_dependency_pairwise_correlations.csv",
        index=False,
    )

    dependency_roles = compute_dependency_role_slopes(analysis_dir, datasets)
    dependency_roles.to_csv(analysis_output_dir / "dependency_role_slope_changes.csv", index=False)
    dependency_role_corr = pairwise_unit_correlations(
        dependency_roles,
        unit_column="dependency_role",
        value_column="slope_change_per_year",
        n_column="n_roles",
    )
    dependency_role_corr.to_csv(
        analysis_output_dir / "dependency_role_pairwise_correlations.csv",
        index=False,
    )

    dependency_bigrams = compute_dependency_bigram_slopes(analysis_dir, datasets)
    dependency_bigrams.to_csv(analysis_output_dir / "dependency_bigram_slope_changes.csv", index=False)
    dependency_bigram_corr = pairwise_unit_correlations(
        dependency_bigrams,
        unit_column="dependency_bigram",
        value_column="slope_change_per_year",
        n_column="n_bigrams",
    )
    dependency_bigram_corr.to_csv(
        analysis_output_dir / "dependency_bigram_pairwise_correlations.csv",
        index=False,
    )

    save_correlation_heatmap(
        pairwise,
        datasets,
        visuals_output_dir / "corpus_correlation_heatmap.png",
    )
    save_group_correlation_heatmap(
        group_corr,
        visuals_output_dir / "feature_group_correlation_heatmap.png",
    )
    save_pairwise_scatter_grid(
        matrix,
        primary_datasets or datasets[:4],
        visuals_output_dir / "primary_corpus_slope_scatter.png",
    )
    save_abstract_replicated_effects(
        stats,
        primary_datasets,
        visuals_output_dir / "abstract_replicated_effects.png",
    )
    save_readability_effects(
        stats,
        primary_datasets,
        visuals_output_dir / "readability_standardized_effects.png",
    )
    save_readability_syntax_heatmap(
        stats,
        primary_datasets,
        visuals_output_dir / "readability_syntax_heatmap.png",
    )
    save_all_feature_family_effects(
        stats,
        primary_datasets,
        visuals_output_dir / "feature_group_effects",
    )
    save_determiner_context_trends(
        analysis_dir,
        primary_datasets,
        visuals_output_dir / "determiner_context_trends.png",
    )
    save_cross_corpus_feature_trends(
        analysis_dir,
        primary_datasets,
        features=(
            "word_across_per_1k_words",
            "word_insight_per_1k_words",
            "verb_delve_per_1k_words",
            "avg_syllables_per_word",
            "clause_depth",
        ),
        path=visuals_output_dir / "lexical_example_trajectories.png",
    )
    save_syntax_dependency_bars(
        syntax,
        primary_datasets or datasets[:4],
        visuals_output_dir / "syntax_dependency_slope_bars.png",
    )
    save_pairwise_correlation_heatmap(
        syntax_corr,
        datasets,
        visuals_output_dir / "syntax_dependency_correlation_heatmap.png",
        colorbar_label="Pearson r",
    )
    save_top_disagreement_bars(
        disagreements,
        primary_datasets or datasets[:4],
        visuals_output_dir / "top_feature_disagreements.png",
    )
    if not dependency_roles.empty:
        save_dependency_role_bars(
            dependency_roles,
            primary_datasets or datasets[:4],
            visuals_output_dir / "dependency_role_slope_bars.png",
        )
    save_pairwise_correlation_heatmap(
        dependency_role_corr,
        datasets,
        visuals_output_dir / "dependency_role_correlation_heatmap.png",
        colorbar_label="Pearson r",
    )
    if not dependency_bigrams.empty:
        save_dependency_bigram_bars(
            dependency_bigrams,
            primary_datasets or datasets[:4],
            visuals_output_dir / "dependency_bigram_slope_bars.png",
        )
    for text_kind in ("titles", "abstracts"):
        kind_datasets = [dataset for dataset in datasets if dataset.endswith(f"_{text_kind}")]
        ai_dataset = f"arxiv_ai_{text_kind}"
        if ai_dataset not in kind_datasets:
            continue
        dependency_role_yearly = load_dependency_role_yearly(analysis_dir, kind_datasets)
        save_cross_corpus_dependency_trends(
            dependency_role_yearly,
            unit_column="dependency_role",
            datasets=kind_datasets,
            path=visuals_output_dir / f"dependency_role_trends_{text_kind}.png",
            ylabel="Dependency role share",
        )
        dependency_bigram_yearly = load_dependency_bigram_yearly(analysis_dir, kind_datasets)
        save_cross_corpus_dependency_trends(
            dependency_bigram_yearly,
            unit_column="dependency_bigram",
            datasets=kind_datasets,
            path=visuals_output_dir / f"dependency_edge_bigram_trends_{text_kind}.png",
            ylabel="Dependency edge share",
        )
    save_pairwise_correlation_heatmap(
        dependency_bigram_corr,
        datasets,
        visuals_output_dir / "dependency_bigram_correlation_heatmap.png",
        colorbar_label="Pearson r",
    )

    print(f"Saved corpus comparison tables to {analysis_output_dir}")
    print(f"Saved corpus comparison figures to {visuals_output_dir}")
    print(summary.to_string(index=False))


def save_cross_corpus_feature_trends(
    analysis_dir: Path,
    datasets: list[str],
    features: tuple[str, ...],
    path: Path,
) -> None:
    """Plot compact monthly and yearly trajectories for the same features across corpora."""
    datasets = [dataset for dataset in datasets if dataset in DATASET_COLORS]
    if not datasets or not features:
        return

    loaded: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for dataset in datasets:
        monthly_path = analysis_dir / dataset / "trends_by_month.csv"
        yearly_path = analysis_dir / dataset / "trends_by_year.csv"
        if not monthly_path.exists() or not yearly_path.exists():
            continue
        monthly = pd.read_csv(monthly_path)
        yearly = pd.read_csv(yearly_path).copy()
        monthly["month_ts"] = pd.to_datetime(monthly["month_ts"], errors="coerce")
        yearly["year"] = pd.to_numeric(yearly["year"], errors="coerce")
        yearly["year_ts"] = pd.to_datetime(yearly["year"].astype("Int64").astype(str), errors="coerce")
        loaded[dataset] = (monthly, yearly)
    if not loaded:
        return

    fig_width = 1.5 * len(features)
    fig, axes = plt.subplots(1, len(features), figsize=(fig_width, 1.62), squeeze=False, sharex=True)
    axes = axes.ravel()
    for index, (ax, feature) in enumerate(zip(axes, features)):
        yearly_col = f"{feature}_yearly_mean"
        for dataset in datasets:
            if dataset not in loaded:
                continue
            _, yearly = loaded[dataset]
            color = DATASET_COLORS[dataset]
            if yearly_col in yearly:
                values = pd.to_numeric(yearly[yearly_col], errors="coerce")
                ax.plot(
                    yearly["year_ts"], values, color=color, lw=0.85,
                    marker=DATASET_MARKERS[dataset], markersize=2.4,
                )
        ax.axvline(pd.Timestamp("2022-11-30"), color="0.25", lw=0.65, ls="--")
        if feature.startswith(("word_", "verb_")) and feature.endswith("_per_1k_words"):
            word = feature.split("_", 1)[1].removesuffix("_per_1k_words")
            ylabel = rf"$\it{{{word}}}$"
        else:
            ylabel = {
                "avg_syllables_per_word": "syllables/word",
                "clause_depth": "max tree depth",
            }.get(feature, pretty_feature_label(feature))
        ax.set_ylabel(ylabel, fontsize=6)
        ax.set_xlabel("Year", fontsize=6)
        ax.set_xlim(pd.Timestamp("2015-01-01"), pd.Timestamp("2026-03-01"))
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="both", labelsize=5.5, length=2, pad=1.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        ax.grid(alpha=0.22)
    fig.tight_layout(pad=0.35, w_pad=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.015)
    plt.close(fig)

    handles = [
        Line2D(
            [0], [0], color=DATASET_COLORS[dataset], lw=1.35,
            marker=DATASET_MARKERS[dataset], markersize=3.2,
            label=SHORT_DATASET_LABELS.get(dataset, dataset),
        )
        for dataset in datasets if dataset in loaded
    ]
    legend_fig = plt.figure(figsize=(2.15, 0.43))
    legend_fig.legend(
        handles=handles, loc="center", ncol=2, frameon=False,
        fontsize=7, handlelength=1.8, columnspacing=1.0, handletextpad=0.4,
        borderaxespad=0,
    )
    legend_fig.savefig(
        path.with_name(f"{path.stem}_legend{path.suffix}"), dpi=300,
        bbox_inches="tight", facecolor=WHITE, pad_inches=0.005,
    )
    plt.close(legend_fig)


def discover_datasets(analysis_dir: Path) -> list[str]:
    datasets = []
    for path in sorted(analysis_dir.iterdir()):
        if not path.is_dir() or "mini" in path.name:
            continue
        if not (path.name.endswith("_abstracts") or path.name.endswith("_titles")):
            continue
        if (path / "its_stats.csv").exists():
            datasets.append(path.name)
    return datasets


def load_all_its(analysis_dir: Path, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        path = analysis_dir / dataset / "its_stats.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        required = {"feature", "family", "standardized_slope_change_per_year"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.copy()
        frame.insert(0, "dataset", dataset)
        frame["standardized_slope_change_per_year"] = pd.to_numeric(
            frame["standardized_slope_change_per_year"],
            errors="coerce",
        )
        frame["slope_change_q"] = pd.to_numeric(frame.get("slope_change_q"), errors="coerce")
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_slope_matrix(stats: pd.DataFrame) -> pd.DataFrame:
    metadata = (
        stats[["feature", "family"]]
        .drop_duplicates("feature")
        .sort_values(["family", "feature"])
    )
    values = stats.pivot_table(
        index="feature",
        columns="dataset",
        values="standardized_slope_change_per_year",
        aggfunc="first",
    )
    result = metadata.merge(values.reset_index(), on="feature", how="left")
    return result


def _significance_stars(q_value: object) -> str:
    q = pd.to_numeric(pd.Series([q_value]), errors="coerce").iloc[0]
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def build_manuscript_feature_table(
    stats: pd.DataFrame,
    datasets: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select up to four individual features per prespecified manuscript group."""
    required = {
        "dataset", "feature", "family", "standardized_slope_change_per_year",
        "standardized_slope_change_per_year_ci_low",
        "standardized_slope_change_per_year_ci_high", "slope_change_q",
    }
    if not datasets or not required.issubset(stats.columns):
        return pd.DataFrame(), pd.DataFrame()
    data = stats[stats["dataset"].isin(datasets)].copy()
    numeric = [
        "standardized_slope_change_per_year",
        "standardized_slope_change_per_year_ci_low",
        "standardized_slope_change_per_year_ci_high",
        "slope_change_q",
    ]
    for column in numeric:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    summary_rows = []
    for (feature, family), group in data.groupby(["feature", "family"], sort=False):
        complete = group.dropna(subset=numeric).drop_duplicates("dataset")
        values = complete.set_index("dataset")["standardized_slope_change_per_year"]
        qs = complete.set_index("dataset")["slope_change_q"]
        all_complete = set(datasets).issubset(values.index)
        same_direction = bool(all_complete and ((values[datasets] > 0).all() or (values[datasets] < 0).all()))
        all_q05 = bool(all_complete and (qs[datasets] < 0.05).all())
        summary_rows.append({
            "feature": feature,
            "family": family,
            "all_four_estimable": all_complete,
            "same_direction": same_direction,
            "all_four_q_lt_05": all_q05,
            "min_abs_standardized_effect": float(values[datasets].abs().min()) if all_complete else np.nan,
            "mean_abs_standardized_effect": float(values[datasets].abs().mean()) if all_complete else np.nan,
            "is_group_aggregate": str(feature).endswith("_total_per_1k_words"),
        })
    audit = pd.DataFrame(summary_rows)
    family_to_group = {
        family: group
        for group, families in MANUSCRIPT_TABLE_GROUPS.items()
        for family in families
    }
    audit["manuscript_group"] = audit["family"].map(family_to_group)
    eligible = audit[
        audit["all_four_estimable"]
        & audit["all_four_q_lt_05"]
        & ~audit["is_group_aggregate"]
        & audit["manuscript_group"].notna()
    ].copy()

    selected = []
    for manuscript_group in MANUSCRIPT_TABLE_GROUPS:
        candidates = eligible[eligible["manuscript_group"] == manuscript_group].sort_values(
            ["mean_abs_standardized_effect", "feature"], ascending=[False, True]
        )
        # Avoid spending two rows on features with identical four-corpus
        # estimate profiles (currently clause_depth_std/sentence_depth_std).
        distinct_profiles = []
        for feature in candidates["feature"]:
            profile = tuple(
                np.round(
                    data[data["feature"] == feature]
                    .set_index("dataset")
                    .reindex(datasets)["standardized_slope_change_per_year"]
                    .to_numpy(dtype=float),
                    10,
                )
            )
            if profile in distinct_profiles:
                continue
            distinct_profiles.append(profile)
            selected.append(str(feature))
            if len(distinct_profiles) == 4:
                break
    audit["selected"] = audit["feature"].isin(selected)
    audit["selection_reason"] = np.where(
        audit["selected"],
        "top four individual features significant in all four corpora",
        "not selected",
    )
    rows = []
    for order, feature in enumerate(selected):
        feature_data = data[data["feature"] == feature]
        row = {
            "order": order + 1,
            "feature": feature,
            "label": _manuscript_table_label(feature),
            "family": str(feature_data["family"].iloc[0]),
            "manuscript_group": family_to_group[str(feature_data["family"].iloc[0])],
        }
        for dataset in datasets:
            match = feature_data[feature_data["dataset"] == dataset]
            if match.empty:
                continue
            result = match.iloc[0]
            estimate = float(result["standardized_slope_change_per_year"])
            low = float(result["standardized_slope_change_per_year_ci_low"])
            high = float(result["standardized_slope_change_per_year_ci_high"])
            q = float(result["slope_change_q"])
            row[f"{dataset}_estimate"] = estimate
            row[f"{dataset}_ci_low"] = low
            row[f"{dataset}_ci_high"] = high
            row[f"{dataset}_q"] = q
            row[f"{dataset}_stars"] = _significance_stars(q)
            row[dataset] = f"{estimate:.2f} [{low:.2f}, {high:.2f}]{_significance_stars(q)}"
        rows.append(row)
    return pd.DataFrame(rows), audit.sort_values(
        ["selected", "min_abs_standardized_effect"], ascending=[False, False]
    )


def _manuscript_table_label(feature: str) -> str:
    if feature in MANUSCRIPT_FEATURE_LABELS:
        return MANUSCRIPT_FEATURE_LABELS[feature]
    if feature in READABILITY_FEATURE_LABELS:
        return READABILITY_FEATURE_LABELS[feature]
    return pretty_feature_label(feature)


def _latex_label(label: str) -> str:
    if label.startswith("$\\it{"):
        return label
    return label.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")


def save_manuscript_feature_table_latex(table: pd.DataFrame, datasets: list[str], path: Path) -> None:
    if table.empty:
        path.unlink(missing_ok=True)
        return
    headers = [SHORT_DATASET_LABELS.get(dataset, dataset) for dataset in datasets]
    lines = [
        r"\begin{tabular}{l" + "c" * len(datasets) + "}",
        r"\toprule",
        "Feature & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    previous_family = None
    for row in table.itertuples(index=False):
        if previous_family is not None and row.manuscript_group != previous_family:
            lines.append(r"\addlinespace[2pt]")
        cells = []
        for dataset in datasets:
            estimate = getattr(row, f"{dataset}_estimate")
            low = getattr(row, f"{dataset}_ci_low")
            high = getattr(row, f"{dataset}_ci_high")
            stars = getattr(row, f"{dataset}_stars")
            star_text = rf"$^{{{stars}}}$" if stars else ""
            cells.append(rf"{estimate:.2f}{star_text} [{low:.2f}, {high:.2f}]")
        lines.append(_latex_label(row.label) + " & " + " & ".join(cells) + r" \\")
        previous_family = row.manuscript_group
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{flushleft}\footnotesize Standardized ITS slope changes with 95\% CIs. "
        r"$^{*}q<.05$, $^{**}q<.01$, $^{***}q<.001$ (family-level FDR).\end{flushleft}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pairwise_correlations(matrix: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for left, right in combinations(datasets, 2):
        if left not in matrix.columns or right not in matrix.columns:
            continue
        pair = matrix[["feature", "family", left, right]].dropna()
        if len(pair) < 2:
            continue
        pearson_r, pearson_p = safe_corr_p(pair[left], pair[right])
        rows.append(
            {
                "dataset_a": left,
                "dataset_b": right,
                "feature_set": "all",
                "n_features": len(pair),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "same_direction_share": float((pair[left] * pair[right] > 0).mean()),
                "median_abs_difference": float((pair[left] - pair[right]).abs().median()),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False).reset_index(drop=True)


def feature_group_correlations(stats: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    matrix = build_slope_matrix(stats)
    rows = []
    for family, family_df in matrix.groupby("family", dropna=False):
        for left, right in combinations(datasets, 2):
            if left not in family_df.columns or right not in family_df.columns:
                continue
            pair = family_df[[left, right]].dropna()
            if len(pair) < 2:
                continue
            pearson_r, pearson_p = safe_corr_p(pair[left], pair[right])
            rows.append(
                {
                    "family": family,
                    "dataset_a": left,
                    "dataset_b": right,
                    "n_features": len(pair),
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "same_direction_share": float((pair[left] * pair[right] > 0).mean()),
                    "median_abs_difference": float((pair[left] - pair[right]).abs().median()),
                }
            )
    return pd.DataFrame(rows).sort_values(["family", "pearson_r"], ascending=[True, False]).reset_index(drop=True)


def summarize_dataset_similarity(pairwise: pd.DataFrame) -> pd.DataFrame:
    if pairwise.empty:
        return pairwise
    rows = []
    for dataset in sorted(set(pairwise["dataset_a"]).union(pairwise["dataset_b"])):
        subset = pairwise[(pairwise["dataset_a"] == dataset) | (pairwise["dataset_b"] == dataset)]
        best = subset.sort_values("pearson_r", ascending=False).head(1)
        worst = subset.sort_values("pearson_r", ascending=True).head(1)
        rows.append(
            {
                "dataset": dataset,
                "mean_pearson_r": float(subset["pearson_r"].mean()),
                "best_match": _other_dataset(best.iloc[0], dataset) if not best.empty else pd.NA,
                "best_match_r": float(best["pearson_r"].iloc[0]) if not best.empty else pd.NA,
                "worst_match": _other_dataset(worst.iloc[0], dataset) if not worst.empty else pd.NA,
                "worst_match_r": float(worst["pearson_r"].iloc[0]) if not worst.empty else pd.NA,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_pearson_r", ascending=False)


def _other_dataset(row: pd.Series, dataset: str) -> str:
    return str(row["dataset_b"] if row["dataset_a"] == dataset else row["dataset_a"])


def top_feature_disagreements(matrix: pd.DataFrame, datasets: list[str], *, top_n: int = 80) -> pd.DataFrame:
    value_cols = [dataset for dataset in datasets if dataset in matrix.columns]
    result = matrix[["feature", "family", *value_cols]].copy()
    result["range"] = result[value_cols].max(axis=1) - result[value_cols].min(axis=1)
    result["mean_abs_slope"] = result[value_cols].abs().mean(axis=1)
    return result.sort_values(["range", "mean_abs_slope"], ascending=False).head(top_n).reset_index(drop=True)


def syntax_dependency_subset(stats: pd.DataFrame) -> pd.DataFrame:
    result = stats[
        (stats["family"].astype(str) == "syntax")
        | stats["feature"].astype(str).map(lambda value: any(hint in value for hint in SYNTAX_FEATURE_HINTS))
    ].copy()
    return result.sort_values(["feature", "dataset"]).reset_index(drop=True)


def compute_dependency_role_slopes(analysis_dir: Path, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        path = analysis_dir / dataset / "features.jsonl"
        if not path.exists():
            continue
        records = []
        for chunk in pd.read_json(path, lines=True, chunksize=5000):
            if not {"year", "dependency_distribution"}.issubset(chunk.columns):
                continue
            for row in chunk[["year", "dependency_distribution"]].itertuples(index=False):
                if not isinstance(row.dependency_distribution, dict):
                    continue
                total = sum(float(value) for value in row.dependency_distribution.values())
                if total <= 0:
                    continue
                for role, count in row.dependency_distribution.items():
                    records.append(
                        {
                            "year": int(row.year),
                            "dependency_role": role,
                            "count": float(count),
                            "total": total,
                        }
                    )
        if not records:
            continue
        yearly = pd.DataFrame(records).groupby(["year", "dependency_role"], as_index=False).sum()
        totals = yearly.groupby("year")["total"].first()
        yearly["proportion"] = yearly.apply(
            lambda row: row["count"] / totals.loc[row["year"]],
            axis=1,
        )
        for role, group in yearly.groupby("dependency_role"):
            fit = _simple_post_slope_change(group[["year", "proportion"]], "proportion")
            if fit is None:
                continue
            rows.append({"dataset": dataset, "dependency_role": role, **fit})
    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "dependency_role",
            "slope_change_per_year",
            "slope_change_per_year_se",
            "slope_change_per_year_ci_low",
            "slope_change_per_year_ci_high",
        ],
    )


def load_dependency_role_yearly(analysis_dir: Path, datasets: list[str]) -> pd.DataFrame:
    """Load yearly role shares, retaining counts so the AI corpus can define the top ten."""
    rows = []
    for dataset in datasets:
        path = analysis_dir / dataset / "features.jsonl"
        if not path.exists():
            continue
        records = []
        for chunk in pd.read_json(path, lines=True, chunksize=5000):
            if not {"year", "dependency_distribution"}.issubset(chunk.columns):
                continue
            for row in chunk[["year", "dependency_distribution"]].itertuples(index=False):
                if not isinstance(row.dependency_distribution, dict):
                    continue
                for role, count in row.dependency_distribution.items():
                    records.append({"year": row.year, "dependency_role": role, "count": count})
        if not records:
            continue
        yearly = pd.DataFrame(records)
        yearly["year"] = pd.to_numeric(yearly["year"], errors="coerce")
        yearly["count"] = pd.to_numeric(yearly["count"], errors="coerce")
        yearly = yearly.dropna(subset=["year", "dependency_role", "count"])
        yearly = yearly.groupby(["year", "dependency_role"], as_index=False)["count"].sum()
        totals = yearly.groupby("year")["count"].transform("sum")
        yearly["proportion"] = yearly["count"] / totals.where(totals > 0)
        yearly.insert(0, "dataset", dataset)
        rows.append(yearly)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_dependency_bigram_yearly(analysis_dir: Path, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        path = analysis_dir / dataset / "additional_analysis" / "dependency_bigram_yearly.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        required = {"year", "dependency_bigram", "count", "proportion"}
        if frame.empty or not required.issubset(frame.columns):
            continue
        frame = frame[["year", "dependency_bigram", "count", "proportion"]].copy()
        frame.insert(0, "dataset", dataset)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def save_cross_corpus_dependency_trends(
    yearly: pd.DataFrame,
    *,
    unit_column: str,
    datasets: list[str],
    path: Path,
    ylabel: str,
    top_n: int = 10,
    min_mean_prevalence: float = 0.001,
) -> None:
    """Plot ITS-ranked dependency units with color=unit and line style=corpus."""
    required = {"dataset", "year", unit_column, "count", "proportion"}
    if yearly.empty or not required.issubset(yearly.columns):
        _remove_dependency_trend_artifacts(path)
        return
    data = yearly.copy()
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["count"] = pd.to_numeric(data["count"], errors="coerce")
    data["proportion"] = pd.to_numeric(data["proportion"], errors="coerce")
    data = data.dropna(subset=["year", unit_column, "count", "proportion"])
    data[unit_column] = data[unit_column].astype(str)
    top_units = _top_units_by_standardized_its_change(
        data,
        unit_column=unit_column,
        datasets=datasets,
        top_n=top_n,
        min_mean_prevalence=min_mean_prevalence,
    )
    if not top_units:
        _remove_dependency_trend_artifacts(path)
        return
    data = data[data[unit_column].isin(top_units)]
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(7, 3.5),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    for index, (ax, unit) in enumerate(zip(axes.flat, top_units, strict=False)):
        for dataset in datasets:
            corpus = dataset.removesuffix("_titles").removesuffix("_abstracts")
            series = data[(data["dataset"] == dataset) & (data[unit_column] == unit)].sort_values("year")
            if series.empty:
                continue
            ax.plot(
                series["year"], series["proportion"],
                color=DATASET_COLORS[dataset], linestyle=CORPUS_LINESTYLES.get(corpus, "-"),
                linewidth=1.15,
            )
        ax.axvline(2022.92, color="0.25", linestyle="--", linewidth=0.7, alpha=0.65)
        ax.text(0.03, 0.95, unit, transform=ax.transAxes, va="top",
                color="0.15", fontsize=7.5)
        ax.grid(alpha=0.22)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.tick_params(axis="both", labelsize=6)
        ax.tick_params(axis="x", rotation=45)
    for ax in axes.flat[len(top_units):]:
        ax.set_visible(False)
    fig.supylabel(ylabel, fontsize=8)
    fig.supxlabel("Year", fontsize=8)
    corpus_handles = [
        Line2D([0], [0], color=DATASET_COLORS[next(dataset for dataset in datasets if dataset.startswith(f"{corpus}_"))],
               linestyle=CORPUS_LINESTYLES[corpus], linewidth=1.4,
               label=SHORT_DATASET_LABELS.get(corpus, corpus))
        for corpus in CORPUS_LINESTYLES
        if any(dataset.startswith(f"{corpus}_") for dataset in datasets)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)
    _save_cross_corpus_dependency_legend(
        [],
        corpus_handles,
        path.with_name(f"{path.stem}_legend{path.suffix}"),
    )


def _remove_dependency_trend_artifacts(path: Path) -> None:
    """Remove stale figures when a complete four-corpus comparison cannot be built."""
    for candidate in (path, path.with_name(f"{path.stem}_legend{path.suffix}")):
        candidate.unlink(missing_ok=True)


def _top_units_by_standardized_its_change(
    data: pd.DataFrame,
    *,
    unit_column: str,
    datasets: list[str],
    top_n: int,
    min_mean_prevalence: float,
) -> list[str]:
    """Rank prevalent units by equal-weighted absolute standardized ITS slope change."""
    observed_datasets = set(data["dataset"])
    if not set(datasets).issubset(observed_datasets):
        return []
    units = data[unit_column].astype(str).unique().tolist()
    rows = []
    for dataset in datasets:
        corpus = data[data["dataset"] == dataset]
        years = sorted(corpus["year"].unique())
        if not years:
            return []
        proportions = (
            corpus.pivot_table(
                index="year",
                columns=unit_column,
                values="proportion",
                aggfunc="sum",
                fill_value=0.0,
            )
            .reindex(years, fill_value=0.0)
            .reindex(columns=units, fill_value=0.0)
        )
        for unit in units:
            series = pd.DataFrame({"year": years, "proportion": proportions[unit].to_numpy()})
            fit = _simple_post_slope_change(series, "proportion")
            pre_values = series.loc[series["year"] <= 2022, "proportion"]
            pre_sd = float(pre_values.std(ddof=1)) if len(pre_values) > 1 else float("nan")
            standardized_change = (
                float(fit["slope_change_per_year"]) / pre_sd
                if fit is not None and np.isfinite(pre_sd) and pre_sd > 0
                else float("nan")
            )
            rows.append(
                {
                    "dataset": dataset,
                    unit_column: unit,
                    "mean_prevalence": float(series["proportion"].mean()),
                    "standardized_slope_change_per_year": standardized_change,
                }
            )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return []
    summary = ranking.groupby(unit_column, as_index=True).agg(
        mean_prevalence=("mean_prevalence", "mean"),
        n_valid_corpora=("standardized_slope_change_per_year", "count"),
        mean_abs_standardized_change=(
            "standardized_slope_change_per_year",
            lambda values: values.abs().mean(),
        ),
    )
    eligible = summary[
        (summary["mean_prevalence"] >= min_mean_prevalence)
        & (summary["n_valid_corpora"] == len(datasets))
    ]
    return eligible.nlargest(top_n, "mean_abs_standardized_change").index.astype(str).tolist()


def _save_cross_corpus_dependency_legend(
    unit_handles: list[Line2D],
    corpus_handles: list[Line2D],
    path: Path,
) -> None:
    """Save dependency colors and corpus line styles as a standalone figure."""
    fig, ax = plt.subplots(figsize=(4.8, 1.05))
    ax.axis("off")
    ax.legend(
        handles=corpus_handles,
        loc="center",
        ncol=4,
        fontsize=6.5,
        frameon=False,
        title="Corpus",
        title_fontsize=7,
        handlelength=2.4,
        columnspacing=1.2,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.04)
    plt.close(fig)


def compute_dependency_bigram_slopes(analysis_dir: Path, datasets: list[str]) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        path = analysis_dir / dataset / "additional_analysis" / "dependency_bigram_yearly.csv"
        if not path.exists():
            continue
        yearly = pd.read_csv(path)
        required = {"year", "dependency_bigram", "proportion"}
        if yearly.empty or not required.issubset(yearly.columns):
            continue
        yearly = yearly.copy()
        yearly["year"] = pd.to_numeric(yearly["year"], errors="coerce")
        yearly["proportion"] = pd.to_numeric(yearly["proportion"], errors="coerce")
        yearly = yearly.dropna(subset=["year", "dependency_bigram", "proportion"])
        if yearly.empty:
            continue
        for bigram, group in yearly.groupby("dependency_bigram"):
            fit = _simple_post_slope_change(group[["year", "proportion"]], "proportion")
            if fit is None:
                continue
            head_dep, child_dep = split_dependency_bigram(str(bigram))
            rows.append(
                {
                    "dataset": dataset,
                    "dependency_bigram": bigram,
                    "head_dep": head_dep,
                    "child_dep": child_dep,
                    **fit,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "dataset",
            "dependency_bigram",
            "head_dep",
            "child_dep",
            "slope_change_per_year",
            "slope_change_per_year_se",
            "slope_change_per_year_ci_low",
            "slope_change_per_year_ci_high",
        ],
    )


def _simple_post_slope_change(frame: pd.DataFrame, value_column: str) -> dict[str, float] | None:
    data = frame.dropna().copy()
    data = data.sort_values("year")
    if data["year"].nunique() < 6 or (data["year"] >= 2023).sum() < 2:
        return None
    year = data["year"].to_numpy(dtype=float)
    time = year - year.min()
    post = (year >= 2023).astype(float)
    first_post = time[post == 1].min()
    time_after = np.where(post == 1, time - first_post + 1.0, 0.0)
    x = np.column_stack([np.ones(len(data)), time, post, time_after])
    y = data[value_column].to_numpy(dtype=float)
    if len(y) <= x.shape[1]:
        return None
    params = np.linalg.pinv(x.T @ x) @ x.T @ y
    residuals = y - x @ params
    df = max(1, len(y) - x.shape[1])
    sigma2 = float((residuals @ residuals) / df)
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    slope_change = float(params[3])
    slope_se = float(se[3])
    return {
        "slope_change_per_year": slope_change,
        "slope_change_per_year_se": slope_se,
        "slope_change_per_year_ci_low": slope_change - 1.96 * slope_se,
        "slope_change_per_year_ci_high": slope_change + 1.96 * slope_se,
    }


def dependency_role_pairwise_correlations(role_slopes: pd.DataFrame) -> pd.DataFrame:
    return pairwise_unit_correlations(
        role_slopes,
        unit_column="dependency_role",
        value_column="slope_change_per_year",
        n_column="n_roles",
    )


def pairwise_unit_correlations(
    slopes: pd.DataFrame,
    *,
    unit_column: str,
    value_column: str,
    n_column: str,
) -> pd.DataFrame:
    output_columns = [
        "dataset_a",
        "dataset_b",
        n_column,
        "pearson_r",
        "pearson_p",
        "same_direction_share",
        "median_abs_difference",
    ]
    if slopes.empty or not {unit_column, "dataset", value_column}.issubset(slopes.columns):
        return pd.DataFrame(columns=output_columns)
    matrix = slopes.pivot_table(
        index=unit_column,
        columns="dataset",
        values=value_column,
        aggfunc="first",
    ).reset_index()
    datasets = [column for column in matrix.columns if column != unit_column]
    rows = []
    for left, right in combinations(datasets, 2):
        pair = matrix[[left, right]].dropna()
        if len(pair) < 2:
            continue
        pearson_r, pearson_p = safe_corr_p(pair[left], pair[right])
        rows.append(
            {
                "dataset_a": left,
                "dataset_b": right,
                n_column: len(pair),
                "pearson_r": pearson_r,
                "pearson_p": pearson_p,
                "same_direction_share": float((pair[left] * pair[right] > 0).mean()),
                "median_abs_difference": float((pair[left] - pair[right]).abs().median()),
            }
        )
    if not rows:
        return pd.DataFrame(columns=output_columns)
    return pd.DataFrame(rows, columns=output_columns).sort_values("pearson_r", ascending=False).reset_index(drop=True)


def save_abstract_replicated_effects(
    stats: pd.DataFrame,
    datasets: list[str],
    path: Path,
) -> None:
    """Compact manuscript-facing forest plot for replicated abstract effects."""
    datasets = [dataset for dataset in datasets if dataset.endswith("_abstracts")]
    if not datasets:
        return
    data = stats[
        stats["dataset"].isin(datasets)
        & stats["feature"].isin(MANUSCRIPT_ABSTRACT_FEATURES)
    ].copy()
    if data.empty:
        return

    feature_order = [
        feature for feature in MANUSCRIPT_ABSTRACT_FEATURES
        if feature in set(data["feature"])
    ]
    y_base = np.arange(len(feature_order), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(datasets))
    fig, ax = plt.subplots(figsize=(4.8, 3.45))
    for row in range(len(feature_order)):
        if row % 2 == 0:
            ax.axhspan(row - 0.5, row + 0.5, color="0.94", zorder=0)
    for offset, dataset in zip(offsets, datasets, strict=True):
        subset = data[data["dataset"] == dataset].set_index("feature").reindex(feature_order)
        estimate = subset["standardized_slope_change_per_year"].to_numpy(dtype=float)
        low = subset["standardized_slope_change_per_year_ci_low"].to_numpy(dtype=float)
        high = subset["standardized_slope_change_per_year_ci_high"].to_numpy(dtype=float)
        xerr = np.vstack([estimate - low, high - estimate])
        ax.errorbar(
            estimate,
            y_base + offset,
            xerr=xerr,
            fmt=DATASET_MARKERS[dataset],
            color=DATASET_COLORS[dataset],
            ecolor=DATASET_COLORS[dataset],
            markersize=3.6,
            linewidth=0.7,
            capsize=1.5,
            label=SHORT_DATASET_LABELS.get(dataset, dataset),
        )
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_yticks(y_base, [MANUSCRIPT_FEATURE_LABELS[feature] for feature in feature_order])
    ax.invert_yaxis()
    ax.set_xlabel(r"Standardized $\Delta$ slope/year", fontsize=7.5)
    ax.tick_params(axis="both", labelsize=6.5)
    ax.grid(axis="x", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    save_marker_legend(datasets, path.with_name(f"{path.stem}_legend{path.suffix}"))


def save_readability_effects(stats: pd.DataFrame, datasets: list[str], path: Path) -> None:
    """Compact forest plot of every readability metric in the abstract analyses."""
    data = stats[
        stats["dataset"].isin(datasets) & stats["feature"].isin(READABILITY_FEATURES)
    ].copy()
    feature_order = [feature for feature in READABILITY_FEATURES if feature in set(data["feature"])]
    save_feature_family_effects(
        data,
        datasets,
        path,
        feature_order=feature_order,
        feature_labels=READABILITY_FEATURE_LABELS,
    )


def save_readability_syntax_heatmap(stats: pd.DataFrame, datasets: list[str], path: Path) -> None:
    """Compact four-corpus heatmap for the readability and syntax families."""
    datasets = [dataset for dataset in datasets if dataset.endswith("_abstracts")]
    data = stats[
        stats["dataset"].isin(datasets)
        & stats["family"].astype(str).isin(["readability", "syntax"])
    ].copy()
    if data.empty or not datasets:
        path.unlink(missing_ok=True)
        path.with_name(f"{path.stem}_legend{path.suffix}").unlink(missing_ok=True)
        return
    complete = data.groupby("feature")["dataset"].nunique()
    features = complete[complete == len(datasets)].index
    data = data[data["feature"].isin(features)].copy()
    data["standardized_slope_change_per_year"] = pd.to_numeric(
        data["standardized_slope_change_per_year"], errors="coerce"
    )
    data["slope_change_q"] = pd.to_numeric(data["slope_change_q"], errors="coerce")
    family_order = {"readability": 0, "syntax": 1}
    ordering = (
        data.groupby(["feature", "family"], as_index=False)
        .agg(mean_abs=("standardized_slope_change_per_year", lambda values: values.abs().mean()))
        .assign(family_order=lambda frame: frame["family"].map(family_order))
        .sort_values(["family_order", "mean_abs"], ascending=[True, False])
    )
    feature_order = ordering["feature"].tolist()
    values = data.pivot(index="feature", columns="dataset", values="standardized_slope_change_per_year").reindex(
        index=feature_order, columns=datasets
    )
    qs = data.pivot(index="feature", columns="dataset", values="slope_change_q").reindex(
        index=feature_order, columns=datasets
    )
    finite = values.to_numpy(dtype=float)
    limit = float(np.nanmax(np.abs(finite))) if np.isfinite(finite).any() else 1.0
    fig, ax = plt.subplots(figsize=(3.25, max(2.15, 0.205 * len(feature_order) + 0.45)))
    image = ax.imshow(values, aspect="auto", cmap=ORCHID_GREEN_DIVERGING_CMAP, vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(datasets)), [SHORT_DATASET_LABELS.get(dataset, dataset) for dataset in datasets])
    ax.set_yticks(range(len(feature_order)), [_manuscript_table_label(feature) for feature in feature_order])
    ax.tick_params(axis="x", labelsize=6.5, rotation=0, length=0)
    ax.tick_params(axis="y", labelsize=6.2, length=0)
    for row in range(len(feature_order)):
        for column in range(len(datasets)):
            value = values.iat[row, column]
            if np.isfinite(value):
                ax.text(
                    column, row, f"{value:.1f}{_significance_stars(qs.iat[row, column])}",
                    ha="center", va="center", fontsize=5.1, color="black",
                )
    families = ordering["family"].tolist()
    for index in range(1, len(families)):
        if families[index] != families[index - 1]:
            ax.axhline(index - 0.5, color="black", linewidth=0.75)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.025)
    plt.close(fig)

    legend_path = path.with_name(f"{path.stem}_legend{path.suffix}")
    legend_fig, legend_ax = plt.subplots(figsize=(2.25, 0.34))
    legend_ax.axis("off")
    colorbar = legend_fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=-limit, vmax=limit), cmap=ORCHID_GREEN_DIVERGING_CMAP),
        ax=legend_ax, orientation="horizontal", fraction=0.65, pad=0.0,
    )
    colorbar.set_label(r"Standardized $\Delta$ slope/year", fontsize=6)
    colorbar.ax.tick_params(labelsize=5.5, length=2)
    legend_fig.savefig(legend_path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.01)
    plt.close(legend_fig)


def save_all_feature_family_effects(
    stats: pd.DataFrame,
    datasets: list[str],
    output_dir: Path,
) -> None:
    """Write one four-corpus compact forest plot for every feature family."""
    datasets = [dataset for dataset in datasets if dataset.endswith("_abstracts")]
    abstract_stats = stats[stats["dataset"].isin(datasets)].copy()
    if abstract_stats.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for family in sorted(abstract_stats["family"].dropna().astype(str).unique()):
        family_data = abstract_stats[abstract_stats["family"].astype(str) == family].copy()
        complete_counts = family_data.groupby("feature")["dataset"].nunique()
        features = complete_counts[complete_counts == len(datasets)].index.tolist()
        features = [feature for feature in features if feature not in FAMILY_AGGREGATE_FEATURES]
        finite_counts = (
            family_data.assign(
                _finite=np.isfinite(
                    pd.to_numeric(
                        family_data["standardized_slope_change_per_year"],
                        errors="coerce",
                    )
                )
            )
            .groupby("feature")["_finite"]
            .sum()
        )
        features = [feature for feature in features if finite_counts.get(feature, 0) > 0]
        if not features:
            continue
        ranking = (
            family_data[family_data["feature"].isin(features)]
            .groupby("feature")["standardized_slope_change_per_year"]
            .apply(lambda values: values.abs().mean())
            .sort_values(ascending=False)
        )
        feature_order = ranking.index.tolist()
        labels = {feature: pretty_feature_label(feature) for feature in feature_order}
        slug = family.replace(" ", "_").replace("/", "_")
        save_feature_family_effects(
            family_data,
            datasets,
            output_dir / f"standardized_effects_{slug}.png",
            feature_order=feature_order,
            feature_labels=labels,
        )


def save_feature_family_effects(
    data: pd.DataFrame,
    datasets: list[str],
    path: Path,
    *,
    feature_order: list[str],
    feature_labels: dict[str, str],
) -> None:
    """Shared compact forest-plot grammar used for every feature family."""
    datasets = [dataset for dataset in datasets if dataset.endswith("_abstracts")]
    if data.empty or not datasets or not feature_order:
        return
    row_spacing = 0.58
    y_base = np.arange(len(feature_order), dtype=float) * row_spacing
    offsets = np.linspace(-0.17, 0.17, len(datasets))
    fig_height = max(0.82, 0.185 * len(feature_order) + 0.40)
    fig, ax = plt.subplots(figsize=(2.62, fig_height))
    for row, y_value in enumerate(y_base):
        if row % 2 == 0:
            ax.axhspan(y_value - row_spacing / 2, y_value + row_spacing / 2, color="0.94", zorder=0)
    for offset, dataset in zip(offsets, datasets, strict=True):
        subset = data[data["dataset"] == dataset].set_index("feature").reindex(feature_order)
        estimate = subset["standardized_slope_change_per_year"].to_numpy(dtype=float)
        low = subset["standardized_slope_change_per_year_ci_low"].to_numpy(dtype=float)
        high = subset["standardized_slope_change_per_year_ci_high"].to_numpy(dtype=float)
        ax.errorbar(
            estimate, y_base + offset, xerr=np.vstack([estimate - low, high - estimate]),
            fmt=DATASET_MARKERS[dataset], color=DATASET_COLORS[dataset],
            ecolor=DATASET_COLORS[dataset], markersize=2.15, linewidth=0.6,
            capsize=1.0, label=SHORT_DATASET_LABELS.get(dataset, dataset),
        )
    ax.axvline(0, color="0.25", linewidth=0.75)
    ax.set_yticks(y_base, [feature_labels.get(feature, pretty_feature_label(feature)) for feature in feature_order])
    ax.invert_yaxis()
    ax.set_xlabel(r"Standardized $\Delta$ slope/year", fontsize=6)
    ax.tick_params(axis="both", labelsize=5.5, length=2, pad=1.5)
    ax.grid(axis="x", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.01)
    plt.close(fig)
    save_marker_legend(datasets, path.with_name(f"{path.stem}_legend{path.suffix}"))


def save_determiner_context_trends(
    analysis_dir: Path,
    datasets: list[str],
    path: Path,
) -> None:
    """Two-panel cross-corpus summary of the determiner-context mechanism."""
    datasets = [dataset for dataset in datasets if dataset.endswith("_abstracts")]
    parts = []
    for dataset in datasets:
        source = analysis_dir / dataset / "additional_analysis" / "determiner_decomposition_yearly.csv"
        if not source.exists():
            continue
        frame = pd.read_csv(source)
        required = {"year", "det_pobj_per_1k_words", "prenominal_to_prepositional_ratio"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[list(required)].copy()
        frame["dataset"] = dataset
        parts.append(frame)
    if len(parts) != len(datasets):
        return
    data = pd.concat(parts, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(3.65, 1.62), sharex=True)
    panels = (
        ("det_pobj_per_1k_words", "det–pobj frequency"),
        ("prenominal_to_prepositional_ratio", "modifier ratio"),
    )
    for ax, (column, ylabel) in zip(axes, panels, strict=True):
        for dataset in datasets:
            series = data[data["dataset"] == dataset].sort_values("year")
            ax.plot(
                series["year"], series[column], color=DATASET_COLORS[dataset],
                marker=DATASET_MARKERS[dataset], markersize=2.4, linewidth=0.85,
            )
        ax.axvline(2022.92, color="0.25", linestyle="--", linewidth=0.65)
        ax.set_ylabel(ylabel, fontsize=6)
        ax.set_xlabel("Year", fontsize=6)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.tick_params(axis="both", labelsize=5.5, length=2, pad=1.5)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.22)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35, w_pad=0.8)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.01)
    plt.close(fig)
    save_marker_legend(datasets, path.with_name(f"{path.stem}_legend{path.suffix}"))


def save_correlation_heatmap(pairwise: pd.DataFrame, datasets: list[str], path: Path) -> None:
    if pairwise.empty:
        return
    datasets = [dataset for dataset in datasets if dataset in DEFAULT_PRIMARY_DATASETS]
    if len(datasets) < 2:
        return
    save_pairwise_correlation_heatmap(
        pairwise, datasets, path, colorbar_label="Pearson r"
    )


def save_pairwise_correlation_heatmap(
    pairwise: pd.DataFrame,
    datasets: list[str],
    path: Path,
    *,
    colorbar_label: str,
) -> None:
    if pairwise.empty:
        return
    matrix = pd.DataFrame(np.eye(len(datasets)), index=datasets, columns=datasets)
    for row in pairwise.itertuples(index=False):
        matrix.loc[row.dataset_a, row.dataset_b] = row.pearson_r
        matrix.loc[row.dataset_b, row.dataset_a] = row.pearson_r
    save_heatmap(
        matrix,
        path,
        vmin=-1,
        vmax=1,
        cmap=COMPARISON_HEATMAP_CMAP,
        fmt=".2f",
        colorbar_label="Pearson r",
    )


def save_group_correlation_heatmap(group_corr: pd.DataFrame, path: Path) -> None:
    if group_corr.empty:
        return
    summary = group_corr.groupby("family", as_index=True)["pearson_r"].mean().sort_values(ascending=False)
    fig_height = max(1.8, 0.22 * len(summary) + 0.5)
    fig, ax = plt.subplots(figsize=(4, fig_height))
    colors = [INCREASE_COLOR if value >= 0 else DECREASE_COLOR for value in summary]
    ordered = summary.iloc[::-1]
    bars = ax.barh(ordered.index, ordered, color=colors[::-1], edgecolor="0.25", linewidth=0.35)
    for bar, value in zip(bars, ordered, strict=False):
        bar.set_hatch(sign_hatch(value))
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel("mean Pearson r", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_pairwise_scatter_grid(matrix: pd.DataFrame, datasets: list[str], path: Path) -> None:
    if len(datasets) < 2:
        return
    pairs = comparison_scatter_pairs(datasets)
    cols = min(3, len(pairs))
    rows = int(np.ceil(len(pairs) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.35, max(1.35, 0.98 * rows)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    pair_columns = sorted({dataset for pair in pairs for dataset in pair})
    max_abs = max(float(matrix[pair_columns].abs().max().max()), 1.0)
    limit = max_abs * 1.1
    for ax, pair in zip(axes.ravel(), pairs, strict=False):
        left, right = pair
        data = matrix[[left, right]].dropna()
        r, p_value = safe_corr_p(data[left], data[right]) if len(data) > 1 else (np.nan, np.nan)
        ax.scatter(data[left], data[right], s=8, alpha=0.7, color=INCREASE_COLOR)
        ax.axhline(0, color="0.75", linewidth=0.6)
        ax.axvline(0, color="0.75", linewidth=0.6)
        ax.plot([-limit, limit], [-limit, limit], color=DECREASE_COLOR, linewidth=0.6)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.04,
            0.94,
            f"{label(left)}-{label(right)}\nr={r:.2f}{p_stars(p_value)}",
            transform=ax.transAxes,
            va="top",
            fontsize=5,
        )
        ax.tick_params(axis="both", labelsize=5, length=2, pad=1)
        ax.grid(alpha=0.25)
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    fig.text(0.52, 0.015, "std. slope", ha="center", va="bottom", fontsize=6)
    fig.text(0.015, 0.53, "std. slope", ha="left", va="center", rotation="vertical", fontsize=6)
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.135, top=0.995, wspace=0.08, hspace=0.12)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def comparison_scatter_pairs(datasets: list[str]) -> list[tuple[str, str]]:
    preferred_pairs = [
        ("arxiv_ai_abstracts", "arxiv_qbio_abstracts"),
        ("arxiv_ai_abstracts", "arxiv_stat_abstracts"),
        ("arxiv_ai_abstracts", "medarxiv_abstracts"),
        ("arxiv_ai_titles", "arxiv_qbio_titles"),
        ("arxiv_ai_titles", "arxiv_stat_titles"),
        ("arxiv_ai_titles", "medarxiv_titles"),
    ]
    available = set(datasets)
    pairs = [pair for pair in preferred_pairs if pair[0] in available and pair[1] in available]
    return pairs or list(combinations(datasets, 2))[:6]


def save_syntax_dependency_bars(syntax: pd.DataFrame, datasets: list[str], path: Path) -> None:
    if syntax.empty:
        return
    matrix = syntax.pivot_table(
        index=["feature", "family"],
        columns="dataset",
        values="standardized_slope_change_per_year",
        aggfunc="first",
    ).reset_index()
    value_cols = [dataset for dataset in datasets if dataset in matrix.columns]
    if not value_cols:
        return
    matrix["_rank"] = matrix[value_cols].abs().mean(axis=1)
    matrix = matrix.sort_values("_rank", ascending=False).head(12).iloc[::-1]
    save_grouped_horizontal_bars(
        matrix,
        value_cols,
        path,
        xlabel="std. slope/year",
    )


def save_top_disagreement_bars(disagreements: pd.DataFrame, datasets: list[str], path: Path) -> None:
    value_cols = [dataset for dataset in datasets if dataset in disagreements.columns]
    data = disagreements.head(12).iloc[::-1]
    save_grouped_horizontal_bars(
        data,
        value_cols,
        path,
        xlabel="std. slope/year",
    )


def save_dependency_role_bars(role_slopes: pd.DataFrame, datasets: list[str], path: Path) -> None:
    matrix = role_slopes.pivot_table(
        index="dependency_role",
        columns="dataset",
        values="slope_change_per_year",
        aggfunc="first",
    ).reset_index()
    value_cols = [dataset for dataset in datasets if dataset in matrix.columns]
    if not value_cols:
        return
    matrix["feature"] = matrix["dependency_role"]
    matrix["_rank"] = matrix[value_cols].abs().mean(axis=1)
    matrix = matrix.sort_values("_rank", ascending=False).head(12).iloc[::-1]
    save_grouped_horizontal_bars(
        matrix,
        value_cols,
        path,
        xlabel="role-share slope/year",
    )


def save_dependency_bigram_bars(bigram_slopes: pd.DataFrame, datasets: list[str], path: Path) -> None:
    matrix = bigram_slopes.pivot_table(
        index="dependency_bigram",
        columns="dataset",
        values="slope_change_per_year",
        aggfunc="first",
    ).reset_index()
    value_cols = [dataset for dataset in datasets if dataset in matrix.columns]
    if not value_cols:
        return
    matrix["feature"] = matrix["dependency_bigram"]
    matrix["_rank"] = matrix[value_cols].abs().mean(axis=1)
    matrix = matrix.sort_values("_rank", ascending=False).head(12).iloc[::-1]
    save_grouped_horizontal_bars(
        matrix,
        value_cols,
        path,
        xlabel=r"$\Delta$ edge-proportion slope/year",
    )


def save_grouped_horizontal_bars(
    data: pd.DataFrame,
    value_cols: list[str],
    path: Path,
    *,
    xlabel: str,
) -> None:
    if data.empty or not value_cols:
        return
    y = np.arange(len(data))
    height = min(0.8 / len(value_cols), 0.22)
    fig_height = max(2.2, 0.2 * len(data) + 0.3)
    fig, ax = plt.subplots(figsize=(3, fig_height))
    for row in range(len(data)):
        if row % 2 == 0:
            ax.axhspan(row - 0.5, row + 0.5, color="0.94", zorder=0)
    offsets = np.linspace(-height * (len(value_cols) - 1) / 2, height * (len(value_cols) - 1) / 2, len(value_cols))
    for offset, column in zip(offsets, value_cols, strict=True):
        bars = ax.barh(
            y + offset,
            data[column],
            height=height,
            color=DATASET_COLORS.get(column, "0.5"),
            hatch=DATASET_HATCHES.get(column, ""),
            edgecolor="0.25",
            linewidth=0.35,
            label=label(column),
        )
    labels = [short_feature_label(row.feature, getattr(row, "family", "")) for row in data.itertuples(index=False)]
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=6)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    save_separate_legend(value_cols, path.with_name(f"{path.stem}_legend{path.suffix}"))


def save_separate_legend(value_cols: list[str], path: Path) -> None:
    handles = [
        Patch(
            facecolor=DATASET_COLORS.get(column, "0.5"),
            edgecolor="0.25",
            hatch=DATASET_HATCHES.get(column, ""),
            label=label(column),
        )
        for column in value_cols
    ]
    if not handles:
        return
    ncol = min(4, len(handles))
    rows = int(np.ceil(len(handles) / ncol))
    fig, ax = plt.subplots(figsize=(3, 0.2 * rows + 0.15))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        ncol=ncol,
        fontsize=6,
        frameon=False,
        handlelength=1.2,
        columnspacing=0.9,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def save_marker_legend(value_cols: list[str], path: Path) -> None:
    handles = [
        Line2D(
            [0], [0], marker=DATASET_MARKERS[column], color=DATASET_COLORS[column],
            linestyle="none", label=label(column), markersize=4,
        )
        for column in value_cols
    ]
    fig, ax = plt.subplots(figsize=(2.8, 0.28))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=len(handles), fontsize=6,
              frameon=False, handletextpad=0.35, columnspacing=0.8)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE, pad_inches=0.01)
    plt.close(fig)


def save_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    *,
    vmin: float,
    vmax: float,
    cmap: str | Colormap,
    fmt: str,
    colorbar_label: str | None = None,
) -> None:
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(2.62, 2.18))
    image = ax.imshow(
        matrix.to_numpy(dtype=float), vmin=vmin, vmax=vmax, cmap=cmap,
        aspect="equal", interpolation="none",
    )
    ax.set_xticks(range(n), [label(value) for value in matrix.columns], rotation=35, ha="right")
    ax.set_yticks(range(n), [label(value) for value in matrix.index])
    ax.tick_params(axis="both", labelsize=7.2, length=2.5, pad=2)
    for i in range(n):
        for j in range(n):
            value = matrix.iat[i, j]
            if pd.notna(value):
                if i == j:
                    text = "1"
                else:
                    text = compact_float(value, fmt)
                ax.text(j, i, text, ha="center", va="center", fontsize=8.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.045, shrink=0.84)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    fig.subplots_adjust(left=0.22, right=0.88, bottom=0.24, top=0.985)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.02)
    plt.close(fig)


def safe_corr_p(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    left = pd.to_numeric(left, errors="coerce")
    right = pd.to_numeric(right, errors="coerce")
    valid = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(valid) < 2:
        return float("nan"), float("nan")
    if valid["left"].std(ddof=0) == 0 or valid["right"].std(ddof=0) == 0:
        return float("nan"), float("nan")
    result = stats.pearsonr(valid["left"], valid["right"])
    return float(result.statistic), float(result.pvalue)


def p_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def compact_float(value: float, fmt: str) -> str:
    text = format(float(value), fmt)
    if text.startswith("0"):
        return text[1:]
    if text.startswith("-0"):
        return "-" + text[2:]
    return text


def split_dependency_bigram(bigram: str) -> tuple[str, str]:
    if "->" not in bigram:
        return bigram, ""
    head, child = bigram.split("->", 1)
    return head, child


def short_feature_label(feature: str, family: str | None = None) -> str:
    feature = str(feature)
    mapped = pretty_feature_label(feature)
    if mapped != feature:
        return mapped
    return feature.replace("_per_1k_words", "").replace("_", " ")


def label(dataset: str) -> str:
    return SHORT_DATASET_LABELS.get(dataset, dataset.replace("arxiv_", "").replace("_", " "))


if __name__ == "__main__":
    main()
