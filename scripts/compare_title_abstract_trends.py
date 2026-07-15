from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from not_an_llm.analysis.label_map import pretty_feature_label


DEFAULT_PAIRS = {
    "arxiv_ai": ("arxiv_ai_abstracts", "arxiv_ai_titles"),
    "arxiv_qbio": ("arxiv_qbio_abstracts", "arxiv_qbio_titles"),
    "arxiv_stat": ("arxiv_stat_abstracts", "arxiv_stat_titles"),
    "medarxiv": ("medarxiv_abstracts", "medarxiv_titles"),
}

CORPUS_LABELS = {
    "arxiv_ai": "arXiv AI",
    "arxiv_qbio": "arXiv q-bio",
    "arxiv_stat": "arXiv Statistics",
    "medarxiv": "medRxiv",
}

FAMILY_LABELS = {
    "adjectives": "adj.",
    "causal_markers": "causal",
    "contrast_markers": "contrast",
    "discourse_marker_totals": "disc.",
    "emphasis_markers": "emph.",
    "marker_words": "marker",
    "phrases": "phr.",
    "punctuation": "punct.",
    "readability": "read.",
    "sequential_markers": "seq.",
    "summary_markers": "summ.",
    "syntax": "syntax",
    "verbs": "verb",
}

SYNTAX_FEATURE_HINTS = (
    "dependency",
    "clause",
    "sentence_depth",
    "coordination",
    "list_of_three",
    "avg_words_per_sentence",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare title-only and abstract-only ITS trends."
    )
    parser.add_argument("--analysis-dir", default="data/analysis")
    parser.add_argument(
        "--analysis-output-dir",
        default="data/analysis/title_abstract_comparison",
        help="Directory for comparison CSV tables.",
    )
    parser.add_argument(
        "--visuals-output-dir",
        default="data/visuals/title_abstract_comparison",
        help="Directory for all comparison figures.",
    )
    parser.add_argument(
        "--output-dir",
        dest="legacy_analysis_output_dir",
        default=None,
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir)
    analysis_output_dir = Path(args.legacy_analysis_output_dir or args.analysis_output_dir)
    visuals_output_dir = Path(args.visuals_output_dir)
    analysis_output_dir.mkdir(parents=True, exist_ok=True)
    visuals_output_dir.mkdir(parents=True, exist_ok=True)

    comparison = build_comparison(analysis_dir)
    comparison.to_csv(analysis_output_dir / "title_abstract_its_comparison.csv", index=False)

    dependency = comparison[comparison["is_dependency_or_syntax"]].copy()
    dependency.to_csv(analysis_output_dir / "dependency_title_abstract_its_comparison.csv", index=False)

    summary = summarize(comparison)
    summary.to_csv(analysis_output_dir / "title_abstract_comparison_summary.csv", index=False)

    save_scatter(
        comparison,
        visuals_output_dir / "title_abstract_standardized_slope_scatter.png",
        title="Title vs abstract ITS slope changes",
    )
    save_scatter(
        dependency,
        visuals_output_dir / "dependency_title_abstract_standardized_slope_scatter.png",
        title="Dependency/syntax ITS slope changes",
    )
    save_dependency_bars(
        dependency,
        visuals_output_dir / "dependency_title_abstract_standardized_slope_bars.png",
    )
    save_top_difference_bars(
        comparison,
        visuals_output_dir / "top_title_abstract_standardized_slope_differences.png",
    )

    print(f"Saved comparison tables to {analysis_output_dir}")
    print(f"Saved comparison figures to {visuals_output_dir}")
    print(summary.to_string(index=False))


def build_comparison(analysis_dir: Path) -> pd.DataFrame:
    rows = []
    for corpus, (abstract_name, title_name) in DEFAULT_PAIRS.items():
        abstract = load_its(analysis_dir / abstract_name / "its_stats.csv")
        title = load_its(analysis_dir / title_name / "its_stats.csv")

        merged = abstract.merge(
            title,
            on=["feature", "family"],
            suffixes=("_abstract", "_title"),
        )
        merged.insert(0, "corpus", corpus)
        rows.append(merged)

    comparison = pd.concat(rows, ignore_index=True)
    comparison["standardized_slope_difference"] = (
        comparison["standardized_slope_change_per_year_title"]
        - comparison["standardized_slope_change_per_year_abstract"]
    )
    comparison["same_direction"] = (
        comparison["standardized_slope_change_per_year_title"].fillna(0)
        * comparison["standardized_slope_change_per_year_abstract"].fillna(0)
        > 0
    )
    comparison["is_dependency_or_syntax"] = comparison.apply(is_dependency_or_syntax, axis=1)
    return comparison


def load_its(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    columns = [
        "feature",
        "family",
        "pre_mean",
        "post_mean",
        "slope_change_per_year",
        "slope_change_per_year_ci_low",
        "slope_change_per_year_ci_high",
        "standardized_slope_change_per_year",
        "standardized_slope_change_per_year_ci_low",
        "standardized_slope_change_per_year_ci_high",
        "slope_change_q",
    ]
    return pd.read_csv(path, usecols=lambda column: column in columns)


def is_dependency_or_syntax(row: pd.Series) -> bool:
    feature = str(row["feature"])
    family = str(row["family"])
    return family == "syntax" or any(hint in feature for hint in SYNTAX_FEATURE_HINTS)


def summarize(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for corpus, group in comparison.groupby("corpus", sort=False):
        rows.append(summary_row(corpus, "all_common_features", group))
        rows.append(
            summary_row(
                corpus,
                "dependency_or_syntax",
                group[group["is_dependency_or_syntax"]],
            )
        )
    return pd.DataFrame(rows)


def summary_row(corpus: str, feature_set: str, group: pd.DataFrame) -> dict[str, object]:
    abstract = group["standardized_slope_change_per_year_abstract"]
    title = group["standardized_slope_change_per_year_title"]
    return {
        "corpus": corpus,
        "feature_set": feature_set,
        "n_features": int(len(group)),
        "pearson_r_standardized_slope": float(abstract.corr(title)) if len(group) > 1 else pd.NA,
        "same_direction_share": float(group["same_direction"].mean()) if len(group) else pd.NA,
        "median_abs_standardized_difference": float(
            group["standardized_slope_difference"].abs().median()
        )
        if len(group)
        else pd.NA,
        "title_larger_abs_share": float(
            (
                title.abs()
                > abstract.abs()
            ).mean()
        )
        if len(group)
        else pd.NA,
    }


def save_scatter(data: pd.DataFrame, path: Path, *, title: str) -> None:
    if data.empty:
        return

    corpora = list(DEFAULT_PAIRS)
    fig, axes = plt.subplots(1, len(corpora), figsize=(10.5, 3.2), sharex=True, sharey=True)
    if len(corpora) == 1:
        axes = [axes]

    max_abs = max(
        data["standardized_slope_change_per_year_abstract"].abs().max(),
        data["standardized_slope_change_per_year_title"].abs().max(),
    )
    limit = max(1.0, float(max_abs) * 1.1)

    colors = {
        "arxiv_ai": "#1f77b4",
        "arxiv_qbio": "#ff7f0e",
        "arxiv_stat": "#9467bd",
        "medarxiv": "#2ca02c",
    }

    for ax, corpus in zip(axes, corpora, strict=True):
        subset = data[data["corpus"] == corpus]
        ax.scatter(
            subset["standardized_slope_change_per_year_abstract"],
            subset["standardized_slope_change_per_year_title"],
            color=colors[corpus],
            label=CORPUS_LABELS.get(corpus, corpus),
            s=24,
            alpha=0.75,
        )
        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.axvline(0, color="0.7", linewidth=0.8)
        ax.plot([-limit, limit], [-limit, limit], color="0.35", linewidth=0.8)
        ax.set_title(CORPUS_LABELS.get(corpus, corpus))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("Title standardized slope change")
    for ax in axes:
        ax.set_xlabel("Abstract standardized slope change")
        ax.legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_dependency_bars(data: pd.DataFrame, path: Path) -> None:
    if data.empty:
        return

    corpora = list(DEFAULT_PAIRS)
    features = (
        data.groupby("feature")[
            [
                "standardized_slope_change_per_year_abstract",
                "standardized_slope_change_per_year_title",
            ]
        ]
        .mean()
        .abs()
        .sum(axis=1)
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )
    plot_data = data[data["feature"].isin(features)].copy()
    plot_data["feature"] = pd.Categorical(plot_data["feature"], categories=features[::-1], ordered=True)

    fig, axes = plt.subplots(len(corpora), 1, figsize=(3.45, 8.0), sharex=True)
    for ax, corpus in zip(axes, corpora, strict=True):
        subset = plot_data[plot_data["corpus"] == corpus].sort_values("feature")
        _draw_pair_bars(ax, subset, include_family=False)
        ax.set_title(CORPUS_LABELS.get(corpus, corpus), fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.04),
        frameon=True,
    )
    fig.supxlabel(r"Standardized $\Delta$ slope/year", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    for corpus in corpora:
        subset = plot_data[plot_data["corpus"] == corpus].sort_values("feature")
        save_single_bar_panel(
            subset,
            path.with_name(f"{path.stem}_{corpus}.png"),
            title=f"{CORPUS_LABELS.get(corpus, corpus)} syntax",
            include_family=False,
        )


def save_top_difference_bars(data: pd.DataFrame, path: Path, *, top_n: int = 10) -> None:
    if data.empty:
        return

    corpora = list(DEFAULT_PAIRS)
    fig, axes = plt.subplots(len(corpora), 1, figsize=(3.45, 8.0), sharex=False)

    for ax, corpus in zip(axes, corpora, strict=True):
        subset = data[data["corpus"] == corpus].copy()
        subset = subset.sort_values(
            "standardized_slope_difference",
            key=lambda series: series.abs(),
            ascending=False,
        ).head(top_n)
        subset = subset.iloc[::-1]
        _draw_pair_bars(ax, subset, include_family=True)
        ax.set_title(CORPUS_LABELS.get(corpus, corpus), fontsize=10)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.04),
        frameon=True,
    )
    fig.supxlabel(r"Standardized $\Delta$ slope/year", fontsize=9)
    fig.suptitle("Largest title/abstract differences", y=1.01, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    for corpus in corpora:
        subset = data[data["corpus"] == corpus].copy()
        subset = subset.sort_values(
            "standardized_slope_difference",
            key=lambda series: series.abs(),
            ascending=False,
        ).head(top_n)
        subset = subset.iloc[::-1]
        save_single_bar_panel(
            subset,
            path.with_name(f"{path.stem}_{corpus}.png"),
            title=CORPUS_LABELS.get(corpus, corpus),
            include_family=True,
        )

def save_single_bar_panel(
    data: pd.DataFrame,
    path: Path,
    *,
    title: str,
    include_family: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 3.35))
    _draw_pair_bars(ax, data, include_family=include_family)
    ax.set_title(title, fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize=7,
        frameon=True,
    )
    ax.set_xlabel(r"Standardized $\Delta$ slope/year", fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _draw_pair_bars(ax, data: pd.DataFrame, *, include_family: bool) -> None:
    abstract_color = "#1f77b4"
    title_color = "#ff7f0e"
    y = range(len(data))
    ax.barh(
        [i - 0.18 for i in y],
        data["standardized_slope_change_per_year_abstract"],
        height=0.34,
        color=abstract_color,
        label="abstract",
    )
    ax.barh(
        [i + 0.18 for i in y],
        data["standardized_slope_change_per_year_title"],
        height=0.34,
        color=title_color,
        label="title",
    )
    labels = [
        short_feature_label(row.feature, row.family, include_family=include_family)
        for row in data.itertuples(index=False)
    ]
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_yticks(list(y), labels)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="x", alpha=0.25)


def short_feature_label(feature: str, family: str, *, include_family: bool) -> str:
    feature = str(feature)
    family = str(family)
    label = pretty_feature_label(feature).replace("`", "")
    if include_family:
        aggregate_labels = {
            "marker_words_total_per_1k_words": "marker words",
            "marker_verbs_total_per_1k_words": "marker verbs",
            "marker_adjectives_total_per_1k_words": "marker adj.",
            "marker_phrases_total_per_1k_words": "marker phr.",
        }
        if feature in aggregate_labels:
            return aggregate_labels[feature]
        family_label = FAMILY_LABELS.get(family, family.replace("_", " "))
        return f"{label} [{family_label}]"
    return label


if __name__ == "__main__":
    main()
