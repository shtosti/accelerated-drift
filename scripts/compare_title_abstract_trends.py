from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from not_an_llm.analysis.label_map import pretty_feature_label
from not_an_llm.analysis.visual_style import (
    DATASET_COLORS,
    DATASET_MARKERS,
    GREEN,
    HATCHES,
    ORCHID,
    WHITE,
)
from compare_corpus_trends import MANUSCRIPT_TABLE_GROUPS, _manuscript_table_label


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
    save_title_abstract_direction_heatmap(
        comparison,
        visuals_output_dir / "title_abstract_direction_alignment.png",
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
    title = pd.to_numeric(
        comparison["standardized_slope_change_per_year_title"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    abstract = pd.to_numeric(
        comparison["standardized_slope_change_per_year_abstract"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    comparison["standardized_slope_change_per_year_title"] = title
    comparison["standardized_slope_change_per_year_abstract"] = abstract
    complete = title.notna() & abstract.notna()
    comparison["complete_standardized_pair"] = complete
    comparison["same_direction"] = pd.Series(pd.NA, index=comparison.index, dtype="boolean")
    comparison.loc[complete, "same_direction"] = (
        title.loc[complete] * abstract.loc[complete] > 0
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
    abstract_all = group["standardized_slope_change_per_year_abstract"]
    title_all = group["standardized_slope_change_per_year_title"]
    complete = group[abstract_all.notna() & title_all.notna()].copy()
    abstract = complete["standardized_slope_change_per_year_abstract"]
    title = complete["standardized_slope_change_per_year_title"]
    return {
        "corpus": corpus,
        "feature_set": feature_set,
        "n_configured_features": int(len(group)),
        "n_complete_pairs": int(len(complete)),
        "n_inestimable_abstract": int(abstract_all.isna().sum()),
        "n_inestimable_title": int(title_all.isna().sum()),
        "pearson_r_standardized_slope": float(abstract.corr(title)) if len(complete) > 1 else pd.NA,
        "same_direction_share": float(complete["same_direction"].mean()) if len(complete) else pd.NA,
        "median_abs_standardized_difference": float(
            complete["standardized_slope_difference"].abs().median()
        )
        if len(complete)
        else pd.NA,
        "title_larger_abs_share": float(
            (title.abs() > abstract.abs()).mean()
        )
        if len(complete)
        else pd.NA,
    }


TITLE_COMPARISON_GROUPS = {
    "Lexical markers": {"marker_words"},
    "Punctuation": {"punctuation"},
    "Readability": {"readability"},
    "Syntax": {"syntax"},
}


def select_dumbbell_features(comparison: pd.DataFrame, *, per_group: int = 3) -> list[str]:
    """Select three shared individual features from each comparison group."""
    family_to_group = {
        family: group
        for group, families in TITLE_COMPARISON_GROUPS.items()
        for family in families
    }
    data = comparison.copy()
    data["manuscript_group"] = data["family"].map(family_to_group)
    data = data[
        data["manuscript_group"].notna()
        & ~data["feature"].astype(str).str.endswith("_total_per_1k_words")
    ]
    complete = (
        data.groupby("feature")["complete_standardized_pair"].sum()
        == len(DEFAULT_PAIRS)
    )
    data = data[data["feature"].isin(complete[complete].index)].copy()
    ranking = (
        data.groupby(["feature", "manuscript_group"], as_index=False)
        .agg(
            mean_abs_abstract=("standardized_slope_change_per_year_abstract", lambda x: x.abs().mean()),
        )
    )
    selected: list[str] = []
    for group in TITLE_COMPARISON_GROUPS:
        candidates = ranking[ranking["manuscript_group"] == group].sort_values(
            ["mean_abs_abstract", "feature"], ascending=[False, True]
        )
        profiles: list[tuple[float, ...]] = []
        for feature in candidates["feature"].astype(str):
            feature_rows = data[data["feature"] == feature].set_index("corpus").reindex(DEFAULT_PAIRS)
            profile = tuple(
                np.round(
                    feature_rows[[
                        "standardized_slope_change_per_year_abstract",
                        "standardized_slope_change_per_year_title",
                    ]].to_numpy(dtype=float).ravel(),
                    10,
                )
            )
            if profile in profiles:
                continue
            profiles.append(profile)
            selected.append(feature)
            if len(profiles) == per_group:
                break
    return selected


def save_title_abstract_dumbbells(comparison: pd.DataFrame, path: Path) -> None:
    features = select_dumbbell_features(comparison)
    if not features:
        return
    data = comparison[comparison["feature"].isin(features)].copy()
    family_to_group = {
        family: group
        for group, families in TITLE_COMPARISON_GROUPS.items()
        for family in families
    }
    data["manuscript_group"] = data["family"].map(family_to_group)
    order = []
    for group in TITLE_COMPARISON_GROUPS:
        order.extend(
            feature for feature in features
            if family_to_group.get(str(data.loc[data["feature"] == feature, "family"].iloc[0])) == group
        )
    y = np.arange(len(order), dtype=float)
    values = data[[
        "standardized_slope_change_per_year_abstract",
        "standardized_slope_change_per_year_title",
    ]].to_numpy(dtype=float)
    limit = max(1.0, float(np.nanmax(np.abs(values))) * 1.08)
    fig, axes = plt.subplots(2, 2, figsize=(5.7, 4.3), sharex=True, sharey=True)
    for ax, corpus in zip(axes.flat, DEFAULT_PAIRS, strict=True):
        color = DATASET_COLORS[f"{corpus}_abstracts"]
        subset = data[data["corpus"] == corpus].set_index("feature").reindex(order)
        abstract = subset["standardized_slope_change_per_year_abstract"].to_numpy(dtype=float)
        title = subset["standardized_slope_change_per_year_title"].to_numpy(dtype=float)
        for row, (left, right) in enumerate(zip(abstract, title, strict=True)):
            ax.plot([left, right], [row, row], color=color, linewidth=0.8, alpha=0.65, zorder=1)
        ax.scatter(abstract, y, marker="o", s=19, color=color, edgecolor="black", linewidth=0.25, zorder=3)
        ax.scatter(title, y, marker="s", s=18, facecolor=WHITE, edgecolor=color, linewidth=0.9, zorder=3)
        ax.axvline(0, color="black", linewidth=0.65)
        ax.set_title(CORPUS_LABELS[corpus].removeprefix("arXiv "), fontsize=8.5, color="black", pad=2)
        ax.set_xlim(-limit, limit)
        ax.set_yticks(y, [_manuscript_table_label(feature) for feature in order])
        ax.invert_yaxis()
        ax.tick_params(axis="both", labelsize=5.8)
        ax.grid(axis="x", alpha=0.22, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.55)
    fig.supxlabel(r"Standardized $\Delta$ slope/year", fontsize=7)
    fig.subplots_adjust(left=0.20, right=0.995, bottom=0.10, top=0.96, wspace=0.08, hspace=0.16)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.025)
    plt.close(fig)

    legend_path = path.with_name(f"{path.stem}_legend{path.suffix}")
    legend_fig, legend_ax = plt.subplots(figsize=(2.0, 0.27))
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="black", markerfacecolor="black", linestyle="none", markersize=3.8, label="Abstract"),
            Line2D([0], [0], marker="s", color="black", markerfacecolor=WHITE, linestyle="none", markersize=3.8, label="Title"),
        ],
        loc="center", ncol=2, frameon=False, fontsize=6.5,
        handletextpad=0.4, columnspacing=1.0,
    )
    legend_fig.savefig(legend_path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.01)
    plt.close(legend_fig)

    selection = data[["feature", "family", "manuscript_group"]].drop_duplicates("feature")
    selection.to_csv(path.with_name("title_abstract_selected_dumbbells_features.csv"), index=False)


def save_title_abstract_direction_heatmap(comparison: pd.DataFrame, path: Path) -> None:
    """Show abstract/title direction combinations for the selected shared features."""
    from matplotlib.colors import ListedColormap

    features = select_dumbbell_features(comparison)
    data = comparison[comparison["feature"].isin(features)].copy()
    family_to_group = {
        family: group
        for group, families in TITLE_COMPARISON_GROUPS.items()
        for family in families
    }
    data["manuscript_group"] = data["family"].map(family_to_group)
    order = []
    for group in TITLE_COMPARISON_GROUPS:
        order.extend(
            feature for feature in features
            if family_to_group.get(str(data.loc[data["feature"] == feature, "family"].iloc[0])) == group
        )
    corpora = list(DEFAULT_PAIRS)
    codes = np.full((len(order), len(corpora)), np.nan)
    symbols = np.full((len(order), len(corpora)), "", dtype=object)
    for row, feature in enumerate(order):
        for column, corpus in enumerate(corpora):
            match = data[(data["feature"] == feature) & (data["corpus"] == corpus)]
            if match.empty:
                continue
            abstract = float(match.iloc[0]["standardized_slope_change_per_year_abstract"])
            title = float(match.iloc[0]["standardized_slope_change_per_year_title"])
            if abstract > 0 and title > 0:
                codes[row, column], symbols[row, column] = 0, "↑↑"
            elif abstract < 0 and title < 0:
                codes[row, column], symbols[row, column] = 1, "↓↓"
            elif abstract > 0 and title < 0:
                codes[row, column], symbols[row, column] = 2, "↑↓"
            elif abstract < 0 and title > 0:
                codes[row, column], symbols[row, column] = 3, "↓↑"
    cmap = ListedColormap(["#B9DCC1", "#D6AFD2", "#F3C36B", "#56B4E9"])
    fig, ax = plt.subplots(figsize=(2.75, 2.35))
    ax.imshow(codes, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5)
    ax.set_xticks(range(len(corpora)), [CORPUS_LABELS[c].removeprefix("arXiv ") for c in corpora])
    ax.set_yticks(range(len(order)), [_manuscript_table_label(feature) for feature in order])
    ax.tick_params(axis="x", labelsize=6, length=0)
    ax.tick_params(axis="y", labelsize=6, length=0)
    for row in range(len(order)):
        for column in range(len(corpora)):
            ax.text(column, row, symbols[row, column], ha="center", va="center", fontsize=7.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.025)
    plt.close(fig)

    legend_path = path.with_name(f"{path.stem}_legend{path.suffix}")
    handles = [
        Patch(facecolor=color, edgecolor="0.25", linewidth=0.35, label=label)
        for color, label in zip(cmap.colors, ["↑↑ both positive", "↓↓ both negative", "↑↓ abstract+/title−", "↓↑ abstract−/title+"], strict=True)
    ]
    legend_fig, legend_ax = plt.subplots(figsize=(2.55, 0.58))
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center", ncol=2, frameon=False, fontsize=6,
                     handlelength=1.0, handletextpad=0.4, columnspacing=0.8, labelspacing=0.35)
    legend_fig.savefig(legend_path, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.01)
    plt.close(legend_fig)


def save_scatter(data: pd.DataFrame, path: Path, *, title: str) -> None:
    if data.empty:
        return

    corpora = list(DEFAULT_PAIRS)
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 5.5), sharex=True, sharey=True)
    axes = axes.ravel()
    if len(corpora) == 1:
        axes = [axes]

    max_abs = max(
        data["standardized_slope_change_per_year_abstract"].abs().max(),
        data["standardized_slope_change_per_year_title"].abs().max(),
    )
    limit = max(1.0, float(max_abs) * 1.1)

    for ax, corpus in zip(axes, corpora, strict=True):
        subset = data[
            (data["corpus"] == corpus)
            & data["standardized_slope_change_per_year_abstract"].notna()
            & data["standardized_slope_change_per_year_title"].notna()
        ]
        ax.scatter(
            subset["standardized_slope_change_per_year_abstract"],
            subset["standardized_slope_change_per_year_title"],
            color=DATASET_COLORS[f"{corpus}_abstracts"],
            marker=DATASET_MARKERS[f"{corpus}_abstracts"],
            label=CORPUS_LABELS.get(corpus, corpus),
            s=24,
            alpha=0.75,
        )
        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.axvline(0, color="0.7", linewidth=0.8)
        ax.plot([-limit, limit], [-limit, limit], color="0.35", linestyle="--", linewidth=0.8)
        ax.set_title(CORPUS_LABELS.get(corpus, corpus), fontsize=9)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.grid(alpha=0.25)

    fig.supylabel("Title standardized slope change/year", x=0.02, fontsize=8)
    fig.supxlabel("Abstract standardized slope change/year", y=0.02, fontsize=8)
    fig.tight_layout(rect=(0.04, 0.05, 1, 1))
    fig.savefig(path.resolve(), dpi=200, bbox_inches="tight")
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
        ax.set_title(CORPUS_LABELS.get(corpus, corpus), fontsize=9)

    fig.supxlabel(r"Standardized $\Delta$ slope/year", fontsize=9)
    fig.tight_layout()
    fig.savefig(path.resolve(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    save_scope_legend(path.with_name(f"{path.stem}_legend{path.suffix}"))

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
        ax.set_title(CORPUS_LABELS.get(corpus, corpus), fontsize=9)

    fig.supxlabel(r"Standardized $\Delta$ slope/year", fontsize=9)
    fig.tight_layout()
    fig.savefig(path.resolve(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    save_scope_legend(path.with_name(f"{path.stem}_legend{path.suffix}"))

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
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"Standardized $\Delta$ slope/year", fontsize=8)
    fig.tight_layout()
    fig.savefig(path.resolve(), dpi=200, bbox_inches="tight")
    plt.close(fig)
    save_scope_legend(path.with_name(f"{path.stem}_legend{path.suffix}"))


def _draw_pair_bars(ax, data: pd.DataFrame, *, include_family: bool) -> None:
    y = range(len(data))
    abstract_bars = ax.barh(
        [i - 0.18 for i in y],
        data["standardized_slope_change_per_year_abstract"],
        height=0.34,
        color=GREEN,
        edgecolor="0.25",
        linewidth=0.35,
        label="abstract",
    )
    title_bars = ax.barh(
        [i + 0.18 for i in y],
        data["standardized_slope_change_per_year_title"],
        height=0.34,
        color=ORCHID,
        hatch=HATCHES[1],
        edgecolor="0.25",
        linewidth=0.35,
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


def save_scope_legend(path: Path) -> None:
    handles = [
        Patch(facecolor=GREEN, edgecolor="0.25", label="abstract"),
        Patch(facecolor=ORCHID, edgecolor="0.25", hatch=HATCHES[1], label="title"),
    ]
    fig, ax = plt.subplots(figsize=(1.7, 0.28))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=2, frameon=False,
              fontsize=7, handlelength=1.2, columnspacing=0.8)
    fig.savefig(path.resolve(), dpi=200, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def short_feature_label(feature: str, family: str, *, include_family: bool) -> str:
    feature = str(feature)
    family = str(family)
    label = pretty_feature_label(feature)
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
