from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from scipy import stats

from not_an_llm.analysis.label_map import pretty_feature_label


DEFAULT_PRIMARY_DATASETS = (
    "arxiv_ai_abstracts",
    "arxiv_qbio_abstracts",
    "arxiv_stat_abstracts",
    "medarxiv_abstracts",
    "arxiv_ai_titles",
    "arxiv_qbio_titles",
    "arxiv_stat_titles",
    "medarxiv_titles",
)
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
    "arxiv_ai_abstracts": "AI abs.",
    "arxiv_qbio_abstracts": "q-bio abs.",
    "arxiv_stat_abstracts": "stat abs.",
    "medarxiv_abstracts": "medRxiv abs.",
    "arxiv_ai_titles": "AI title",
    "arxiv_qbio_titles": "q-bio title",
    "arxiv_stat_titles": "stat title",
    "medarxiv_titles": "medRxiv title",
}
DECREASE_COLOR = "#943F8B"
INCREASE_COLOR = "#54A066"
COMPARISON_HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "orchid_white_green",
    [DECREASE_COLOR, "#FFFFFF", INCREASE_COLOR],
)
DATASET_COLORS = {
    "arxiv_ai_abstracts": "#2F7E41",
    "arxiv_qbio_abstracts": "#54A066",
    "arxiv_stat_abstracts": "#86BD8A",
    "medarxiv_abstracts": "#B7D8B1",
    "arxiv_ai_titles": "#6D286D",
    "arxiv_qbio_titles": "#943F8B",
    "arxiv_stat_titles": "#B873AA",
    "medarxiv_titles": "#D3A8CD",
}


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
    save_pairwise_correlation_heatmap(
        dependency_bigram_corr,
        datasets,
        visuals_output_dir / "dependency_bigram_correlation_heatmap.png",
        colorbar_label="Pearson r",
    )

    print(f"Saved corpus comparison tables to {analysis_output_dir}")
    print(f"Saved corpus comparison figures to {visuals_output_dir}")
    print(summary.to_string(index=False))


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


def save_correlation_heatmap(pairwise: pd.DataFrame, datasets: list[str], path: Path) -> None:
    if pairwise.empty:
        return
    save_pairwise_correlation_heatmap(pairwise, datasets, path, colorbar_label="Pearson r")


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
    ax.barh(summary.index[::-1], summary.iloc[::-1], color=colors[::-1])
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
        xlabel="bigram-share slope/year",
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
    fig_height = max(2.2, 0.22 * len(data) + 0.8)
    fig, ax = plt.subplots(figsize=(3.35, fig_height))
    offsets = np.linspace(-height * (len(value_cols) - 1) / 2, height * (len(value_cols) - 1) / 2, len(value_cols))
    for offset, column in zip(offsets, value_cols, strict=True):
        ax.barh(y + offset, data[column], height=height, color=DATASET_COLORS.get(column, "0.5"), label=label(column))
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
        Patch(facecolor=DATASET_COLORS.get(column, "0.5"), edgecolor="none", label=label(column))
        for column in value_cols
    ]
    if not handles:
        return
    ncol = min(4, len(handles))
    rows = int(np.ceil(len(handles) / ncol))
    fig, ax = plt.subplots(figsize=(3.35, 0.24 * rows + 0.18))
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


def save_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    *,
    vmin: float,
    vmax: float,
    cmap: str | LinearSegmentedColormap,
    fmt: str,
    colorbar_label: str | None = None,
) -> None:
    n = len(matrix)
    fig, ax = plt.subplots(figsize=(3.35, max(2.4, 0.22 * n + 0.55)))
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(n), [label(value) for value in matrix.columns], rotation=45, ha="right")
    ax.set_yticks(range(n), [label(value) for value in matrix.index])
    ax.tick_params(axis="both", labelsize=6)
    for i in range(n):
        for j in range(n):
            value = matrix.iat[i, j]
            if pd.notna(value):
                if i == j:
                    text = "1"
                else:
                    text = compact_float(value, fmt)
                ax.text(j, i, text, ha="center", va="center", fontsize=4.5, linespacing=0.8)
    cbar = fig.colorbar(image, ax=ax, shrink=0.75)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=6)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
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
    mapped = pretty_feature_label(feature).replace("`", "")
    if mapped != feature:
        return mapped
    return feature.replace("_per_1k_words", "").replace("_", " ")


def label(dataset: str) -> str:
    return SHORT_DATASET_LABELS.get(dataset, dataset.replace("arxiv_", "").replace("_", " "))


if __name__ == "__main__":
    main()
