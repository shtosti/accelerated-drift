from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from not_an_llm.analysis.interrupted_time_series import compute_interrupted_time_series
from not_an_llm.analysis.label_map import pretty_feature_label


DEFAULT_TOPIC_FEATURES = [
    "marker_words_total_per_1k_words",
    "em_dash_per_1k_words",
    "certainty_ratio",
    "hedge_ratio",
    "avg_syllables_per_word",
    "gunning_fog",
    "dependency_entropy",
    "list_of_three_per_1k_words",
]


DOMAIN_LABELS = {
    "arxiv": "arXiv",
    "medarxiv": "medRxiv",
}


@dataclass(slots=True)
class TopicComparisonArtifacts:
    distribution_csv: Path
    feature_strength_csv: Path
    its_stats_csv: Path
    combined_csv: Path
    percent_change_heatmap_path: Path | None
    standardized_its_heatmap_path: Path | None
    selected_features_csv: Path | None = None


def select_top_its_features(
    *,
    analysis_dir: Path,
    domains: tuple[str, ...],
    top_n: int = 15,
    q_threshold: float = 0.05,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Select top unique features with the same ranking logic used for the main ITS table."""
    rows = []
    for domain in domains:
        path = analysis_dir / f"{domain}_its_stats.csv"
        _require_file(path)
        df = pd.read_csv(path)
        df["domain"] = domain
        df["dataset"] = DOMAIN_LABELS.get(domain, domain)
        rows.append(df)

    ranked = pd.concat(rows, ignore_index=True)
    ranked = ranked[pd.to_numeric(ranked["slope_change_q"], errors="coerce") < q_threshold].copy()
    ranked["abs_delta_beta_sd"] = pd.to_numeric(
        ranked["standardized_slope_change_per_year"],
        errors="coerce",
    ).abs()
    ranked = ranked.sort_values("abs_delta_beta_sd", ascending=False).reset_index(drop=True)
    ranked.insert(0, "global_rank", range(1, len(ranked) + 1))
    ranked = ranked.drop_duplicates("feature", keep="first").head(top_n).copy()
    ranked.insert(0, "feature_rank", range(1, len(ranked) + 1))

    feature_columns = [
        "feature_rank",
        "global_rank",
        "domain",
        "dataset",
        "family",
        "feature",
        "pre_mean",
        "post_mean",
        "standardized_slope_change_per_year",
        "standardized_slope_change_per_year_ci_low",
        "standardized_slope_change_per_year_ci_high",
        "slope_change_q",
    ]
    ranked = ranked[[column for column in feature_columns if column in ranked.columns]]
    features = tuple(dict.fromkeys(ranked["feature"].tolist()))
    return features, ranked


def compare_topic_distributions_and_features(
    *,
    analysis_dir: Path,
    output_dir: Path,
    domains: tuple[str, ...] = ("arxiv", "medarxiv"),
    features: tuple[str, ...] = tuple(DEFAULT_TOPIC_FEATURES),
    intervention_year: int = 2022,
    post_start_year: int = 2023,
    latest_year: int | None = None,
    min_topic_share: float = 0.0,
    make_heatmap: bool = True,
) -> TopicComparisonArtifacts:
    """Compare topic prevalence and feature strength for existing topic outputs.

    The function expects files produced by the main analysis pipeline:
    ``{domain}_topic_summary.csv``, ``{domain}_topic_prevalence_yearly.csv``,
    and ``{domain}_topics/topic_{id}/trends_by_year.csv``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    distribution = pd.concat(
        [
            _load_topic_distribution(
                analysis_dir=analysis_dir,
                domain=domain,
                intervention_year=intervention_year,
                latest_year=latest_year,
                min_topic_share=min_topic_share,
            )
            for domain in domains
        ],
        ignore_index=True,
    )

    feature_strength = pd.concat(
        [
            _load_topic_feature_strength(
                analysis_dir=analysis_dir,
                domain=domain,
                features=features,
                intervention_year=intervention_year,
                post_start_year=post_start_year,
                min_topic_share=min_topic_share,
            )
            for domain in domains
        ],
        ignore_index=True,
    )

    its_stats = pd.concat(
        [
            _load_topic_its_stats(
                analysis_dir=analysis_dir,
                domain=domain,
                features=features,
                min_topic_share=min_topic_share,
            )
            for domain in domains
        ],
        ignore_index=True,
    )

    combined = feature_strength.merge(
        distribution[
            [
                "domain",
                "topic_id",
                "intervention_year_pct",
                "latest_year",
                "latest_year_pct",
                "pct_point_change_since_intervention",
            ]
        ],
        on=["domain", "topic_id"],
        how="left",
    )

    distribution_csv = output_dir / "topic_distribution_comparison.csv"
    feature_strength_csv = output_dir / "topic_feature_strength_comparison.csv"
    its_stats_csv = output_dir / "topic_its_standardized_slope_comparison.csv"
    combined_csv = output_dir / "topic_distribution_feature_strength_comparison.csv"

    distribution.to_csv(distribution_csv, index=False)
    feature_strength.to_csv(feature_strength_csv, index=False)
    its_stats.to_csv(its_stats_csv, index=False)
    combined.to_csv(combined_csv, index=False)

    percent_change_heatmap_path = None
    standardized_its_heatmap_path = None
    if make_heatmap:
        percent_change_heatmap_path = output_dir / "topic_feature_strength_percent_change_heatmap.png"
        _save_feature_strength_heatmap(combined, features, percent_change_heatmap_path)
        standardized_its_heatmap_path = output_dir / "topic_standardized_its_slope_change_heatmap.png"
        _save_standardized_its_heatmap(its_stats, features, standardized_its_heatmap_path)

    return TopicComparisonArtifacts(
        distribution_csv=distribution_csv,
        feature_strength_csv=feature_strength_csv,
        its_stats_csv=its_stats_csv,
        combined_csv=combined_csv,
        percent_change_heatmap_path=percent_change_heatmap_path,
        standardized_its_heatmap_path=standardized_its_heatmap_path,
    )


def save_selected_its_features(selected: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)
    return output_path


def _load_topic_distribution(
    *,
    analysis_dir: Path,
    domain: str,
    intervention_year: int,
    latest_year: int | None,
    min_topic_share: float,
) -> pd.DataFrame:
    summary_path = analysis_dir / f"{domain}_topic_summary.csv"
    prevalence_path = analysis_dir / f"{domain}_topic_prevalence_yearly.csv"
    _require_file(summary_path)
    _require_file(prevalence_path)

    summary = pd.read_csv(summary_path)
    prevalence = pd.read_csv(prevalence_path)
    summary = summary[summary["abstract_share"] >= min_topic_share].copy()

    available_years = sorted(int(year) for year in prevalence["year"].dropna().unique())
    if not available_years:
        raise ValueError(f"No yearly topic prevalence rows found in {prevalence_path}")

    selected_latest_year = latest_year or max(available_years)
    if selected_latest_year not in available_years:
        raise ValueError(
            f"Requested latest_year={selected_latest_year} is absent from {prevalence_path}; "
            f"available years: {available_years}"
        )

    wide = prevalence.pivot_table(index="topic_id", columns="year", values="pct", aggfunc="first")
    rows = []
    for topic in summary.itertuples(index=False):
        topic_id = int(topic.topic_id)
        intervention_pct = _value_at_year(wide, topic_id, intervention_year)
        latest_pct = _value_at_year(wide, topic_id, selected_latest_year)
        rows.append(
            {
                "domain": domain,
                "topic_id": topic_id,
                "topic_label": topic.topic_label,
                "abstract_count": int(topic.abstract_count),
                "abstract_share": float(topic.abstract_share),
                "intervention_year": intervention_year,
                "intervention_year_pct": intervention_pct,
                "latest_year": selected_latest_year,
                "latest_year_pct": latest_pct,
                "pct_point_change_since_intervention": latest_pct - intervention_pct,
            }
        )

    return pd.DataFrame(rows).sort_values(["domain", "abstract_count"], ascending=[True, False])


def _load_topic_feature_strength(
    *,
    analysis_dir: Path,
    domain: str,
    features: tuple[str, ...],
    intervention_year: int,
    post_start_year: int,
    min_topic_share: float,
) -> pd.DataFrame:
    summary_path = analysis_dir / f"{domain}_topic_summary.csv"
    _require_file(summary_path)

    summary = pd.read_csv(summary_path)
    summary = summary[summary["abstract_share"] >= min_topic_share].copy()

    rows = []
    for topic in summary.itertuples(index=False):
        topic_id = int(topic.topic_id)
        trends_path = analysis_dir / f"{domain}_topics" / f"topic_{topic_id}" / "trends_by_year.csv"
        _require_file(trends_path)
        trends = pd.read_csv(trends_path)
        pre = trends[trends["year"] <= intervention_year]
        post = trends[trends["year"] >= post_start_year]

        row: dict[str, object] = {
            "domain": domain,
            "topic_id": topic_id,
            "topic_label": topic.topic_label,
            "abstract_count": int(topic.abstract_count),
            "abstract_share": float(topic.abstract_share),
            "pre_end_year": intervention_year,
            "post_start_year": post_start_year,
        }

        for feature in features:
            mean_col = f"{feature}_yearly_mean"
            if mean_col not in trends.columns:
                continue
            pre_mean = _weighted_mean(pre, mean_col)
            post_mean = _weighted_mean(post, mean_col)
            row[f"{feature}_pre_mean"] = pre_mean
            row[f"{feature}_post_mean"] = post_mean
            row[f"{feature}_absolute_change"] = post_mean - pre_mean
            row[f"{feature}_pct_change"] = _pct_change(pre_mean, post_mean)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["domain", "abstract_count"], ascending=[True, False])


def _load_topic_its_stats(
    *,
    analysis_dir: Path,
    domain: str,
    features: tuple[str, ...],
    min_topic_share: float,
) -> pd.DataFrame:
    summary_path = analysis_dir / f"{domain}_topic_summary.csv"
    _require_file(summary_path)

    summary = pd.read_csv(summary_path)
    summary = summary[summary["abstract_share"] >= min_topic_share].copy()

    rows = []
    for topic in summary.itertuples(index=False):
        topic_id = int(topic.topic_id)
        trends_path = analysis_dir / f"{domain}_topics" / f"topic_{topic_id}" / "trends_by_month.csv"
        _require_file(trends_path)
        monthly = pd.read_csv(trends_path)
        available_features = [
            feature
            for feature in features
            if f"{feature}_monthly_mean" in monthly.columns
        ]
        if not available_features:
            continue

        stats = compute_interrupted_time_series(monthly, list(available_features))
        if stats.empty:
            continue

        stats.insert(0, "domain", domain)
        stats.insert(1, "topic_id", topic_id)
        stats.insert(2, "topic_label", topic.topic_label)
        stats.insert(3, "abstract_count", int(topic.abstract_count))
        stats.insert(4, "abstract_share", float(topic.abstract_share))
        rows.append(stats)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True).sort_values(
        ["domain", "abstract_count", "feature"],
        ascending=[True, False, True],
    )


def _save_feature_strength_heatmap(df: pd.DataFrame, features: tuple[str, ...], output_path: Path) -> None:
    heatmap_cols = [f"{feature}_pct_change" for feature in features if f"{feature}_pct_change" in df.columns]
    if df.empty or not heatmap_cols:
        return

    plot_df = df.copy()
    plot_df["topic"] = plot_df.apply(
        lambda row: f"{_domain_label(row['domain'])}: {row['topic_id']} ({row['abstract_share']:.1%})",
        axis=1,
    )
    values = plot_df[heatmap_cols].replace([np.inf, -np.inf], np.nan)
    labels = [pretty_feature_label(col.removesuffix("_pct_change")) for col in heatmap_cols]

    fig_height = 0.25 * len(plot_df)
    fig_width = 0.5 * len(heatmap_cols)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    vmax = float(np.nanpercentile(np.abs(values.to_numpy(dtype=float)), 95))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(plot_df)), labels=plot_df["topic"])
    # ax.set_title("Post/pre feature change by topic (%)")

    for y, (_, row) in enumerate(values.iterrows()):
        for x, value in enumerate(row):
            if pd.isna(value):
                text = ""
            else:
                text = f"{value:.0f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(image, ax=ax, label="Post/pre change (%)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_standardized_its_heatmap(df: pd.DataFrame, features: tuple[str, ...], output_path: Path) -> None:
    if df.empty or "standardized_slope_change_per_year" not in df.columns:
        return

    value_table = df.pivot_table(
        index=["domain", "topic_id", "abstract_share"],
        columns="feature",
        values="standardized_slope_change_per_year",
        aggfunc="first",
    )
    available_features = [feature for feature in features if feature in value_table.columns]
    if not available_features:
        return
    value_table = value_table[available_features].reset_index()
    value_table["topic"] = value_table.apply(
        lambda row: f"{_domain_label(row['domain'])}: {int(row['topic_id'])} ({row['abstract_share']:.1%})",
        axis=1,
    )

    values = value_table[available_features].replace([np.inf, -np.inf], np.nan)
    labels = [pretty_feature_label(feature) for feature in available_features]

    fig_height = 0.25 * len(value_table)
    fig_width = 0.5 * len(available_features)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    vmax = float(np.nanpercentile(np.abs(values.to_numpy(dtype=float)), 95))
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    image = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(value_table)), labels=value_table["topic"])
    # ax.set_title("Topic ITS slope change (pre SD/year)")

    for y, (_, row) in enumerate(values.iterrows()):
        for x, value in enumerate(row):
            text = "" if pd.isna(value) else f"{value:.1f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=7, color="black")

    fig.colorbar(image, ax=ax, label=rf"Standardized $\Delta$ slope")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _weighted_mean(df: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(df[column], errors="coerce")
    weights = pd.to_numeric(df.get("paper_count"), errors="coerce")
    valid = values.notna() & weights.notna() & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _pct_change(pre_mean: float, post_mean: float) -> float:
    if not np.isfinite(pre_mean) or pre_mean == 0.0 or not np.isfinite(post_mean):
        return float("nan")
    return float(100.0 * (post_mean - pre_mean) / abs(pre_mean))


def _value_at_year(wide: pd.DataFrame, topic_id: int, year: int) -> float:
    if topic_id not in wide.index or year not in wide.columns:
        return float("nan")
    value = wide.loc[topic_id, year]
    return float(value) if pd.notna(value) else float("nan")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required topic analysis file not found: {path}")


def _domain_label(domain: object) -> str:
    value = str(domain)
    return DOMAIN_LABELS.get(value, value)
