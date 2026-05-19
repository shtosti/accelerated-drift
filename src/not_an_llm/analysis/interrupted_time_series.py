from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from not_an_llm.analysis.feature_groups import FEATURE_GROUPS
from not_an_llm.analysis.label_map import LABEL_MAP


DEFAULT_INTERVENTION_DATE = "2022-11-30"
DEFAULT_PLACEBO_YEARS = (2018, 2019, 2020, 2021)


@dataclass(frozen=True, slots=True)
class ITSConfig:
    intervention_date: str = DEFAULT_INTERVENTION_DATE
    hac_lags: int = 6
    min_pre_months: int = 12
    min_post_months: int = 6
    placebo_years: tuple[int, ...] = DEFAULT_PLACEBO_YEARS


def compute_interrupted_time_series(
    monthly: pd.DataFrame,
    features: list[str] | None = None,
    *,
    config: ITSConfig | None = None,
) -> pd.DataFrame:
    """Fit monthly segmented regressions for analysis features.

    The primary estimand is ``slope_change``: the change in monthly trend after
    the intervention date. Standard errors use a HAC covariance estimate so the
    monthly residuals can be autocorrelated without making p-values too eager.
    """

    config = config or ITSConfig()
    features = features or _features_from_monthly(monthly)
    rows = [
        row
        for feature in features
        if (row := _fit_feature(monthly, feature, pd.Timestamp(config.intervention_date), config)) is not None
    ]
    result = pd.DataFrame(rows)
    if result.empty:
        return _empty_its_frame()

    result["slope_change_q"] = _adjust_p_values_by_family(result, "slope_change_p")
    result["level_shift_q"] = _adjust_p_values_by_family(result, "level_shift_p")
    result = result.sort_values(["family", "slope_change_q", "feature"], na_position="last")
    return result.reset_index(drop=True)


def compute_placebo_interrupted_time_series(
    monthly: pd.DataFrame,
    features: list[str] | None = None,
    *,
    config: ITSConfig | None = None,
) -> pd.DataFrame:
    """Run the same model against pre-declared placebo intervention years."""

    config = config or ITSConfig()
    features = features or _features_from_monthly(monthly)
    rows: list[dict[str, object]] = []
    for year in config.placebo_years:
        placebo_config = ITSConfig(
            intervention_date=f"{year}-01-01",
            hac_lags=config.hac_lags,
            min_pre_months=config.min_pre_months,
            min_post_months=config.min_post_months,
            placebo_years=config.placebo_years,
        )
        intervention = pd.Timestamp(placebo_config.intervention_date)
        for feature in features:
            row = _fit_feature(monthly, feature, intervention, placebo_config)
            if row is None:
                continue
            row["placebo_year"] = year
            rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return _empty_placebo_frame()

    result["slope_change_q"] = _adjust_p_values_by_family(result, "slope_change_p")
    return result.sort_values(["placebo_year", "family", "slope_change_q", "feature"]).reset_index(drop=True)


def save_its_slope_change_plot(
    stats: pd.DataFrame,
    output_path: Path,
    *,
    label_map: dict[str, str] | None = None,
    top_n: int = 20,
) -> Path | None:
    if stats.empty or "slope_change_per_year" not in stats.columns:
        return None

    plot_df = stats.dropna(subset=["slope_change_per_year"]).copy()
    if plot_df.empty:
        return None

    plot_df["_rank"] = plot_df["slope_change_per_year"].abs()
    plot_df = plot_df.sort_values("_rank", ascending=False).head(top_n)
    plot_df = plot_df.sort_values("slope_change_per_year")

    labels = label_map if label_map is not None else LABEL_MAP
    plot_df["label"] = plot_df["feature"].map(lambda feature: _pretty_label(str(feature), labels))

    plot_df["annotation"] = plot_df.apply(_format_its_annotation, axis=1)

    fig_height = max(4, len(plot_df) * 0.3 + 1.6)
    fig, ax = plt.subplots(figsize=(7, fig_height))
    colors = ["#943F8B" if value < 0 else "#54A066" for value in plot_df["slope_change_per_year"]]
    xerr = None
    if {"slope_change_per_year_ci_low", "slope_change_per_year_ci_high"}.issubset(plot_df.columns):
        xerr = np.vstack(
            [
                plot_df["slope_change_per_year"] - plot_df["slope_change_per_year_ci_low"],
                plot_df["slope_change_per_year_ci_high"] - plot_df["slope_change_per_year"],
            ]
        )
    bars = ax.barh(plot_df["label"], plot_df["slope_change_per_year"], color=colors, xerr=xerr, capsize=2)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Delta slope after intervention, feature units per year")

    for bar, annotation in zip(bars, plot_df["annotation"]):
        ax.text(
            1.01,
            bar.get_y() + bar.get_height() / 2,
            annotation,
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )

    fig.tight_layout(rect=(0, 0, 0.78, 1))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _fit_feature(
    monthly: pd.DataFrame,
    feature: str,
    intervention_date: pd.Timestamp,
    config: ITSConfig,
) -> dict[str, object] | None:
    value_col = f"{feature}_monthly_mean"
    if value_col not in monthly.columns:
        return None

    frame = _monthly_model_frame(monthly, value_col, intervention_date)
    if frame is None:
        return None

    pre_count = int((frame["post"] == 0).sum())
    post_count = int((frame["post"] == 1).sum())
    if pre_count < config.min_pre_months or post_count < config.min_post_months:
        return None

    x = frame[["const", "time", "post", "time_after"]].to_numpy(dtype=float)
    y = frame["value"].to_numpy(dtype=float)
    weights = frame["paper_count"].clip(lower=1.0).to_numpy(dtype=float)

    fit = _fit_weighted_hac(x, y, weights, config.hac_lags)
    if fit is None:
        return None

    pre_slope = float(fit["params"][1])
    level_shift = float(fit["params"][2])
    slope_change = float(fit["params"][3])
    level_shift_se = float(fit["se"][2])
    slope_change_se = float(fit["se"][3])
    post_slope = pre_slope + slope_change
    family = _feature_family(feature)

    return {
        "feature": feature,
        "family": family,
        "intervention_date": intervention_date.date().isoformat(),
        "n_months": len(frame),
        "n_pre_months": pre_count,
        "n_post_months": post_count,
        "pre_mean": float(frame.loc[frame["post"] == 0, "value"].mean()),
        "post_mean": float(frame.loc[frame["post"] == 1, "value"].mean()),
        "pre_slope_per_month": pre_slope,
        "pre_slope_per_year": pre_slope * 12.0,
        "level_shift": level_shift,
        "level_shift_se": level_shift_se,
        "level_shift_ci_low": level_shift - 1.96 * level_shift_se,
        "level_shift_ci_high": level_shift + 1.96 * level_shift_se,
        "level_shift_p": float(fit["p_values"][2]),
        "slope_change_per_month": slope_change,
        "slope_change_per_year": slope_change * 12.0,
        "slope_change_se": slope_change_se,
        "slope_change_per_year_se": slope_change_se * 12.0,
        "slope_change_per_year_ci_low": (slope_change - 1.96 * slope_change_se) * 12.0,
        "slope_change_per_year_ci_high": (slope_change + 1.96 * slope_change_se) * 12.0,
        "slope_change_p": float(fit["p_values"][3]),
        "post_slope_per_month": post_slope,
        "post_slope_per_year": post_slope * 12.0,
        "r_squared": float(fit["r_squared"]),
        "model": "monthly_segmented_wls_hac",
        "hac_lags": config.hac_lags,
    }


def _fit_weighted_hac(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    hac_lags: int,
) -> dict[str, np.ndarray | float] | None:
    if len(y) <= x.shape[1]:
        return None

    sqrt_w = np.sqrt(np.clip(weights, 1e-12, None))
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w

    xtx_inv = np.linalg.pinv(xw.T @ xw)
    params = xtx_inv @ xw.T @ yw
    residuals = y - x @ params
    weighted_residuals = residuals * sqrt_w

    meat = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    for t in range(len(y)):
        xt_ut = xw[t][:, None] * weighted_residuals[t]
        meat += xt_ut @ xt_ut.T

    max_lag = max(0, min(int(hac_lags), len(y) - 1))
    for lag in range(1, max_lag + 1):
        kernel_weight = 1.0 - lag / (max_lag + 1.0)
        gamma = np.zeros_like(meat)
        for t in range(lag, len(y)):
            left = xw[t][:, None] * weighted_residuals[t]
            right = xw[t - lag][:, None] * weighted_residuals[t - lag]
            gamma += left @ right.T
        meat += kernel_weight * (gamma + gamma.T)

    covariance = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    t_values = np.zeros_like(params)
    valid_se = se > 1e-12
    t_values[valid_se] = params[valid_se] / se[valid_se]
    t_values[~valid_se & (np.abs(params) > 1e-12)] = np.inf
    p_values = np.array([erfc(abs(value) / sqrt(2.0)) for value in t_values])

    y_mean = np.average(y, weights=weights)
    ss_res = float(np.sum(weights * residuals**2))
    ss_tot = float(np.sum(weights * (y - y_mean) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else np.nan

    return {
        "params": params,
        "se": se,
        "p_values": p_values,
        "r_squared": r_squared,
    }


def _monthly_model_frame(
    monthly: pd.DataFrame,
    value_col: str,
    intervention_date: pd.Timestamp,
) -> pd.DataFrame | None:
    if "month_ts" in monthly.columns:
        month_ts = pd.to_datetime(monthly["month_ts"], errors="coerce")
    elif {"year", "month"}.issubset(monthly.columns):
        month_ts = pd.to_datetime(
            {
                "year": pd.to_numeric(monthly["year"], errors="coerce"),
                "month": pd.to_numeric(monthly["month"], errors="coerce"),
                "day": 1,
            },
            errors="coerce",
        )
    else:
        return None

    if "paper_count" in monthly.columns:
        paper_count = pd.to_numeric(monthly["paper_count"], errors="coerce").fillna(1.0)
    else:
        paper_count = pd.Series(1.0, index=monthly.index)

    frame = pd.DataFrame(
        {
            "month_ts": month_ts,
            "value": pd.to_numeric(monthly[value_col], errors="coerce"),
            "paper_count": paper_count,
        }
    ).dropna(subset=["month_ts", "value"])
    if frame.empty:
        return None

    frame = frame.sort_values("month_ts").reset_index(drop=True)
    frame["time"] = np.arange(len(frame), dtype=float)
    frame["post"] = (frame["month_ts"] >= intervention_date).astype(float)
    if not frame["post"].any() or frame["post"].all():
        return None

    first_post_time = float(frame.loc[frame["post"] == 1, "time"].iloc[0])
    frame["time_after"] = np.where(frame["post"] == 1, frame["time"] - first_post_time + 1.0, 0.0)
    frame["const"] = 1.0
    return frame


def _features_from_monthly(monthly: pd.DataFrame) -> list[str]:
    return sorted(
        column.removesuffix("_monthly_mean")
        for column in monthly.columns
        if column.endswith("_monthly_mean")
    )


def _feature_family(feature: str) -> str:
    for family, members in FEATURE_GROUPS.items():
        if feature in members:
            return family
    if feature.startswith("word_") or feature.startswith("sequential_") or feature.startswith("causal_"):
        return "marker_words"
    if feature.startswith("verb_"):
        return "verbs"
    if feature.startswith("adjective_"):
        return "adjectives"
    if feature.startswith("phrase_"):
        return "phrases"
    readability_features = {
        "avg_words_per_sentence",
        "avg_syllables_per_word",
        "dale_chall",
        "smog_index",
        "automated_readability_index",
        "gunning_fog",
    }
    if "readability" in feature or feature.startswith("flesch") or feature in readability_features:
        return "readability"
    return "other"


def _format_its_annotation(row: pd.Series) -> str:
    delta = _format_number(row.get("slope_change_per_year"))
    ci_low = _format_number(row.get("slope_change_per_year_ci_low"))
    ci_high = _format_number(row.get("slope_change_per_year_ci_high"))
    q_value = _format_p_value(row.get("slope_change_q"))
    return f"Delta/year {delta} [{ci_low}, {ci_high}]; q={q_value}"


def _format_number(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(numeric):
        return "NA"
    abs_value = abs(numeric)
    if abs_value >= 10:
        return f"{numeric:.1f}"
    if abs_value >= 1:
        return f"{numeric:.2f}"
    return f"{numeric:.3f}"


def _format_p_value(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(numeric):
        return "NA"
    if numeric < 0.001:
        return "<0.001"
    return f"{numeric:.3f}"


def _adjust_p_values_by_family(frame: pd.DataFrame, p_column: str) -> pd.Series:
    adjusted = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, group in frame.groupby("family", dropna=False):
        p_values = pd.to_numeric(group[p_column], errors="coerce")
        valid = p_values.dropna()
        if valid.empty:
            continue
        adjusted.loc[valid.index] = _benjamini_hochberg(valid.to_numpy())
    return adjusted


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def _pretty_label(feature: str, label_map: dict[str, str]) -> str:
    mapped = label_map.get(feature, feature)
    return mapped.replace("`", "").replace("_", " ")


def _empty_its_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "feature",
            "family",
            "intervention_date",
            "n_months",
            "n_pre_months",
            "n_post_months",
            "pre_mean",
            "post_mean",
            "pre_slope_per_year",
            "level_shift",
            "level_shift_p",
            "level_shift_q",
            "slope_change_per_year",
            "slope_change_p",
            "slope_change_q",
            "post_slope_per_year",
            "r_squared",
            "model",
            "hac_lags",
        ]
    )


def _empty_placebo_frame() -> pd.DataFrame:
    frame = _empty_its_frame()
    frame["placebo_year"] = pd.Series(dtype="Int64")
    return frame
