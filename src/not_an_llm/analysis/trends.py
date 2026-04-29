from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .label_map import LABEL_MAP
from .feature_groups import FEATURE_GROUPS


class TrendAnalyzer:
    """Aggregate and visualize per-year feature trends with mean + confidence intervals."""

    def __init__(
        self,
        feature_columns: list[str],
        label_map: dict[str, str] | None = None,
    ) -> None:
        self.feature_columns = feature_columns
        self.label_map = label_map if label_map is not None else LABEL_MAP

        self.colors = {
            "monthly": "#1E9B4C",
            "yearly": "#9B4D8C",
            "events": "#484A59",
        }

    # =========================================================
    # LABELS
    # =========================================================
    def _pretty_label(self, feature: str) -> str:
        return self.label_map.get(feature, feature)

    # =========================================================
    # UTILS
    # =========================================================
    def _format_xticks(self, ax):
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    def _resolve_month_timestamp(self, df: pd.DataFrame) -> pd.Series:
        if "publicationDate" in df.columns:
            return pd.to_datetime(df["publicationDate"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        return pd.to_datetime(df["year"], format="%Y", errors="coerce")

    def _add_event_lines(self, ax, event_dates: dict[str, pd.Timestamp]) -> None:
        for label, d in event_dates.items():
            ax.axvline(d, linestyle="--", alpha=0.7, color="black")

    def _filter_features(self, features, exclude=None):
        if not exclude:
            return features
        return [f for f in features if f not in exclude]

    # =========================================================
    # YEARLUY/MONTHLY AGGREGATION
    # =========================================================

    def aggregate_monthly(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        if "year" not in df.columns:
            raise ValueError("Expected a 'year' column.")

        month_ts = self._resolve_month_timestamp(df)

        df = df.copy()
        df["month_ts"] = month_ts
        df = df.dropna(subset=["month_ts"])

        agg = {"paper_count": ("text_clean", "size")}

        for c in self.feature_columns:
            if c in df.columns:
                agg[f"{c}_monthly_mean"] = (c, "mean")

        monthly = df.groupby("month_ts", as_index=False).agg(**agg)
        return monthly.sort_values("month_ts").reset_index(drop=True)

    def aggregate_yearly(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        if "year" not in df.columns:
            raise ValueError("Expected a 'year' column.")

        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["year"]).copy()
        df["year"] = df["year"].astype(int)

        agg = {"paper_count": ("text_clean", "size")}

        for c in self.feature_columns:
            if c not in df.columns:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
            agg[f"{c}_yearly_mean"] = (c, "mean")
            agg[f"{c}_yearly_std"] = (c, "std")
            agg[f"{c}_yearly_n"] = (c, "count")

        yearly = df.groupby("year", as_index=False).agg(**agg)
        return yearly.sort_values("year").reset_index(drop=True)

    # =========================================================
    # STACKED PLOTS
    # =========================================================
    def save_stacked_word_plots(
        self,
        yearly: pd.DataFrame,
        monthly: pd.DataFrame,
        output_dir: Path,
        events: dict[str, str] | None = None,
        smoothing_window: int | None = None,
        exclude_features: set[str] | None = None,
    ) -> Path:

        output_dir.mkdir(parents=True, exist_ok=True)

        features = [
            c.removesuffix("_yearly_mean")
            for c in yearly.columns
            if c.endswith("_yearly_mean")
        ]

        features = self._filter_features(features, exclude_features)

        events = events or {
            "ChatGPT": "2022-11-30",
            "Delve": "2024-01-15",
        }

        event_dates = {k: pd.to_datetime(v) for k, v in events.items()}

        fig, axes = plt.subplots(len(features), 1, figsize=(12, 2.5 * len(features)))
        if len(features) == 1:
            axes = [axes]

        for ax, feature in zip(axes, features):

            ycol = f"{feature}_yearly_mean"
            mcol = f"{feature}_monthly_mean"

            if mcol in monthly.columns:
                y_m = monthly[mcol]
                if smoothing_window:
                    y_m = y_m.rolling(smoothing_window, center=True).mean()

                ax.plot(monthly["month_ts"], y_m, color="green", label="Monthly")

            x = pd.to_datetime(yearly["year"].astype(str) + "-01-01")
            y = yearly[ycol]

            ax.plot(x, y, marker="o", color="purple", label="Yearly")

            self._add_event_lines(ax, event_dates)

            ax.set_ylabel(self._pretty_label(feature))
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Year")
        for ax in axes:
            self._format_xticks(ax)

        fig.tight_layout()

        out_path = output_dir / "stacked_trends.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        return out_path

    # =========================================================
    # PRE/POST GLOBAL DIFF
    # =========================================================
    def save_pre_post_diff_plot(self, yearly: pd.DataFrame, output_dir: Path) -> Path:

        cols = [c for c in yearly.columns if c.endswith("_yearly_mean")]

        pre = yearly[yearly["year"] <= 2022]
        post = yearly[yearly["year"] >= 2023]

        rows = []

        for col in cols:
            feature = col.replace("_yearly_mean", "")

            pre_vals = pd.to_numeric(pre[col], errors="coerce")
            post_vals = pd.to_numeric(post[col], errors="coerce")

            if pre_vals.dropna().empty or post_vals.dropna().empty:
                continue

            rows.append({
                "feature": feature,
                "diff": post_vals.mean() - pre_vals.mean()
            })

        df = pd.DataFrame(rows).sort_values("diff")

        return save_grouped_difference_plot(
            df,
            output_dir / "pre_post_diff.png",
            "feature",
            "diff",
            "Post - Pre mean",
            self.label_map,
        )

    # =========================================================
    # GROUPED DIFFS (THIS IS YOUR MAIN REQUEST)
    # =========================================================
    def save_grouped_feature_diffs(
        self,
        yearly: pd.DataFrame,
        output_dir: Path,
        groups: dict[str, list[str]] = FEATURE_GROUPS,
        pre_cut: int = 2022,
        post_cut: int = 2023,
    ) -> dict[str, Path]:

        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}

        pre = yearly[yearly["year"] <= pre_cut]
        post = yearly[yearly["year"] >= post_cut]

        for group_name, features in groups.items():

            rows = []
            valid_cols = []

            for f in features:
                col = f"{f}_yearly_mean"
                if col not in yearly.columns:
                    continue

                valid_cols.append(col)

                pre_vals = pd.to_numeric(pre[col], errors="coerce")
                post_vals = pd.to_numeric(post[col], errors="coerce")

                if pre_vals.dropna().empty or post_vals.dropna().empty:
                    continue

                rows.append({
                    "feature": f,
                    "diff": post_vals.mean() - pre_vals.mean()
                })

            df = pd.DataFrame(rows)

            if df.empty:
                continue

            # TOTAL (correct pooled mean)
            if valid_cols:
                pre_total = pre[valid_cols].mean().mean()
                post_total = post[valid_cols].mean().mean()

                df = pd.concat([
                    df,
                    pd.DataFrame([{
                        "feature": f"{group_name}_TOTAL",
                        "diff": post_total - pre_total
                    }])
                ])

            df = df.sort_values("diff")

            out_path = output_dir / f"{group_name}_diff.png"

            save_grouped_difference_plot(
                df,
                out_path,
                "feature",
                "diff",
                "Post - Pre mean",
                self.label_map,
            )

            outputs[group_name] = out_path

        return outputs

    # =========================================================
    # GROUPED WORD PLOTS
    # =========================================================
    def save_grouped_word_plots(
        self,
        yearly: pd.DataFrame,
        monthly: pd.DataFrame,
        output_dir: Path,
        group_specs: dict[str, dict[str, object]],
        events: dict[str, str] | None = None,
        smoothing_window: int | None = None,
        exclude_features: set[str] | None = None,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        events = events or {
            "ChatGPT": "2022-11-30",
            "GPT-4": "2023-03-14",
            "AI detection": "2023-06-01",
            "Delve paper": "2024-01-15",
        }

        event_dates = {k: pd.to_datetime(v) for k, v in events.items()}

        for group_name, spec in group_specs.items():
            label = str(spec.get("label", group_name))
            rate_feature = str(spec.get("rate_feature", "")).strip()
            if not rate_feature or rate_feature not in yearly.columns:
                continue

            yearly_col = f"{rate_feature}_yearly_mean"
            monthly_col = f"{rate_feature}_monthly_mean"

            fig, ax = plt.subplots(figsize=(10, 4))

            if monthly_col in monthly.columns:
                y_m = monthly[monthly_col]
                if smoothing_window:
                    y_m = y_m.rolling(smoothing_window, center=True).mean()
                ax.plot(monthly["month_ts"], y_m, color=self.colors["monthly"], linewidth=1.2, alpha=1.0, label="Monthly mean")

            x = pd.to_datetime(yearly["year"].astype(str) + "-01-01")
            y = yearly[yearly_col]

            if smoothing_window:
                y_plot = pd.Series(y).rolling(smoothing_window, center=True).mean()
            else:
                y_plot = y

            ax.plot(x, y, marker="o", linewidth=1.8, color=self.colors["yearly"], label="Yearly mean")
            ax.plot(x, y_plot, linewidth=2.4, alpha=0.9, color=self.colors.get("ci", self.colors["yearly"]))

            self._add_event_lines(ax, event_dates)

            # ax.set_title(label)
            ax.set_ylabel("Mean count per 1k words")
            ax.set_xlabel("Year")
            ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
            ax.grid(True, alpha=0.3)

            top_y = ax.get_ylim()[1]
            for event_label, d in event_dates.items():
                ax.text(d, top_y, event_label, rotation=90, va="bottom", fontsize=8, alpha=0.8)

            ax.legend()
            self._format_xticks(ax)

            fig.tight_layout()

            out_path = output_dir / f"{group_name}_trend.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            paths.append(out_path)

        return paths

    # =========================================================
    # GENERIC PLOTS (single-feature mean plots)
    # =========================================================
    def save_plots(
        self,
        yearly: pd.DataFrame,
        monthly: pd.DataFrame,
        output_dir: Path,
        events: dict[str, str] | None = None,
        exclude_features: set[str] | None = None,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        plot_columns = [c for c in yearly.columns if c.endswith("_yearly_mean")]
        plot_columns = [column for column in plot_columns if column.removesuffix("_yearly_mean") not in (exclude_features or set())]

        event_dates = {k: pd.to_datetime(v) for k, v in (events or {}).items()}

        for yearly_column in plot_columns:
            feature = yearly_column.removesuffix("_yearly_mean")
            monthly_col = f"{feature}_monthly_mean"

            fig, ax = plt.subplots(figsize=(6, 4))

            if monthly_col in monthly.columns:
                ax.plot(monthly["month_ts"], monthly[monthly_col], color=self.colors["monthly"], linewidth=1.0, alpha=1.0, label="Monthly mean")

            x = pd.to_datetime(yearly["year"].astype(str) + "-01-01")
            y = yearly[yearly_column]

            ax.plot(x, y, marker="o", linewidth=1.5, color=self.colors["yearly"], label="Yearly mean")

            self._add_event_lines(ax, event_dates)

            ax.set_xlabel("Year")
            ax.set_ylabel(self._pretty_label(feature))
            ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
            ax.grid(True, alpha=0.3)
            ax.legend()
            self._format_xticks(ax)

            fig.tight_layout()

            out_path = output_dir / f"{feature}_mean.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            paths.append(out_path)

        return paths

    # =========================================================
    # DEPENDENCY DISTRIBUTION (stacked bars)
    # =========================================================
    def save_dependency_distribution_plot(self, df: pd.DataFrame, output_dir: Path, events: dict[str, str] | None = None) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        if "dependency_distribution" not in df.columns:
            raise ValueError("Expected 'dependency_distribution' column in dataframe")

        dep_df = df["dependency_distribution"].apply(pd.Series).fillna(0)
        dep_df["year"] = df["year"]

        dep_yearly = dep_df.groupby("year").sum(numeric_only=True)

        top_deps = dep_yearly.sum().sort_values(ascending=False).head(15).index
        dep_plot = dep_yearly[top_deps].copy()
        dep_plot["OTHER"] = dep_yearly.drop(columns=top_deps).sum(axis=1)

        dep_plot = dep_plot.div(dep_plot.sum(axis=1), axis=0)
        dep_plot = dep_plot.sort_index()
        dep_plot = dep_plot[dep_plot.mean().sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(10, 5))
        dep_plot.plot(kind="bar", stacked=True, ax=ax, width=0.9)

        ax.set_ylabel("Proportion")
        ax.set_xlabel("Year")
        # ax.set_title("Dependency Distribution Over Time")

        if events:
            event_years = {k: pd.to_datetime(v).year for k, v in events.items()}
            years = dep_plot.index.tolist()
            for label, year in event_years.items():
                if year in years:
                    x_pos = years.index(year)
                    color = "k" if "gpt" in label.lower() else self.colors["events"]
                    ax.axvline(x=x_pos, linestyle="--", alpha=0.7, color=color)
                    ax.text(x_pos, 1.02, label, rotation=90, va="bottom", fontsize=8)

        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8, title="Dependency")
        self._format_xticks(ax)
        plt.tight_layout()

        out_path = output_dir / "dependency_distribution_stacked.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path


# =========================================================
# PLOTTING FUNCTION
# =========================================================
def save_grouped_difference_plot(
    df: pd.DataFrame,
    output_path: Path,
    feature_column: str,
    diff_column: str,
    xlabel: str,
    label_map: dict[str, str],
    top_n: int = 50,
) -> Path:

    def _pretty_diff_label(feature: str) -> str:
        mapped = label_map.get(feature)
        if mapped:
            # Remove markdown-style quoting for cleaner axis labels.
            return mapped.replace("`", "")

        if feature.endswith("_TOTAL"):
            group = feature.removesuffix("_TOTAL").replace("_", " ")
            return f"{group} total"

        cleaned = feature
        cleaned = cleaned.removesuffix("_per_1k_words")
        cleaned = cleaned.removesuffix("_total")

        for prefix in ("word_", "verb_", "adjective_", "phrase_"):
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break

        return cleaned.replace("_", " ")

    df = df.dropna().copy()
    df = df.sort_values(diff_column, key=lambda s: s.abs(), ascending=False).head(top_n)
    df = df.sort_values(diff_column)

    df["label"] = df[feature_column].astype(str).map(_pretty_diff_label)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(df))))

    colors = ["#943F8B" if v < 0 else "#54A066" for v in df[diff_column]]

    ax.barh(df["label"], df[diff_column], color=colors)
    ax.axvline(0, color="black", linewidth=1)

    # ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path