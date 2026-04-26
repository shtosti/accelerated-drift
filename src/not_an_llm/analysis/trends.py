from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TrendAnalyzer:
    """Aggregate and visualize per-year feature trends with mean + confidence intervals."""

    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns
        self.colors = {
            "monthly": "#1E9B4C",
            "yearly": "#9B4D8C",
            "ci": "#55A868",
            "events": "#484A59",
            "dependencies": [
                "#4C72B0", "#DD8452", "#55A868", "#C44E52",
                "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
                "#CCB974", "#64B5CD"
            ]
        }

    def _format_xticks(self, ax):
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    def _add_event_lines(self, ax, event_dates: dict[str, pd.Timestamp]) -> None:
        for d in event_dates.values():
            ax.axvline(d, linestyle="--", alpha=0.7)

    def _filter_plot_features(
        self,
        features: list[str],
        exclude_features: set[str] | None = None,
    ) -> list[str]:
        if not exclude_features:
            return features
        return [feature for feature in features if feature not in exclude_features]

    # =========================================================
    # YEARLY AGGREGATION (MEAN + STD + COUNT FOR CI)
    # =========================================================
    def aggregate_yearly(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        if "year" not in df.columns:
            raise ValueError("Expected a 'year' column in the dataset.")

        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        valid = df.dropna(subset=["year"]).copy()
        valid["year"] = valid["year"].astype(int)

        aggregations = {
            "paper_count": ("text_clean", "size")
        }

        for column in self.feature_columns:
            if column not in valid.columns:
                continue

            valid[column] = pd.to_numeric(valid[column], errors="coerce")

            aggregations[f"{column}_yearly_mean"] = (column, "mean")
            aggregations[f"{column}_yearly_std"] = (column, "std")
            aggregations[f"{column}_yearly_n"] = (column, "count")

        yearly = valid.groupby("year", as_index=False).agg(**aggregations)
        return yearly.sort_values("year").reset_index(drop=True)

    # =========================================================
    # MONTHLY
    # =========================================================
    def aggregate_monthly(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        if "year" not in df.columns:
            raise ValueError("Expected a 'year' column in the dataset.")

        month_ts = self._resolve_month_timestamp(df)

        valid = df.copy()
        valid["month_ts"] = month_ts
        valid = valid.dropna(subset=["month_ts"]).copy()

        aggregations = {"paper_count": ("text_clean", "size")}

        for column in self.feature_columns:
            if column not in valid.columns:
                continue
            aggregations[f"{column}_monthly_mean"] = (column, "mean")

        monthly = valid.groupby("month_ts", as_index=False).agg(**aggregations)
        return monthly.sort_values("month_ts").reset_index(drop=True)

    # =========================================================
    # STACKED WORD PLOTS WITH EVENTS + CI
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
        features = self._filter_plot_features(features, exclude_features)

        if not features:
            raise ValueError("No features found to plot.")

        if events is None:
            events = {
                "ChatGPT": "2022-11-30",
                "GPT-4": "2023-03-14",
                "AI detection": "2023-06-01",
                "Delve paper": "2024-01-15",
            }

        event_dates = {k: pd.to_datetime(v) for k, v in events.items()}

        n = len(features)
        fig, axes = plt.subplots(
            nrows=n,
            ncols=1,
            figsize=(12, 2.5 * n),
            sharex=True
        )

        if n == 1:
            axes = [axes]

        for ax, feature in zip(axes, features):

            yearly_col = f"{feature}_yearly_mean"
            std_col = f"{feature}_yearly_std"
            n_col = f"{feature}_yearly_n"
            monthly_col = f"{feature}_monthly_mean"

            # -----------------------------
            # MONTHLY
            # -----------------------------
            if monthly_col in monthly.columns:
                y_m = monthly[monthly_col]

                if smoothing_window:
                    y_m = y_m.rolling(smoothing_window, center=True).mean()

                ax.plot(
                    monthly["month_ts"],
                    y_m,
                    color = self.colors["monthly"],
                    alpha=1.0,
                    linewidth=1.0,
                    label="Monthly",
                )

            # -----------------------------
            # YEARLY
            # -----------------------------
            x = pd.to_datetime(yearly["year"].astype(str) + "-01-01")
            y = yearly[yearly_col]

            if smoothing_window:
                y_plot = pd.Series(y).rolling(smoothing_window, center=True).mean()
            else:
                y_plot = y

            # # -----------------------------
            # # CONFIDENCE INTERVAL (95%)
            # # -----------------------------
            # if std_col in yearly.columns and n_col in yearly.columns:
            #     std = yearly[std_col]
            #     n_vals = yearly[n_col].replace(0, np.nan)

            #     se = std / np.sqrt(n_vals)
            #     ci = 1.96 * se

            #     lower = y_plot - ci
            #     upper = y_plot + ci

            #     ax.fill_between(
            #         x,
            #         lower,
            #         upper,
            #         alpha=0.15,
            #         label="95% CI",
            #     )

            # -----------------------------
            # LINES
            # -----------------------------
            ax.plot(
                x,
                y,
                marker="o",
                color = self.colors["yearly"],
                linewidth=1.0,
                alpha=1.0,
                label="Yearly mean (raw)",
            )

            # ax.plot(
            #     x,
            #     y_plot,
            #     linewidth=2.0,
            #     label="Smoothed trend",
            # )

            # -----------------------------
            # EVENTS
            # -----------------------------
            for d in event_dates.values():
                ax.axvline(d, linestyle="--", alpha=0.7, color=self.colors["events"])

            ax.set_ylabel(feature)
            ax.grid(alpha=0.9)

        # -----------------------------
        # EVENT LABELS
        # -----------------------------
        top_ax = axes[0]
        for label, d in event_dates.items():
            top_ax.text(
                d,
                top_ax.get_ylim()[1],
                label,
                rotation=90,
                va="bottom",
                fontsize=8,
                alpha=0.8,
            )

        axes[-1].set_xlabel("Year")
        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        for ax in axes:
            self._format_xticks(ax)

        fig.tight_layout()

        out_path = output_dir / "stacked_word_trends_with_events.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        return out_path

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

        if events is None:
            events = {
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

            if exclude_features and rate_feature in exclude_features:
                continue

            yearly_col = f"{rate_feature}_yearly_mean"
            monthly_col = f"{rate_feature}_monthly_mean"

            if yearly_col not in yearly.columns:
                continue

            fig, ax = plt.subplots(figsize=(10, 4))

            if monthly_col in monthly.columns:
                y_m = monthly[monthly_col]
                if smoothing_window:
                    y_m = y_m.rolling(smoothing_window, center=True).mean()
                ax.plot(
                    monthly["month_ts"],
                    y_m,
                    color=self.colors["monthly"],
                    linewidth=1.2,
                    alpha=1.0,
                    label="Monthly mean",
                )

            x = pd.to_datetime(yearly["year"].astype(str) + "-01-01")
            y = yearly[yearly_col]

            if smoothing_window:
                y_plot = pd.Series(y).rolling(smoothing_window, center=True).mean()
            else:
                y_plot = y

            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                color=self.colors["yearly"],
                label="Yearly mean",
            )

            ax.plot(
                x,
                y_plot,
                linewidth=2.4,
                alpha=0.9,
                color=self.colors["ci"],
                label="Smoothed trend",
            )

            for d in event_dates.values():
                ax.axvline(d, linestyle="--", alpha=0.7, color=self.colors["events"])

            ax.set_title(label)
            ax.set_ylabel("Mean count per 1k words")
            ax.set_xlabel("Year")

            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.grid(True, alpha=0.3)

            top_y = ax.get_ylim()[1]
            for event_label, d in event_dates.items():
                ax.text(
                    d,
                    top_y,
                    event_label,
                    rotation=90,
                    va="bottom",
                    fontsize=8,
                    alpha=0.8,
                )

            ax.legend()
            self._format_xticks(ax)

            fig.tight_layout()

            out_path = output_dir / f"{group_name}_trend.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            paths.append(out_path)

        return paths

    def save_dependency_distribution_plot(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        events: dict[str, str] | None = None,
    ) -> Path:

        output_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # EXPAND dependency distributions
        # -----------------------------
        dep_df = df["dependency_distribution"].apply(pd.Series).fillna(0)
        dep_df["year"] = df["year"]

        # -----------------------------
        # AGGREGATE per year
        # -----------------------------
        dep_yearly = dep_df.groupby("year").sum(numeric_only=True)

        # -----------------------------
        # SELECT top dependencies
        # -----------------------------
        top_deps = dep_yearly.sum().sort_values(ascending=False).head(15).index

        dep_plot = dep_yearly[top_deps].copy()
        dep_plot["OTHER"] = dep_yearly.drop(columns=top_deps).sum(axis=1)

        # -----------------------------
        # NORMALIZE (sum = 1 per year)
        # -----------------------------
        dep_plot = dep_plot.div(dep_plot.sum(axis=1), axis=0)

        # -----------------------------
        # SORT for stable plotting
        # -----------------------------
        dep_plot = dep_plot.sort_index()
        dep_plot = dep_plot[dep_plot.mean().sort_values(ascending=False).index]

        # -----------------------------
        # PLOT
        # -----------------------------
        fig, ax = plt.subplots(figsize=(10, 5))

        dep_plot.plot(
            kind="bar",
            stacked=True,
            ax=ax,
            width=0.9,
        )

        ax.set_ylabel("Proportion")
        ax.set_xlabel("Year")
        ax.set_title("Dependency Distribution Over Time")

        # -----------------------------
        # EVENTS
        # -----------------------------
        if events:
            event_years = {k: pd.to_datetime(v).year for k, v in events.items()}
            years = dep_plot.index.tolist()

            for label, year in event_years.items():
                if year in years:
                    x_pos = years.index(year)
                    ax.axvline(x=x_pos, linestyle="--", alpha=0.7, color=self.colors["events"])
                    ax.text(
                        x_pos,
                        1.02,
                        label,
                        va="bottom",
                        fontsize=8,
                    )

        # -----------------------------
        # LEGEND
        # -----------------------------
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=8,
            title="Dependency",
        )

        self._format_xticks(ax)

        plt.tight_layout()

        # -----------------------------
        # SAVE
        # -----------------------------
        out_path = output_dir / "dependency_distribution_stacked.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return out_path
    

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
        plot_columns = [
            column for column in plot_columns
            if column.removesuffix("_yearly_mean") not in (exclude_features or set())
        ]
        event_dates = (
        {k: pd.to_datetime(v) for k, v in events.items()}
        if events else {}
        )

        for yearly_column in plot_columns:
            feature = yearly_column.removesuffix("_yearly_mean")

            std_col = f"{feature}_yearly_std"
            n_col = f"{feature}_yearly_n"
            monthly_col = f"{feature}_monthly_mean"

            fig, ax = plt.subplots(figsize=(5, 4))

            # -----------------------------
            # MONTHLY
            # -----------------------------
            if monthly_col in monthly.columns:
                ax.plot(
                    monthly["month_ts"],
                    monthly[monthly_col],
                    color=self.colors["monthly"],
                    linewidth=1.0,
                    alpha=1.0,
                    label="Monthly mean",
                )

            x = pd.to_datetime(yearly["year"].astype(str) + "-01-01")
            y = yearly[yearly_column]

            # # -----------------------------
            # # CONFIDENCE INTERVAL
            # # -----------------------------
            # if std_col in yearly.columns and n_col in yearly.columns:
            #     std = yearly[std_col]
            #     n_vals = yearly[n_col].replace(0, np.nan)

            #     se = std / np.sqrt(n_vals)
            #     ci = 1.96 * se

            #     lower = y - ci
            #     upper = y + ci

            #     ax.fill_between(
            #         x,
            #         lower,
            #         upper,
            #         alpha=0.15,
            #         label="95% CI",
            #     )

            # -----------------------------
            # LINE
            # -----------------------------
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.5,
                color=self.colors["yearly"],
                label="Yearly mean",
            )
            ax.set_xlabel("Year")

            self._add_event_lines(ax, event_dates)

            ax.set_title(feature)
            # ax.set_xlabel("Year")
            # ax.set_ylabel("Value")

            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

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
    # MONTHLY
    # =========================================================
    def aggregate_monthly(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        if "year" not in df.columns:
            raise ValueError("Expected a 'year' column in the dataset.")

        month_ts = self._resolve_month_timestamp(df)

        valid = df.copy()
        valid["month_ts"] = month_ts
        valid = valid.dropna(subset=["month_ts"]).copy()

        aggregations = {"paper_count": ("text_clean", "size")}

        for column in self.feature_columns:
            if column not in valid.columns:
                continue
            aggregations[f"{column}_monthly_mean"] = (column, "mean")

        monthly = valid.groupby("month_ts", as_index=False).agg(**aggregations)
        return monthly.sort_values("month_ts").reset_index(drop=True)

    # =========================================================
    # TIME RESOLUTION
    # =========================================================
    def _resolve_month_timestamp(self, frame: pd.DataFrame) -> pd.Series:
        if "publicationDate" in frame.columns:
            pub = pd.to_datetime(frame["publicationDate"], errors="coerce")
            pub = pub.dt.to_period("M").dt.to_timestamp()
        else:
            pub = pd.Series(pd.NaT, index=frame.index)

        year = pd.to_numeric(frame.get("year"), errors="coerce").astype("Int64")
        fallback = pd.to_datetime(year.astype(str) + "-01-01", errors="coerce")

        return pub.fillna(fallback)