from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TrendAnalyzer:
    """Aggregate and visualize per-year feature trends with mean, std, and prevalence."""

    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns

    # =========================================================
    # YEARLY AGGREGATION (MEAN + STD + PREVALENCE)
    # =========================================================
    def aggregate_yearly(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        if "year" not in df.columns:
            raise ValueError("Expected a 'year' column in the dataset.")

        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        valid = df.dropna(subset=["year"]).copy()
        valid["year"] = valid["year"].astype(int)

        aggregations = {"paper_count": ("text_clean", "size")}

        for column in self.feature_columns:
            if column not in valid.columns:
                continue

            # ensure numeric
            valid[column] = pd.to_numeric(valid[column], errors="coerce")

            aggregations[f"{column}_yearly_mean"] = (column, "mean")
            aggregations[f"{column}_yearly_std"] = (column, "std")

            # prevalence: fraction of papers with feature > 0
            aggregations[f"{column}_yearly_prevalence"] = (
                column,
                lambda x: (x > 0).mean()
            )

        yearly = valid.groupby("year", as_index=False).agg(**aggregations)
        return yearly.sort_values("year").reset_index(drop=True)

    # =========================================================
    # MONTHLY (UNCHANGED)
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
    # PLOTTING (MEAN + STD + PREVALENCE)
    # =========================================================
    def save_plots(
        self,
        yearly: pd.DataFrame,
        monthly: pd.DataFrame,
        output_dir: Path
    ) -> list[Path]:

        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        plot_columns = [c for c in yearly.columns if c.endswith("_yearly_mean")]

        # =====================================================
        # SINGLE PLOTS
        # =====================================================
        for yearly_column in plot_columns:
            feature = yearly_column.removesuffix("_yearly_mean")

            std_col = f"{feature}_yearly_std"
            prev_col = f"{feature}_yearly_prevalence"
            monthly_col = f"{feature}_monthly_mean"

            fig, ax = plt.subplots(figsize=(6, 4))

            # -----------------------------
            # MONTHLY
            # -----------------------------
            if monthly_col in monthly.columns:
                ax.plot(
                    monthly["month_ts"],
                    monthly[monthly_col],
                    linewidth=1.0,
                    alpha=0.5,
                    label="Monthly mean",
                )

            # -----------------------------
            # YEARLY
            # -----------------------------
            x = pd.to_datetime(yearly["year"].astype(str) + "-01-01", errors="coerce")
            y = yearly[yearly_column]

            # remove NaNs
            mask = ~(y.isna())
            x = x[mask]
            y = y[mask]

            # -----------------------------
            # STD BAND
            # -----------------------------
            if std_col in yearly.columns:
                std = yearly[std_col][mask]

                lower = (y - std).clip(lower=0)
                upper = y + std

                ax.fill_between(
                    x,
                    lower,
                    upper,
                    alpha=0.25,
                    label="±1 std dev",
                )

            # -----------------------------
            # MEAN LINE
            # -----------------------------
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.5,
                label="Yearly mean",
            )

            # -----------------------------
            # PREVALENCE (SECOND AXIS)
            # -----------------------------
            if prev_col in yearly.columns:
                ax2 = ax.twinx()

                prev = yearly[prev_col][mask]

                ax2.plot(
                    x,
                    prev,
                    linestyle="--",
                    linewidth=1.2,
                    label="Prevalence",
                )

                ax2.set_ylabel("Prevalence (fraction)")
                ax2.set_ylim(0, 1)

            # -----------------------------
            # FORMATTING
            # -----------------------------
            ax.set_title(feature)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")

            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

            ax.grid(True, alpha=0.3)

            # combined legend
            lines, labels = ax.get_legend_handles_labels()
            if prev_col in yearly.columns:
                lines2, labels2 = ax2.get_legend_handles_labels()
                lines += lines2
                labels += labels2

            ax.legend(lines, labels)

            fig.tight_layout()

            out_path = output_dir / f"{feature}_mean_std_prevalence.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            paths.append(out_path)

        return paths

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