from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TrendAnalyzer:
    """Aggregate and visualize per-year feature trends."""

    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns

    def aggregate_yearly(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        if "year" not in df.columns:
            raise ValueError("Expected a 'year' column in the analyzed dataset.")

        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        valid = df.dropna(subset=["year"]).copy()
        valid["year"] = valid["year"].astype(int)

        aggregations: dict[str, str] = {"paper_count": ("text_clean", "size")}
        for column in self.feature_columns:
            if column not in valid.columns:
                continue
            aggregations[f"{column}_yearly_mean"] = (column, "mean")
            aggregations[f"{column}_yearly_median"] = (column, "median")

        yearly = valid.groupby("year", as_index=False).agg(**aggregations)
        yearly = yearly.sort_values("year").reset_index(drop=True)
        return yearly

    def aggregate_monthly(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        if "year" not in df.columns:
            raise ValueError("Expected a 'year' column in the analyzed dataset.")

        month_ts = self._resolve_month_timestamp(df)
        valid = df.copy()
        valid["month_ts"] = month_ts
        valid = valid.dropna(subset=["month_ts"]).copy()

        aggregations: dict[str, str] = {"paper_count": ("text_clean", "size")}
        for column in self.feature_columns:
            if column not in valid.columns:
                continue
            aggregations[f"{column}_monthly_mean"] = (column, "mean")
            aggregations[f"{column}_monthly_median"] = (column, "median")

        monthly = valid.groupby("month_ts", as_index=False).agg(**aggregations)
        monthly = monthly.sort_values("month_ts").reset_index(drop=True)
        return monthly

    def save_plots(self, yearly: pd.DataFrame, monthly: pd.DataFrame, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        plot_columns = [col for col in yearly.columns if col.endswith("_yearly_mean")]
        for yearly_column in plot_columns:
            feature_name = yearly_column.removesuffix("_yearly_mean")
            monthly_column = f"{feature_name}_monthly_mean"

            fig, ax = plt.subplots(figsize=(10, 4))

            if monthly_column in monthly.columns:
                ax.plot(
                    monthly["month_ts"],
                    monthly[monthly_column],
                    marker="o",
                    markersize=2,
                    linewidth=1.3,
                    alpha=0.8,
                    label="Monthly mean",
                )

            ax.plot(
                pd.to_datetime(yearly["year"].astype(str) + "-01-01", errors="coerce"),
                yearly[yearly_column],
                marker="o",
                markersize=5,
                linewidth=1.0,
                alpha=0.9,
                label="Yearly mean",
            )

            ax.set_title(yearly_column)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.minorticks_off()
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            out_path = output_dir / f"{yearly_column}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            paths.append(out_path)

        if plot_columns:
            ncols = 3
            nrows = int(np.ceil(len(plot_columns) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
            flat_axes = axes.flatten()

            for index, yearly_column in enumerate(plot_columns):
                feature_name = yearly_column.removesuffix("_yearly_mean")
                monthly_column = f"{feature_name}_monthly_mean"
                axis = flat_axes[index]
                if monthly_column in monthly.columns:
                    axis.plot(
                        monthly["month_ts"],
                        monthly[monthly_column],
                        marker="o",
                        markersize=1.5,
                        linewidth=1.0,
                        alpha=0.8,
                    )
                axis.plot(
                    pd.to_datetime(yearly["year"].astype(str) + "-01-01", errors="coerce"),
                    yearly[yearly_column],
                    marker="o",
                    markersize=3.5,
                    linewidth=0.9,
                    alpha=0.9,
                )
                axis.set_title(yearly_column)
                axis.set_xlabel("Year")
                axis.xaxis.set_major_locator(mdates.YearLocator())
                axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                axis.minorticks_off()
                axis.grid(True, alpha=0.3)

            for index in range(len(plot_columns), len(flat_axes)):
                flat_axes[index].set_visible(False)

            fig.tight_layout()
            combined_path = output_dir / "all_features_trends.png"
            fig.savefig(combined_path, dpi=150)
            plt.close(fig)
            paths.append(combined_path)

        return paths

    def _resolve_month_timestamp(self, frame: pd.DataFrame) -> pd.Series:
        if "publicationDate" in frame.columns:
            publication_date = pd.to_datetime(frame["publicationDate"], errors="coerce")
            publication_date = publication_date.dt.to_period("M").dt.to_timestamp()
        else:
            publication_date = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")

        fallback_year = pd.to_numeric(frame.get("year"), errors="coerce")
        fallback_year = fallback_year.astype("Int64")
        fallback_date = pd.to_datetime(fallback_year.astype(str) + "-01-01", errors="coerce")
        return publication_date.fillna(fallback_date)
