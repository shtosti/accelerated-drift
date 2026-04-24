from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TrendAnalyzer:
    """Aggregate and visualize per-year feature trends with IQR uncertainty bands."""

    def __init__(self, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns

    # =========================================================
    # YEARLY AGGREGATION (NOW INCLUDES QUANTILES)
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

            aggregations[f"{column}_yearly_mean"] = (column, "mean")
            aggregations[f"{column}_yearly_median"] = (column, "median")

            # 🔥 IQR components
            aggregations[f"{column}_yearly_q25"] = (column, lambda x: x.quantile(0.25))
            aggregations[f"{column}_yearly_q75"] = (column, lambda x: x.quantile(0.75))

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
    # PLOTTING (IQR BAND VERSION)
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

            q25_col = f"{feature}_yearly_q25"
            q75_col = f"{feature}_yearly_q75"
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

            # IQR band
            if q25_col in yearly.columns and q75_col in yearly.columns:
                q25 = yearly[q25_col]
                q75 = yearly[q75_col]

                ax.fill_between(
                    x,
                    q25,
                    q75,
                    alpha=0.25,
                    label="IQR (25–75%)",
                )

            # median (more robust than mean for NLP)
            median_col = f"{feature}_yearly_median"
            if median_col in yearly.columns:
                ax.plot(
                    x,
                    yearly[median_col],
                    marker="o",
                    linewidth=1.2,
                    label="Median",
                )
            else:
                ax.plot(x, y, marker="o", linewidth=1.2, label="Mean")

            ax.set_title(yearly_column)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.grid(True, alpha=0.3)
            ax.legend()

            fig.tight_layout()

            out_path = output_dir / f"{yearly_column}_iqr.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            paths.append(out_path)

        # =====================================================
        # MULTI-PANEL
        # =====================================================
        if plot_columns:
            ncols = 4
            nrows = int(np.ceil(len(plot_columns) / ncols))

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(5 * ncols, 3.8 * nrows),
                squeeze=False
            )

            axes = axes.flatten()

            for i, yearly_column in enumerate(plot_columns):
                ax = axes[i]

                feature = yearly_column.removesuffix("_yearly_mean")

                q25_col = f"{feature}_yearly_q25"
                q75_col = f"{feature}_yearly_q75"

                x = pd.to_datetime(yearly["year"].astype(str) + "-01-01", errors="coerce")
                y = yearly[yearly_column]

                if q25_col in yearly.columns and q75_col in yearly.columns:
                    ax.fill_between(x, yearly[q25_col], yearly[q75_col], alpha=0.2)

                ax.plot(x, y, marker="o", linewidth=1.0)

                ax.set_title(yearly_column)
                ax.xaxis.set_major_locator(mdates.YearLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                ax.grid(True, alpha=0.3)

            for j in range(len(plot_columns), len(axes)):
                axes[j].set_visible(False)

            fig.tight_layout()

            out_path = output_dir / "all_features_iqr_trends.png"
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