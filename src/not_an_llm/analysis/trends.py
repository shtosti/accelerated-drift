from __future__ import annotations

from pathlib import Path

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

    def save_plots(self, yearly: pd.DataFrame, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        plot_columns = [col for col in yearly.columns if col.endswith("_yearly_mean")]
        for column in plot_columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(yearly["year"], yearly[column], marker="o")
            ax.set_title(column)
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            out_path = output_dir / f"{column}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            paths.append(out_path)

        if plot_columns:
            ncols = 3
            nrows = int(np.ceil(len(plot_columns) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)
            flat_axes = axes.flatten()

            for index, column in enumerate(plot_columns):
                axis = flat_axes[index]
                axis.plot(yearly["year"], yearly[column], marker="o")
                axis.set_title(column)
                axis.set_xlabel("Year")
                axis.grid(True, alpha=0.3)

            for index in range(len(plot_columns), len(flat_axes)):
                flat_axes[index].set_visible(False)

            fig.tight_layout()
            combined_path = output_dir / "all_features_trends.png"
            fig.savefig(combined_path, dpi=150)
            plt.close(fig)
            paths.append(combined_path)

        return paths
