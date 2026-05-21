from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import ruptures as rpt

from .label_map import LABEL_MAP, pretty_feature_label
from .feature_groups import FEATURE_GROUPS


def _simple_percent_change(pre_value: float, post_value: float) -> float:

    pre_value = float(pre_value)
    post_value = float(post_value)
    return 100.0 * (post_value - pre_value) / abs(pre_value) if pre_value != 0 else np.nan


def is_group_total_feature(feature: str) -> bool:
    return feature.endswith("_total_per_1k_words")


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
        return pretty_feature_label(feature, self.label_map)

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
        monthly["year"] = monthly["month_ts"].dt.year.astype(int)
        monthly["month"] = monthly["month_ts"].dt.month.astype(int)
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
    
    def _legacy_yearly_segmented_regression(self, df: pd.DataFrame, value_col: str, intervention_year: int = 2022):
        """Compute Interrupted Time Series regression.
        
        Fits: y = β₀ + β₁*time + β₂*intervention + β₃*time_since_intervention
        
        Where:
          - β₀ = baseline level
          - β₁ = pre-intervention slope
          - β₂ = level shift at intervention (immediate jump)
          - β₃ = slope change post-intervention (combined post-slope = β₁ + β₃)
        
        Returns dict with ITS parameters and p-values for interpretation:
          - its_pre_slope: slope before intervention
          - its_level_shift: immediate level change at intervention  
          - its_slope_change: change in slope after intervention
          - And their respective p-values and significance flags
        """
        try:
            # Prepare data
            years = df["year"].values
            y = df[value_col].values
            
            # Need at least 4 data points for 4 parameters
            if len(years) < 4:
                return None
            
            # Create predictor variables for piecewise linear regression
            # Time encoded as years from start (0-indexed)
            time = years - years.min()
            
            # Intervention indicator: 1 if year >= intervention_year, else 0
            intervention = (years >= intervention_year).astype(int)
            
            # Time since intervention: 0 before intervention, then positive after
            time_since_intervention = np.maximum(0, years - intervention_year)
            
            # Design matrix: [1, time, intervention, time_since_intervention]
            X = np.column_stack([
                np.ones(len(years)),
                time,
                intervention,
                time_since_intervention
            ])
            
            # Fit piecewise linear regression
            reg = LinearRegression().fit(X, y)
            
            _, pre_slope, level_shift, slope_change = reg.coef_
            
            # Compute standard errors and t-statistics using pseudo-inverse for robustness
            y_pred = reg.predict(X)
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            mse = ss_res / (len(y) - 4) if len(y) > 4 else np.mean(residuals**2)
            
            # Use pseudo-inverse for robustness
            XtX_inv = np.linalg.pinv(X.T @ X)
            var_covar = mse * XtX_inv
            std_errors = np.sqrt(np.abs(np.diag(var_covar)))  # abs to handle numerical issues
            
            _, se_pre_slope, se_level_shift, se_slope_change = std_errors
            
            # Compute t-statistics and p-values
            t_pre_slope = pre_slope / se_pre_slope if se_pre_slope > 1e-10 else 0
            t_level_shift = level_shift / se_level_shift if se_level_shift > 1e-10 else 0
            t_slope_change = slope_change / se_slope_change if se_slope_change > 1e-10 else 0
            
            df_resid = len(y) - 4  # degrees of freedom
            p_pre_slope = 2 * (1 - stats.t.cdf(abs(t_pre_slope), df_resid)) if df_resid > 0 else 1.0
            p_level_shift = 2 * (1 - stats.t.cdf(abs(t_level_shift), df_resid)) if df_resid > 0 else 1.0
            p_slope_change = 2 * (1 - stats.t.cdf(abs(t_slope_change), df_resid)) if df_resid > 0 else 1.0
            
            # Compute R-squared
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
            
            return {
                "its_pre_slope": float(pre_slope),
                "its_pre_slope_p": float(p_pre_slope),
                "its_level_shift": float(level_shift),
                "its_level_shift_p": float(p_level_shift),
                "its_slope_change": float(slope_change),
                "its_slope_change_p": float(p_slope_change),
                "its_post_slope": float(pre_slope + slope_change),
                "its_r_squared": float(r_squared),
            }
        except Exception as e:
            # Silently return None if ITS computation fails
            # This can happen with constant-valued features or numerical issues
            return None

    def _compute_stats(self, yearly: pd.DataFrame, feature: str, pre_cut=2022, post_cut=2023):
        col = f"{feature}_yearly_mean"
        if col not in yearly.columns:
            return None

        df = yearly[["year", col]].dropna().copy()

        pre = df[df["year"] <= pre_cut][col]
        post = df[df["year"] >= post_cut][col]

        if len(pre) < 2 or len(post) < 2:
            return {
                "feature": feature,
                "diff": np.nan,
                "p_value": np.nan,
                "cohens_d": np.nan,
                "slope": np.nan,
                "slope_p": np.nan,
                "change_point": np.nan,
            }

        # ---------------------------
        # 1. t-test
        # ---------------------------
        t_stat, p_val = stats.ttest_ind(pre, post, equal_var=False)

        # ---------------------------
        # 2. Cohen's d
        # ---------------------------
        def cohens_d(a, b):
            n1, n2 = len(a), len(b)
            s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
            pooled = np.sqrt(((n1 - 1)*s1 + (n2 - 1)*s2) / (n1 + n2 - 2))
            if pooled == 0:
                return 0.0
            return (np.mean(b) - np.mean(a)) / pooled

        d = cohens_d(pre, post)

        # ---------------------------
        # 3. regression slope
        # ---------------------------
        X = df["year"].values.reshape(-1, 1)
        y = df[col].values

        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0]

        # slope significance (approx)
        # _, slope_p = stats.pearsonr(df["year"], y)
        if np.std(y) == 0:
            slope_p = 1.0
        else:
            _, slope_p = stats.pearsonr(df["year"], y)

        # ---------------------------
        # 4. change point (ruptures)
        # ---------------------------
        try:
            algo = rpt.Pelt(model="l2").fit(y)
            cps = algo.predict(pen=1)
            cp_year = df["year"].iloc[cps[0]-1] if cps else None
        except Exception:
            cp_year = None

        # ---------------------------
        # Percent change from the pre-period baseline.
        # ---------------------------
        diff = _simple_percent_change(pre.mean(), post.mean())

        result = {
            "feature": feature,
            "diff": diff,
            "p_value": p_val,
            "cohens_d": d,
            "slope": slope,
            "slope_p": slope_p,
            "change_point": cp_year,
        }

        return result

    def compute_all_stats(self, yearly: pd.DataFrame) -> pd.DataFrame:
        features = [
            c.removesuffix("_yearly_mean")
            for c in yearly.columns
            if c.endswith("_yearly_mean")
        ]

        rows = []
        for f in features:
            res = self._compute_stats(yearly, f)

            if res is None:
                rows.append({"feature": f})  # <-- keep feature, but empty stats
            else:
                res["feature"] = f
                rows.append(res)

        df = pd.DataFrame(rows)

        if df.empty:
            return pd.DataFrame(columns=["feature"])

        df["p_adj"] = np.minimum(df["p_value"] * len(df), 1.0)
        return df.sort_values("p_value")

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

        # Ensure monthly has proper datetime month_ts before plotting.
        monthly = monthly.copy()
        if "month_ts" in monthly.columns:
            monthly["month_ts"] = pd.to_datetime(monthly["month_ts"], errors="coerce")

        fig, axes = plt.subplots(len(features), 1, figsize=(3.5, 2.2 * len(features)))
        if len(features) == 1:
            axes = [axes]

        x = pd.to_datetime(yearly["year"].astype(str) + "-01-01")

        for ax, feature in zip(axes, features):

            ycol = f"{feature}_yearly_mean"
            mcol = f"{feature}_monthly_mean"

            # Plot yearly first to establish datetime x-axis converter.
            y = yearly[ycol]
            ax.plot(x, y, marker="o", color="purple", label="Yearly")

            if mcol in monthly.columns and "month_ts" in monthly.columns:
                y_m = monthly[mcol]
                if smoothing_window:
                    y_m = y_m.rolling(smoothing_window, center=True).mean()

                ax.plot(monthly["month_ts"], y_m, color="green", label="Monthly")

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
    # STACKED WORD-PREFIX PLOTS
    def save_word_prefix_stack_plot(
        self,
        yearly: pd.DataFrame,
        monthly: pd.DataFrame,
        output_dir: Path,
        prefix: str = "word_",
        events: dict[str, str] | None = None,
        smoothing_window: int | None = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        features = [
            c.removesuffix("_yearly_mean")
            for c in yearly.columns
            if c.endswith("_yearly_mean") and c.removesuffix("_yearly_mean").startswith(prefix)
        ]
        if not features:
            raise ValueError(f"No yearly features found with prefix '{prefix}'")

        # Ensure monthly has proper datetime month_ts before plotting.
        monthly = monthly.copy()
        if "month_ts" not in monthly.columns:
            if "year" in monthly.columns and "month" in monthly.columns:
                monthly["month_ts"] = pd.to_datetime(
                    monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2) + "-01",
                    errors="coerce",
                )
            else:
                raise ValueError("monthly DataFrame must have month_ts, or both year and month columns")
        else:
            monthly["month_ts"] = pd.to_datetime(monthly["month_ts"], errors="coerce")

        events = events or {
            "ChatGPT": "2022-11-30",
            "Delve": "2024-01-15",
        }

        event_dates = {k: pd.to_datetime(v) for k, v in events.items()}

        fig, axes = plt.subplots(
            len(features),
            1,
            figsize=(3.5, 2.0 * len(features)),
            sharex=True,
        )
        if len(features) == 1:
            axes = [axes]

        x = pd.to_datetime(yearly["year"].astype(str) + "-01-01")

        for ax, feature in zip(axes, features):
            ycol = f"{feature}_yearly_mean"
            mcol = f"{feature}_monthly_mean"

            # Plot yearly first to establish datetime x-axis converter.
            y = yearly[ycol]
            ax.plot(x, y, marker="o", linewidth=1.6, color=self.colors["yearly"], label="Yearly")

            if mcol in monthly.columns:
                y_m = monthly[mcol]
                if smoothing_window:
                    y_m = y_m.rolling(smoothing_window, center=True).mean()
                ax.plot(monthly["month_ts"], y_m, color=self.colors["monthly"], linewidth=1.0, alpha=0.8, label="Monthly")

            self._add_event_lines(ax, event_dates)
            ax.set_ylabel(self._pretty_label(feature))
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)

        axes[-1].set_xlabel("Year")
        self._format_xticks(axes[-1])

        fig.tight_layout()

        out_path = output_dir / "word_prefix_stack.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return out_path

    def save_verb_prefix_stack_plot(
        self,
        yearly: pd.DataFrame,
        monthly: pd.DataFrame,
        output_dir: Path,
        prefix: str = "verb_",
        events: dict[str, str] | None = None,
        smoothing_window: int | None = None,
    ) -> Path:
        return self.save_word_prefix_stack_plot(
            yearly=yearly,
            monthly=monthly,
            output_dir=output_dir,
            prefix=prefix,
            events=events,
            smoothing_window=smoothing_window,
        )

    def save_adjective_prefix_stack_plot(
        self,
        yearly: pd.DataFrame,
        monthly: pd.DataFrame,
        output_dir: Path,
        prefix: str = "adjective_",
        events: dict[str, str] | None = None,
        smoothing_window: int | None = None,
    ) -> Path:
        return self.save_word_prefix_stack_plot(
            yearly=yearly,
            monthly=monthly,
            output_dir=output_dir,
            prefix=prefix,
            events=events,
            smoothing_window=smoothing_window,
        )
    
    def save_punctuation_stack_plot(
        self,
        yearly: pd.DataFrame,
        monthly: pd.DataFrame,
        output_dir: Path,
        events: dict[str, str] | None = None,
        smoothing_window: int | None = None,
    ) -> Path:
        from .feature_groups import FEATURE_GROUPS
        
        features = FEATURE_GROUPS.get("punctuation", [])
        if not features:
            raise ValueError("No punctuation features found in FEATURE_GROUPS")

        # Ensure monthly has proper datetime month_ts before plotting.
        monthly = monthly.copy()
        if "month_ts" not in monthly.columns:
            if "year" in monthly.columns and "month" in monthly.columns:
                monthly["month_ts"] = pd.to_datetime(
                    monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2) + "-01",
                    errors="coerce",
                )
            else:
                raise ValueError("monthly DataFrame must have month_ts, or both year and month columns")
        else:
            monthly["month_ts"] = pd.to_datetime(monthly["month_ts"], errors="coerce")

        events = events or {
            "ChatGPT": "2022-11-30",
            "Delve": "2024-01-15",
        }

        event_dates = {k: pd.to_datetime(v) for k, v in events.items()}

        fig, axes = plt.subplots(
            len(features),
            1,
            figsize=(3.5, 2.0 * len(features)),
            sharex=True
        )
        if len(features) == 1:
            axes = [axes]

        x = pd.to_datetime(yearly["year"].astype(str) + "-01-01")

        for ax, feature in zip(axes, features):
            ycol = f"{feature}_yearly_mean"
            mcol = f"{feature}_monthly_mean"

            # Plot yearly first to establish datetime x-axis converter.
            y = yearly[ycol]
            ax.plot(x, y, marker="o", linewidth=1.6, color=self.colors["yearly"], label="Yearly")

            if mcol in monthly.columns:
                y_m = monthly[mcol]
                if smoothing_window:
                    y_m = y_m.rolling(smoothing_window, center=True).mean()
                ax.plot(monthly["month_ts"], y_m, color=self.colors["monthly"], linewidth=1.0, alpha=0.8, label="Monthly")

            self._add_event_lines(ax, event_dates)
            ax.set_ylabel(self._pretty_label(feature))
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)

        axes[-1].set_xlabel("Year")
        self._format_xticks(axes[-1])

        fig.tight_layout()

        out_path = output_dir / "punctuation_stack.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        return out_path

    def save_readability_stack_plot(
        self,
        yearly: pd.DataFrame,
        monthly: pd.DataFrame,
        output_dir: Path,
        events: dict[str, str] | None = None,
        smoothing_window: int | None = None,
    ) -> Path | None:
        from .feature_groups import FEATURE_GROUPS
        
        features = []
        for feature in FEATURE_GROUPS.get("readability", []):
            ycol = f"{feature}_yearly_mean"
            if ycol not in yearly.columns:
                continue
            if pd.to_numeric(yearly[ycol], errors="coerce").dropna().empty:
                continue
            features.append(feature)

        if not features:
            return None

        # Ensure monthly has proper datetime month_ts before plotting.
        monthly = monthly.copy()
        if "month_ts" not in monthly.columns:
            if "year" in monthly.columns and "month" in monthly.columns:
                monthly["month_ts"] = pd.to_datetime(
                    monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2) + "-01",
                    errors="coerce",
                )
            else:
                raise ValueError("monthly DataFrame must have month_ts, or both year and month columns")
        else:
            monthly["month_ts"] = pd.to_datetime(monthly["month_ts"], errors="coerce")

        events = events or {
            "ChatGPT": "2022-11-30",
            "Delve": "2024-01-15",
        }

        event_dates = {k: pd.to_datetime(v) for k, v in events.items()}

        fig, axes = plt.subplots(
            len(features),
            1,
            figsize=(3.5, 2.0 * len(features)),
            sharex=True,
        )
        if len(features) == 1:
            axes = [axes]

        x = pd.to_datetime(yearly["year"].astype(str) + "-01-01")

        for ax, feature in zip(axes, features):
            ycol = f"{feature}_yearly_mean"
            mcol = f"{feature}_monthly_mean"

            # Plot yearly first to establish datetime x-axis converter.
            y = yearly[ycol]
            ax.plot(x, y, marker="o", linewidth=1.6, color=self.colors["yearly"], label="Yearly")

            if mcol in monthly.columns:
                y_m = monthly[mcol]
                if smoothing_window:
                    y_m = y_m.rolling(smoothing_window, center=True).mean()
                ax.plot(monthly["month_ts"], y_m, color=self.colors["monthly"], linewidth=1.0, alpha=0.8, label="Monthly")

            self._add_event_lines(ax, event_dates)
            ax.set_ylabel(self._pretty_label(feature), rotation=90, labelpad=10)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)

        axes[-1].set_xlabel("Year")
        self._format_xticks(axes[-1])

        fig.tight_layout()

        out_path = output_dir / "readability_stack.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return out_path

    # =========================================================
    # PRE/POST GLOBAL DIFF
    # =========================================================
    def save_pre_post_diff_plot(self, yearly: pd.DataFrame, output_dir: Path) -> Path | None:

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
                "diff": _simple_percent_change(pre_vals.mean(), post_vals.mean())
            })

        if not rows:
            return None

        df = pd.DataFrame(rows).dropna(subset=["diff"]).sort_values("diff")

        stats_df = self.compute_all_stats(yearly)

        return save_grouped_difference_plot(
            df,
            output_dir / "pre_post_diff.png",
            "feature",
            "diff",
            "change (%)",
            self.label_map,
            stats_df=stats_df,
            significant_only=False,
        )

    # =========================================================
    # GROUPED DIFFS
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

        stats_df = self.compute_all_stats(yearly)

        pre = yearly[yearly["year"] <= pre_cut]
        post = yearly[yearly["year"] >= post_cut]

        for group_name, features in groups.items():

            rows = []
            valid_cols = []

            for f in features:
                if is_group_total_feature(f):
                    continue

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
                    "diff": _simple_percent_change(pre_vals.mean(), post_vals.mean())
                })

            if not rows:
                continue

            df = pd.DataFrame(rows).dropna(subset=["diff"])

            if df.empty:
                continue

            df = df.sort_values("diff")

            out_path = output_dir / f"{group_name}_diff.png"

            saved_path = save_grouped_difference_plot(
                df,
                out_path,
                "feature",
                "diff",
                "change (%)",
                self.label_map,
                stats_df=stats_df,
                annotation_mode=None,
                significant_only=False,
            )

            if saved_path is not None:
                outputs[group_name] = saved_path

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
            "Delve paper": "2024-01-15",
        }

        event_dates = {k: pd.to_datetime(v) for k, v in events.items()}

        month_ts = None
        if "month_ts" in monthly.columns:
            month_ts = pd.to_datetime(monthly["month_ts"], errors="coerce")

        for group_name, spec in group_specs.items():
            label = str(spec.get("label", group_name))
            rate_feature = str(spec.get("rate_feature", "")).strip()
            if not rate_feature or rate_feature not in yearly.columns:
                continue

            yearly_col = f"{rate_feature}_yearly_mean"
            monthly_col = f"{rate_feature}_monthly_mean"

            fig, ax = plt.subplots(figsize=(3.5, 2.3))

            if monthly_col in monthly.columns and month_ts is not None:
                y_m = monthly[monthly_col]
                if smoothing_window:
                    y_m = y_m.rolling(smoothing_window, center=True).mean()
                ax.plot(month_ts, y_m, color=self.colors["monthly"], linewidth=1.2, alpha=1.0, label="Monthly mean")

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
                ax.text(d, top_y, event_label, rotation=90, va="bottom", fontsize=10, alpha=0.8)

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

        month_ts = None
        if "month_ts" in monthly.columns:
            month_ts = pd.to_datetime(monthly["month_ts"], errors="coerce")

        for yearly_column in plot_columns:
            feature = yearly_column.removesuffix("_yearly_mean")
            monthly_col = f"{feature}_monthly_mean"

            fig, ax = plt.subplots(figsize=(3.5, 2.3))

            if monthly_col in monthly.columns and month_ts is not None:
                ax.plot(month_ts, monthly[monthly_col], color=self.colors["monthly"], linewidth=1.0, alpha=1.0, label="Monthly mean")

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
    def save_dependency_distribution_plot(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        events: dict[str, str] | None = None,
        top_n: int = 10,
        pre_cut: int = 2022,
        post_cut: int = 2023,
    ) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)

        if "dependency_distribution" not in df.columns:
            raise ValueError("Expected 'dependency_distribution' column in dataframe")

        dep_df = df["dependency_distribution"].apply(pd.Series).fillna(0)
        dep_df["year"] = df["year"]

        dep_yearly = dep_df.groupby("year").sum(numeric_only=True)
        if dep_yearly.empty:
            raise ValueError("No dependency distribution rows found")

        # Compute proportions for each dependency role.
        dep_yearly_prop = dep_yearly.div(dep_yearly.sum(axis=1), axis=0).fillna(0.0)

        pre = dep_yearly.loc[dep_yearly.index <= pre_cut].sum(numeric_only=True)
        post = dep_yearly.loc[dep_yearly.index >= post_cut].sum(numeric_only=True)

        if pre.sum() == 0 or post.sum() == 0:
            return []

        pre_prop = pre / pre.sum()
        post_prop = post / post.sum()

        diff_df = pd.DataFrame(
            {
                "feature": dep_yearly.columns,
                "pre_prop": pre_prop.values,
                "post_prop": post_prop.values,
                "diff_pct": 100.0 * (post_prop.values - pre_prop.values),
            }
        )
        diff_df = diff_df.dropna(subset=["diff_pct"]).copy()
        diff_df["abs_diff"] = diff_df["diff_pct"].abs()
        diff_df = diff_df.sort_values("abs_diff", ascending=False).head(top_n)
        diff_df = diff_df.sort_values("diff_pct")
        if diff_df.empty:
            return []

        # Save diff plot
        diff_fig, diff_ax = plt.subplots(figsize=(5, 1 * len(diff_df) + 0.2))
        colors = ["#943F8B" if v < 0 else "#54A066" for v in diff_df["diff_pct"]]
        diff_ax.barh(diff_df["feature"], diff_df["diff_pct"], color=colors)
        diff_ax.axvline(0, color="black", linewidth=1)
        diff_ax.set_xlabel("Change in role proportion (percentage points)")
        # diff_ax.set_title("Top dependency role proportion changes after 2023")
        diff_ax.grid(axis="x", alpha=0.3)
        self._format_xticks(diff_ax)
        diff_out_path = output_dir / "dependency_distribution_diff.png"
        diff_fig.tight_layout()
        diff_fig.savefig(diff_out_path, dpi=150, bbox_inches="tight")
        plt.close(diff_fig)

        # Save yearly role trend plot for the top changed roles
        top_roles = diff_df["feature"].tolist()
        plot_yearly = dep_yearly_prop[top_roles].copy()
        plot_yearly = plot_yearly.sort_index()

        year_ts = pd.to_datetime(plot_yearly.index.astype(str) + "-01-01")

        trend_fig, trend_ax = plt.subplots(figsize=(4.5, 3.5 + 0.2))
        for role in top_roles:
            trend_ax.plot(year_ts, plot_yearly[role], marker="o", linewidth=1, label=role)

        if events:
            event_dates = {k: pd.to_datetime(v) for k, v in events.items()}
            self._add_event_lines(trend_ax, event_dates)
            top_y = trend_ax.get_ylim()[1]
            for label, d in event_dates.items():
                trend_ax.text(d, top_y, label, rotation=0, va="bottom", fontsize=10, alpha=0.8)

        trend_ax.set_xlabel("Year")
        trend_ax.set_ylabel("Dependency role proportion")
        # trend_ax.set_title("Dependency role trends over years")
        trend_ax.grid(alpha=0.3)
        trend_ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        self._format_xticks(trend_ax)
        trend_out_path = output_dir / "dependency_distribution_trends.png"
        trend_fig.tight_layout()
        trend_fig.savefig(trend_out_path, dpi=150, bbox_inches="tight")
        plt.close(trend_fig)

        return [diff_out_path, trend_out_path]


# =========================================================
# PLOTTING FUNCTION
# =========================================================
def save_grouped_difference_plot(
    df: pd.DataFrame,
    output_path: Path,
    feature_column: str,
    diff_column: str,
    xlabel: str,
    label_map: dict[str, str] | None = None,
    top_n: int = 30,
    stats_df: pd.DataFrame | None = None,
    annotation_mode: str = None,  # none | p | d | p+d
    significant_only: bool = False,
    title: str | None = None,
) -> Path | None:
    label_map = label_map or {}

    def _pretty_diff_label(feature: str) -> str:
        mapped = label_map.get(feature)

        if mapped:
            return mapped

        if feature.endswith("_TOTAL"):
            group = feature.removesuffix("_TOTAL").replace("_", " ")
            return f"{group} total"

        cleaned = feature
        cleaned = cleaned.removesuffix("_per_1k_words")
        cleaned = cleaned.removesuffix("_total")

        lexical_prefixes = ("word_", "verb_", "adjective_", "phrase_")
        marker_prefixes = (
            "sequential_marker_",
            "causal_marker_",
            "contrast_marker_",
            "emphasis_marker_",
            "summary_marker_",
        )

        for prefix in lexical_prefixes + marker_prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                return f"`{cleaned.replace('_', ' ')}`"

        return cleaned.replace("_", " ")

    df = df.dropna(subset=[feature_column, diff_column]).copy()

    # =====================================================
    # MERGE STATS
    # =====================================================
    if (
        stats_df is not None
        and not stats_df.empty
        and "feature" in stats_df.columns
    ):

        stats_keep = [
            c
            for c in [
                "feature",
                "p_value",
                "p_adj",
                "cohens_d",
            ]
            if c in stats_df.columns
        ]

        df = df.merge(
            stats_df[stats_keep],
            on="feature",
            how="left",
        )

        # optional significance filtering
        if significant_only and "p_adj" in df.columns:
            df = df[df["p_adj"] < 0.05]

        df = (
            df
            .sort_values(
                diff_column,
                key=lambda s: s.abs(),
                ascending=False,
            )
            .head(top_n)
        )

    df = df.sort_values(diff_column)

    if df.empty:
        return None

    df["label"] = (
        df[feature_column]
        .astype(str)
        .map(_pretty_diff_label)
    )

    is_percent_plot = "%" in xlabel or "percent" in xlabel.lower()
    plot_column = "_plot_diff"
    df[plot_column] = pd.to_numeric(df[diff_column], errors="coerce")
    if is_percent_plot:
        df[plot_column] = df[plot_column].clip(lower=-100.0, upper=100.0)
    clipped_percent_column = "_is_clipped_percent_diff"
    df[clipped_percent_column] = False
    if is_percent_plot:
        df[clipped_percent_column] = pd.to_numeric(df[diff_column], errors="coerce").abs() >= 100.0

    # =====================================================
    # FIGURE SIZE
    # =====================================================
    bar_height_inches = 0.15

    fig_height = (
        len(df) * bar_height_inches
        + 1.0
    )

    fig, ax = plt.subplots(
        figsize=(
            4.5,
            max(1, fig_height),
        )
    )

    colors = [
        "#943F8B" if v < 0
        else "#54A066"
        for v in df[diff_column]
    ]

    bars = ax.barh(
        df["label"],
        df[plot_column],
        color=colors,
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1,
    )

    # =====================================================
    # ANNOTATIONS
    # =====================================================
    x_range = abs(
        df[plot_column]
    ).max()

    offset = max(
        x_range * 0.03,
        1.0,
    )

    for bar, (_, row) in zip(
        bars,
        df.iterrows(),
    ):

        pieces = []
        raw_value = float(row.get(diff_column, np.nan))

        is_clipped_percent = bool(row.get(clipped_percent_column, False))

        if is_clipped_percent:
            pieces.append(f"{raw_value:+.1f}%")

        p = row.get(
            "p_adj",
            np.nan,
        )

        d = row.get(
            "cohens_d",
            np.nan,
        )

        if annotation_mode in ("p", "p+d"):

            if pd.notna(p):

                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = "ns"

                pieces.append(
                    f"{sig} p={p:.3f}"
                )

        if (
            annotation_mode
            in ("d", "p+d")
            and pd.notna(d)
        ):
            pieces.append(
                f"d={d:.2f}"
            )

        if not pieces:
            continue

        label = " | ".join(
            pieces
        )

        width = bar.get_width()

        y = (
            bar.get_y()
            + bar.get_height()
            / 2
        )

        inset = max(abs(width) * 0.2, offset)

        if is_clipped_percent:
            if width >= 0:
                x = inset
                ha = "left"
            else:
                x = -inset
                ha = "right"
        elif width >= 0:
            x = width - inset
            ha = "right"
        else:
            x = width + inset
            ha = "left"

        ax.text(
            x,
            y,
            label,
            va="center",
            ha=ha,
            fontsize=7,
            alpha=0.85,
            color="white" if is_clipped_percent else "black",
        )

    if title:
        ax.set_title(title)

    if is_percent_plot:
        ax.set_xlim(-100, 100)
        ax.set_xticks([-100, -50, 0, 50, 100])

    ax.grid(
        axis="x",
        alpha=0.3,
    )

    ax.set_xlabel(
        xlabel
    )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_path
