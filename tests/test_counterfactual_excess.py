from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from not_an_llm.analysis.interrupted_time_series import (
    ITSConfig,
    compute_first_post_year_counterfactual_excess,
    compute_first_two_year_counterfactual_excess,
)


class StrictPreInterventionCounterfactualTests(unittest.TestCase):
    def setUp(self) -> None:
        dates = pd.date_range("2020-01-01", "2024-12-01", freq="MS")
        time = np.arange(len(dates), dtype=float)
        pretrend = 10.0 + 0.2 * time + 0.05 * np.sin(time)
        post_excess = np.where(dates.year == 2023, 3.0, np.where(dates.year == 2024, -1.0, 0.0))
        self.monthly = pd.DataFrame(
            {
                "month_ts": dates,
                "paper_count": np.linspace(100.0, 200.0, len(dates)),
                "synthetic_monthly_mean": pretrend + post_excess,
            }
        )
        self.config = ITSConfig(min_pre_months=12, min_post_months=6, hac_lags=3)

    def test_first_year_counterfactual_trend_is_unchanged_by_post_values(self) -> None:
        baseline = compute_first_post_year_counterfactual_excess(
            self.monthly, ["synthetic"], config=self.config
        ).iloc[0]
        changed = self.monthly.copy()
        changed.loc[changed["month_ts"].dt.year == 2023, "synthetic_monthly_mean"] += np.linspace(0, 100, 12)
        perturbed = compute_first_post_year_counterfactual_excess(
            changed, ["synthetic"], config=self.config
        ).iloc[0]

        self.assertEqual(baseline["model"], "strict_pretrend_counterfactual_wls_hac")
        self.assertAlmostEqual(baseline["pre_slope_per_month"], perturbed["pre_slope_per_month"], places=12)
        self.assertAlmostEqual(
            baseline["target_year_counterfactual_mean"],
            perturbed["target_year_counterfactual_mean"],
            places=12,
        )
        self.assertNotAlmostEqual(
            baseline["first_post_year_excess"], perturbed["first_post_year_excess"], places=6
        )

    def test_two_year_counterfactual_trend_is_unchanged_by_post_values(self) -> None:
        baseline = compute_first_two_year_counterfactual_excess(
            self.monthly, ["synthetic"], config=self.config
        ).iloc[0]
        changed = self.monthly.copy()
        changed.loc[changed["month_ts"].dt.year.isin([2023, 2024]), "synthetic_monthly_mean"] *= 4.0
        perturbed = compute_first_two_year_counterfactual_excess(
            changed, ["synthetic"], config=self.config
        ).iloc[0]

        self.assertAlmostEqual(baseline["pre_slope_per_month"], perturbed["pre_slope_per_month"], places=12)
        self.assertAlmostEqual(
            baseline["target_period_counterfactual_mean"],
            perturbed["target_period_counterfactual_mean"],
            places=12,
        )
        self.assertNotAlmostEqual(
            baseline["first_two_year_excess"], perturbed["first_two_year_excess"], places=6
        )


if __name__ == "__main__":
    unittest.main()
