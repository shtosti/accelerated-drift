from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_title_abstract_trends",
    ROOT / "scripts" / "compare_title_abstract_trends.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TitleAbstractSummaryTests(unittest.TestCase):
    def test_summary_uses_only_complete_finite_pairs(self) -> None:
        frame = pd.DataFrame(
            {
                "corpus": ["demo"] * 4,
                "standardized_slope_change_per_year_abstract": [1.0, -2.0, np.nan, 4.0],
                "standardized_slope_change_per_year_title": [2.0, 1.0, 3.0, np.nan],
                "standardized_slope_difference": [1.0, 3.0, np.nan, np.nan],
                "same_direction": pd.Series([True, False, pd.NA, pd.NA], dtype="boolean"),
            }
        )
        row = MODULE.summary_row("demo", "all", frame)
        self.assertEqual(row["n_configured_features"], 4)
        self.assertEqual(row["n_complete_pairs"], 2)
        self.assertEqual(row["n_inestimable_abstract"], 1)
        self.assertEqual(row["n_inestimable_title"], 1)
        self.assertEqual(row["same_direction_share"], 0.5)
        self.assertEqual(row["title_larger_abs_share"], 0.5)


if __name__ == "__main__":
    unittest.main()
