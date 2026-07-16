from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "decompose_readability_abstracts",
    ROOT / "scripts" / "decompose_readability_abstracts.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AriDecompositionTests(unittest.TestCase):
    def test_monthly_components_reconstruct_ari(self) -> None:
        monthly = pd.DataFrame(
            {
                "month_ts": ["2020-01-01", "2020-02-01"],
                "paper_count": [10, 12],
                MODULE.ARI_COLUMN: [12.0, 13.5],
                MODULE.WORDS_PER_SENTENCE_COLUMN: [20.0, 22.0],
            }
        )
        result = MODULE.build_ari_components(monthly)
        reconstructed = (
            result[f"{MODULE.ARI_SENTENCE_COMPONENT}_monthly_mean"]
            + result[f"{MODULE.ARI_CHARACTER_COMPONENT}_monthly_mean"]
            + MODULE.ARI_INTERCEPT
        )
        pd.testing.assert_series_equal(
            reconstructed,
            result[f"{MODULE.ARI_TOTAL}_monthly_mean"],
            check_names=False,
        )

    def test_monthly_components_reconstruct_fkgl(self) -> None:
        monthly = pd.DataFrame(
            {
                "month_ts": ["2020-01-01", "2020-02-01"],
                "paper_count": [10, 12],
                MODULE.ARI_COLUMN: [12.0, 13.5],
                MODULE.FKGL_COLUMN: [10.0, 11.0],
                MODULE.WORDS_PER_SENTENCE_COLUMN: [20.0, 22.0],
                MODULE.SYLLABLES_PER_WORD_COLUMN: [1.4, 1.5],
            }
        )
        result = MODULE.build_readability_components(monthly)
        reconstructed = (
            result[f"{MODULE.FKGL_SENTENCE_COMPONENT}_monthly_mean"]
            + result[f"{MODULE.FKGL_SYLLABLE_COMPONENT}_monthly_mean"]
            + result[f"{MODULE.FKGL_RECONCILIATION_COMPONENT}_monthly_mean"]
            + MODULE.FKGL_INTERCEPT
        )
        pd.testing.assert_series_equal(
            reconstructed,
            result[f"{MODULE.FKGL_TOTAL}_monthly_mean"],
            check_names=False,
        )

    def test_summary_component_slopes_reconstruct_total(self) -> None:
        stats = pd.DataFrame(
            {
                "feature": [
                    MODULE.ARI_TOTAL,
                    MODULE.ARI_SENTENCE_COMPONENT,
                    MODULE.ARI_CHARACTER_COMPONENT,
                ],
                "slope_change_per_year": [0.8, 0.1, 0.7],
            }
        )
        summary = MODULE.summarize_decomposition("example_abstracts", stats)
        self.assertAlmostEqual(float(summary["reconstruction_error"]), 0.0)
        self.assertEqual(summary["dominant_component"], "implied characters per word")

    def test_fkgl_summary_component_slopes_reconstruct_total(self) -> None:
        stats = pd.DataFrame(
            {
                "feature": [
                    MODULE.FKGL_TOTAL,
                    MODULE.FKGL_SENTENCE_COMPONENT,
                    MODULE.FKGL_SYLLABLE_COMPONENT,
                    MODULE.FKGL_RECONCILIATION_COMPONENT,
                ],
                "slope_change_per_year": [0.6, 0.1, 0.45, 0.05],
            }
        )
        summary = MODULE.summarize_fkgl_decomposition("example_abstracts", stats)
        self.assertAlmostEqual(float(summary["reconstruction_error"]), 0.0)
        self.assertEqual(summary["dominant_component"], "syllables per word")


if __name__ == "__main__":
    unittest.main()
