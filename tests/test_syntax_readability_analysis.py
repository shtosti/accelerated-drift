from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_syntax_readability_abstracts",
    ROOT / "scripts" / "analyze_syntax_readability_abstracts.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SyntaxReadabilityAnalysisTests(unittest.TestCase):
    def test_build_model_frame_adds_roles_bigrams_and_aligned_outcomes(self) -> None:
        documents = pd.DataFrame(
            {
                "document_key": ["d1", "d2", "d3", "d4"],
                "dataset": ["a", "a", "b", "b"],
                "year": [2021, 2023, 2021, 2023],
                "automated_readability_index": [8.0, 10.0, 7.0, 11.0],
                "flesch_kincaid_grade": [7.0, 9.0, 6.0, 10.0],
                "flesch_reading_ease": [70.0, 60.0, 75.0, 55.0],
                "dependency_entropy": [1.0, 1.2, 0.9, 1.3],
                "dependency_distribution": [
                    {"det": 2, "pobj": 1},
                    {"det": 1, "pobj": 2},
                    {"det": 3},
                    {"pobj": 3},
                ],
            }
        )
        bigrams = {
            "d1": {"prep->pobj": 0.4},
            "d2": {"prep->pobj": 0.2, "dobj->amod": 0.3},
            "d3": {"prep->pobj": 0.5},
            "d4": {"dobj->amod": 0.4},
        }
        frame, syntax = MODULE.build_model_frame(documents, bigram_maps=bigrams, top_bigrams=2)
        self.assertIn("dep_role__det", syntax)
        self.assertIn("dep_bigram__prep->pobj", syntax)
        self.assertTrue(frame["readability_composite"].notna().all())
        self.assertGreater(frame.loc[1, "ari_complexity"], frame.loc[0, "ari_complexity"])

    def test_alignment_score_combines_association_and_temporal_change(self) -> None:
        coefficients = pd.DataFrame(
            {
                "model_spec": ["total_association", "total_association"],
                "outcome": ["readability_composite", "readability_composite"],
                "feature": ["dep_bigram__a->b", "dep_role__det"],
                "standardized_coefficient": [0.5, -0.25],
            }
        )
        syntax_its = pd.DataFrame(
            {
                "dataset": ["x", "y", "x", "y"],
                "feature": ["dep_bigram__a->b", "dep_bigram__a->b", "dep_role__det", "dep_role__det"],
                "standardized_slope_change_per_year": [0.4, 0.6, -0.2, -0.4],
            }
        )
        result = MODULE.build_alignment_table(coefficients, syntax_its).set_index("feature")
        self.assertAlmostEqual(result.loc["dep_bigram__a->b", "alignment_score"], 0.25)
        self.assertAlmostEqual(result.loc["dep_role__det", "alignment_score"], 0.075)
        self.assertTrue(result["complexity_aligned"].all())

    def test_grouped_elastic_net_returns_stability_columns(self) -> None:
        rng = np.random.default_rng(7)
        years = np.repeat(np.arange(2018, 2026), 18)
        feature = rng.normal(size=len(years))
        frame = pd.DataFrame(
            {
                "dataset": np.where(np.arange(len(years)) % 2, "a", "b"),
                "year": years,
                "word_count": rng.integers(80, 250, len(years)),
                "dependency_entropy": feature,
                "readability_composite": 0.7 * feature + rng.normal(scale=0.3, size=len(years)),
            }
        )
        result = MODULE.fit_syntax_association(
            frame,
            ["dependency_entropy"],
            outcome="readability_composite",
            model_spec="total_association",
            extra_controls=[],
            bootstrap=4,
            permutation_repeats=1,
        )
        self.assertEqual(len(result), 1)
        self.assertGreater(result.loc[0, "standardized_coefficient"], 0)
        self.assertIn("grouped_permutation_mse_increase", result.columns)
        self.assertIn("bootstrap_selection_frequency", result.columns)


if __name__ == "__main__":
    unittest.main()
