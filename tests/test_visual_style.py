from __future__ import annotations

from pathlib import Path
import re
import unittest

from not_an_llm.analysis.visual_style import (
    CATEGORICAL_COLORS,
    DATASET_COLORS,
    DATASET_HATCHES,
    DATASET_MARKERS,
    DECREASE_HATCH,
    INCREASE_HATCH,
)


ROOT = Path(__file__).resolve().parents[1]
PLOTTING_FILES = [
    ROOT / "scripts" / "decompose_readability_abstracts.py",
    ROOT / "scripts" / "compare_corpus_trends.py",
    ROOT / "scripts" / "compare_title_abstract_trends.py",
    ROOT / "src" / "not_an_llm" / "analysis" / "interrupted_time_series.py",
    ROOT / "src" / "not_an_llm" / "analysis" / "topic_modeling" / "comparison.py",
    ROOT / "src" / "not_an_llm" / "analysis" / "topic_modeling" / "plots.py",
    ROOT / "src" / "not_an_llm" / "analysis" / "trends.py",
    ROOT / "src" / "not_an_llm" / "pipelines" / "additional_analysis.py",
]


class VisualStyleTests(unittest.TestCase):
    def test_dataset_encodings_cover_the_same_datasets(self) -> None:
        self.assertEqual(set(DATASET_COLORS), set(DATASET_MARKERS))
        self.assertEqual(set(DATASET_COLORS), set(DATASET_HATCHES))

    def test_primary_categorical_colors_are_unique(self) -> None:
        self.assertEqual(len(CATEGORICAL_COLORS), len(set(CATEGORICAL_COLORS)))

    def test_signs_have_redundant_hatch_encoding(self) -> None:
        self.assertNotEqual(INCREASE_HATCH, DECREASE_HATCH)

    def test_plotting_modules_do_not_define_local_hex_colors(self) -> None:
        pattern = re.compile(r"#[0-9A-Fa-f]{6}\b")
        offenders = [path for path in PLOTTING_FILES if pattern.search(path.read_text(encoding="utf-8"))]
        self.assertEqual([], offenders, "Define project colors only in visual_style.py")


if __name__ == "__main__":
    unittest.main()
