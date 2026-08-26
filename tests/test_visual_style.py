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
    GREEN,
    HATCHES,
    INCREASE_COLOR,
    INCREASE_HATCH,
    ORCHID,
)


ROOT = Path(__file__).resolve().parents[1]
PLOTTING_FILES = [
    ROOT / "scripts" / "analyze_syntax_readability_abstracts.py",
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
    def test_primary_colors_match_previous_manuscript(self) -> None:
        self.assertEqual(GREEN.upper(), "#54A066")
        self.assertEqual(ORCHID.upper(), "#963E8D")

    def test_hatches_are_varied_and_not_diagonal_only(self) -> None:
        self.assertIn("..", HATCHES)
        self.assertIn("xx", HATCHES)
        self.assertIn("--", HATCHES)
        self.assertIn("///", HATCHES)
        self.assertNotEqual("..", "///")

    def test_abstract_corpus_palette_has_no_teal_sky_blue_or_grey(self) -> None:
        abstract_colors = {
            color.upper() for dataset, color in DATASET_COLORS.items()
            if dataset.endswith("_abstracts")
        }
        self.assertTrue({GREEN.upper(), ORCHID.upper()}.issubset(abstract_colors))
        self.assertNotIn("#00876C", abstract_colors)
        self.assertNotIn("#56B4E9", abstract_colors)
        self.assertNotIn("#666666", abstract_colors)

    def test_each_corpus_keeps_one_color_across_scopes(self) -> None:
        for corpus in ("arxiv_ai", "arxiv_qbio", "arxiv_stat", "medarxiv"):
            self.assertEqual(
                DATASET_COLORS[f"{corpus}_abstracts"],
                DATASET_COLORS[f"{corpus}_titles"],
            )

    def test_positive_effect_and_ai_corpus_share_exact_green(self) -> None:
        self.assertEqual(INCREASE_COLOR, GREEN)
        self.assertEqual(DATASET_COLORS["arxiv_ai_abstracts"], GREEN)

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
