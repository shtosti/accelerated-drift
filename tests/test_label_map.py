from __future__ import annotations

import unittest

from not_an_llm.analysis.label_map import pretty_feature_label


class FeatureLabelTests(unittest.TestCase):
    def test_individual_lexical_items_use_mathtext_italics(self) -> None:
        self.assertEqual(pretty_feature_label("verb_delve_per_1k_words"), r"$\it{delve}$")
        self.assertEqual(pretty_feature_label("verb_align_per_1k_words"), r"$\it{align}$")
        self.assertEqual(pretty_feature_label("word_across_per_1k_words"), r"$\it{across}$")

    def test_aggregate_groups_remain_unquoted(self) -> None:
        self.assertEqual(pretty_feature_label("marker_words_total_per_1k_words"), "words")
        self.assertEqual(pretty_feature_label("hedge_ratio"), "hedge terms")


if __name__ == "__main__":
    unittest.main()
