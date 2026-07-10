from __future__ import annotations

from unittest import TestCase

from scripts.create_micro_dataset import select_micro_records


class MicroDatasetTests(TestCase):
    def test_selection_is_deterministic_and_balanced(self):
        records = [
            {"paperId": f"{year}-{index}", "year": year}
            for year in (2021, 2022)
            for index in range(5)
        ]

        first = select_micro_records(
            records,
            years=[2021, 2022],
            papers_per_year=2,
            seed="test",
        )
        second = select_micro_records(
            list(reversed(records)),
            years=[2021, 2022],
            papers_per_year=2,
            seed="test",
        )

        self.assertEqual(first, second)
        self.assertEqual(
            [record["year"] for record in first],
            [2021, 2021, 2022, 2022],
        )
