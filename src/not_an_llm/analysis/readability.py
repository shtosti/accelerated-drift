from __future__ import annotations

import re
import textstat
import pandas as pd


class ReadabilityAnalyzer:
    """Compute readability signals for trend analysis."""

    def __init__(self, metrics: list[str] | None = None) -> None:
        default_metrics = [
            "avg_words_per_sentence",
            "avg_syllables_per_word",
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "dale_chall",
        ]
        self.metrics = metrics or default_metrics

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        results = []
        for row in df.itertuples(index=False):
            text = str(getattr(row, "text_clean", "") or "")
            word_count = int(getattr(row, "word_count", 0) or 0)
            sentence_count = int(getattr(row, "sentence_count", 0) or 0)

            if sentence_count <= 0:
                sentence_count = self._sentence_count(text)
            sentence_count = max(1, sentence_count)

            safe_word_count = max(1, word_count)
            total_syllables = textstat.syllable_count(text)

            row_result = {
                "avg_words_per_sentence": word_count / sentence_count if word_count > 0 else 0.0,
                "avg_syllables_per_word": total_syllables / safe_word_count if word_count > 0 else 0.0,
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "dale_chall": textstat.dale_chall_readability_score(text),
            }
            results.append(row_result)

        metrics_df = pd.DataFrame(results)

        # attach back
        for col in metrics_df.columns:
            df[col] = metrics_df[col]

        return df

    @staticmethod
    def _sentence_count(text: str) -> int:
        if not text:
            return 0
        parts = re.split(r"[.!?]+", text)
        return sum(1 for p in parts if p.strip())