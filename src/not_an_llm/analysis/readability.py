from __future__ import annotations

import math
import re
import textstat
import pandas as pd


class ReadabilityAnalyzer:
    """Compute readability signals for trend analysis."""

    _word_pattern = re.compile(r"[a-zA-Z]+")

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        readability = df["text_clean"].fillna("").astype(str).apply(self._compute_readability_metrics)
        metrics = pd.DataFrame(readability.tolist(), index=df.index)

        for column in metrics.columns:
            df[column] = metrics[column]

        return df

    def _compute_readability_metrics(self, text: str) -> dict[str, float]:
        # Use textstat functions to compute readability metrics
        flesch_reading_ease = textstat.flesch_reading_ease(text)
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
        dale_chall = textstat.dale_chall_readability_score(text)
        gunning_fog = textstat.gunning_fog(text)
        smog_index = textstat.smog_index(text)

        # Compute additional metrics
        words = self._word_pattern.findall(text)
        word_count = len(words)
        sentence_count = max(1, self._sentence_count(text))

        if word_count == 0:
            return {
                "avg_words_per_sentence": 0.0,
                "avg_syllables_per_word": 0.0,
                "flesch_reading_ease": 0.0,
                "flesch_kincaid_grade": 0.0,
                "dale_chall": 0.0
            }

        syllable_counts = [self._count_syllables(word) for word in words]
        syllables_total = sum(syllable_counts)
        polysyllable_count = sum(1 for count in syllable_counts if count >= 3)

        avg_words_per_sentence = word_count / sentence_count
        avg_syllables_per_word = syllables_total / word_count
        complex_word_ratio = polysyllable_count / word_count


        return {
            "avg_words_per_sentence": avg_words_per_sentence,
            "avg_syllables_per_word": avg_syllables_per_word,
            "flesch_reading_ease": flesch_reading_ease,
            "flesch_kincaid_grade": flesch_kincaid_grade,
            "dale_chall": dale_chall,
        }

    @staticmethod
    def _sentence_count(text: str) -> int:
        if not text:
            return 0
        parts = re.split(r"[.!?]+", text)
        return sum(1 for part in parts if part.strip())

    @staticmethod
    def _count_syllables(word: str) -> int:
        lower = word.lower()
        if not lower:
            return 1

        lower = re.sub(r"[^a-z]", "", lower)
        if not lower:
            return 1

        vowels = "aeiouy"
        count = 0
        prev_vowel = False

        for char in lower:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        if lower.endswith("e") and count > 1:
            count -= 1

        return max(1, count)
