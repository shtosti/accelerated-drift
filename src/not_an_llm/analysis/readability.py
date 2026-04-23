from __future__ import annotations

import re
import textstat
import pandas as pd
import spacy


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

        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        texts = df["text_clean"].fillna("").astype(str).tolist()

        # 🚀 batch processing (FAST)
        docs = list(self.nlp.pipe(texts, batch_size=128))

        results = []

        for doc in docs:
            text = doc.text

            # core readability
            flesch_reading_ease = textstat.flesch_reading_ease(text)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
            dale_chall = textstat.dale_chall_readability_score(text)

            # token-based stats (spaCy)
            words = [t for t in doc if t.is_alpha]
            word_count = len(words)

            sentence_count = len(list(doc.sents)) if doc.has_annotation("SENT_START") else self._sentence_count(text)
            sentence_count = max(1, sentence_count)

            total_syllables = textstat.syllable_count(text)

            if word_count == 0:
                results.append({
                    "avg_words_per_sentence": 0.0,
                    "avg_syllables_per_word": 0.0,
                    "flesch_reading_ease": flesch_reading_ease,
                    "flesch_kincaid_grade": flesch_kincaid_grade,
                    "dale_chall": dale_chall,
                })
                continue

            results.append({
                "avg_words_per_sentence": word_count / sentence_count,
                "avg_syllables_per_word": total_syllables / word_count,
                "flesch_reading_ease": flesch_reading_ease,
                "flesch_kincaid_grade": flesch_kincaid_grade,
                "dale_chall": dale_chall,
            })

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