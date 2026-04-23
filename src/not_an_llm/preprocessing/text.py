from __future__ import annotations

import re
import spacy
import pandas as pd


class TextPreprocessor:
    """Normalize and structure raw title/abstract text for downstream analysis."""

    def __init__(self, *, keep_case: bool = False) -> None:
        self.keep_case = keep_case
        self._nlp = spacy.load(
            "en_core_web_sm",
            disable=["ner"]  # keep tagger + lemmatizer
        )

    def preprocess_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        df["title"] = (
            df.get("title", "")
            .fillna("")
            .astype(str)
            .apply(self._normalize_whitespace)
        )
        df["abstract"] = (
            df.get("abstract", "")
            .fillna("")
            .astype(str)
            .apply(self._normalize_whitespace)
        )

        text_raw = (df["title"].str.strip() + " " + df["abstract"].str.strip()).str.strip()
        df["text_raw"] = text_raw
        df["text_clean"] = text_raw.apply(self.normalize_text)

        lemma_sequences = self._lemmatize_text_series(df["text_clean"].tolist())
        df["text_lemma"] = lemma_sequences

        df["year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")
        df["word_count"] = df["text_clean"].str.split().str.len().fillna(0).astype(int)
        df["sentence_count"] = df["text_clean"].apply(self._sentence_count)

        return df

    def _lemmatize_text_series(self, texts: list[str]) -> list[str]:
        lemma_sequences: list[str] = []
        for doc in self._nlp.pipe(texts, batch_size=128):
            lemmas = [
                token.lemma_.lower() if not self.keep_case else token.lemma_
                for token in doc
                if token.is_alpha
            ]
            lemma_sequences.append(" ".join(lemma for lemma in lemmas if lemma))

        return lemma_sequences

    def normalize_text(self, text: str) -> str:
        value = self._normalize_whitespace(text)
        value = value.replace("\u2014", " -- ")
        value = value.replace("\u2013", " - ")
        value = re.sub(r"\s+", " ", value).strip()

        if not self.keep_case:
            value = value.lower()

        return value

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        value = text or ""
        value = re.sub(r"[\t\r\n]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    @staticmethod
    def _sentence_count(text: str) -> int:
        if not text:
            return 0

        parts = re.split(r"[.!?]+", text)
        count = sum(1 for part in parts if part.strip())
        return count


