from __future__ import annotations

import re
import pandas as pd
import spacy


class TextPreprocessor:
    """
    Normalize and structure raw title/abstract text.

    Uses spaCy as the single source of truth for:
    - tokenization
    - lemmatization
    - word count
    - sentence count
    """

    def __init__(self, *, keep_case: bool = False) -> None:
        self.keep_case = keep_case
        self.nlp = self._load_nlp()

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
        docs = list(self.nlp.pipe(df["text_clean"].tolist(), batch_size=128, n_process=16))
        df["doc"] = docs
        df["text_lemma"] = self._extract_lemmas(docs)
        df["word_count"] = self._word_counts(docs)
        df["sentence_count"] = self._sentence_counts(docs)
        df["year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")

        return df

    def normalize_text(self, text: str) -> str:
        text = self._normalize_whitespace(text)
        text = text.replace("\u2014", " -- ")
        text = text.replace("\u2013", " - ")
        text = " ".join(text.split())

        if not self.keep_case:
            text = text.lower()

        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        return " ".join((text or "").split())

    @staticmethod
    def _extract_lemmas(docs) -> list[str]:
        return [
            " ".join(
                token.lemma_.lower()
                for token in doc
                if token.is_alpha
            )
            for doc in docs
        ]

    @staticmethod
    def _word_counts(docs) -> list[int]:
        return [sum(1 for token in doc if token.is_alpha) for doc in docs]

    @staticmethod
    def _sentence_counts(docs) -> list[int]:
        return [
            sum(1 for _ in doc.sents)
            for doc in docs
        ]

    @staticmethod
    def _load_nlp():
        try:
            nlp = spacy.load(
                "en_core_web_sm",
                disable=["ner"]  # DO NOT disable parser if you want sentences
            )

            # Ensure sentencizer exists if parser is disabled internally
            if "sentencizer" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer", first=True)

            return nlp

        except Exception:
            nlp = spacy.blank("en")

            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")

            return nlp

