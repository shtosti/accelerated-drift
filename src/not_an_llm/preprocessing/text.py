from __future__ import annotations

import re
import pandas as pd


class TextPreprocessor:
    """Normalize and structure raw title/abstract text.
    
    Focuses on text normalization only. All linguistic feature extraction
    (tokens, lemmas, readability, markers) is performed during analysis
    by UnifiedLinguisticFeatures using a single spaCy pass for consistency.
    """

    def __init__(self, *, keep_case: bool = False) -> None:
        self.keep_case = keep_case

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
        df["year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")

        return df

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


