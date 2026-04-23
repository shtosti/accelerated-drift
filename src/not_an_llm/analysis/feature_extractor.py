from __future__ import annotations

from dataclasses import dataclass, field
import re

import pandas as pd


_TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


@dataclass(slots=True)
class FeatureExtractor:
    syntactic_features: dict[str, str] = field(
        default_factory=lambda: {
            "em_dash": r"--",
            "semicolon": r";",
        }
    )
    marker_phrases: list[str] = field(
        default_factory=lambda: [
            "overall",
            "in conclusion",
            "this paper",
            "this study",
            "we propose",
            "we present",
            "we demonstrate",
            "results show",
            "our results",
            "in this work",
        ]
    )
    marker_words: list[str] = field(
        default_factory=lambda: [
            "unparalleled",
            "invaluable",
            "delve",
        ]
    )
    marker_word_matching: str = "exact"
    hedges: list[str] = field(default_factory=lambda: ["may", "might", "could", "suggest", "indicate"])
    certainty_terms: list[str] = field(default_factory=lambda: ["demonstrate", "prove", "show", "confirm"])
    _hedge_patterns: list[re.Pattern[str]] = field(default_factory=list, init=False, repr=False)
    _certainty_patterns: list[re.Pattern[str]] = field(default_factory=list, init=False, repr=False)
    _phrase_patterns: dict[str, re.Pattern[str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        allowed_modes = {"exact", "lemma"}
        if self.marker_word_matching not in allowed_modes:
            raise ValueError("marker_word_matching must be one of: exact, lemma")

        self.marker_phrases = [item.strip().lower() for item in self.marker_phrases if item.strip()]
        self.marker_words = [item.strip().lower() for item in self.marker_words if item.strip()]
        self.hedges = [item.strip().lower() for item in self.hedges if item.strip()]
        self.certainty_terms = [item.strip().lower() for item in self.certainty_terms if item.strip()]

        normalized_syntactic: dict[str, str] = {}
        for name, pattern in self.syntactic_features.items():
            normalized_name = _slugify(str(name))
            normalized_pattern = str(pattern).strip()
            if normalized_name and normalized_pattern:
                normalized_syntactic[normalized_name] = normalized_pattern
        self.syntactic_features = normalized_syntactic

        self._phrase_patterns = {}
        for phrase in self.marker_phrases:
            escaped = re.escape(phrase).replace("\\ ", r"\\s+")
            self._phrase_patterns[phrase] = re.compile(r"\b" + escaped + r"\b")

        self._hedge_patterns = [re.compile(r"\b" + re.escape(term) + r"\b") for term in self.hedges]
        self._certainty_patterns = [
            re.compile(r"\b" + re.escape(term) + r"\b") for term in self.certainty_terms
        ]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        text = df["text_clean"].fillna("").astype(str)
        words = df["word_count"].fillna(0).astype(float) + 1.0
        has_lemma = "text_lemma" in df.columns
        lemma_text = df["text_lemma"].fillna("").astype(str) if has_lemma else None

        for feature_name, pattern in self.syntactic_features.items():
            count_col = f"{feature_name}_count"
            per_1k_col = f"{feature_name}_per_1k_words"
            df[count_col] = text.str.count(pattern)
            df[per_1k_col] = df[count_col] / words * 1000.0

        for marker in self.marker_phrases:
            key = _slugify(marker)
            pattern = self._phrase_patterns[marker]
            df[f"marker_phrase_{key}"] = text.apply(lambda value: len(pattern.findall(value)))

        for marker in self.marker_words:
            key = _slugify(marker)
            if self.marker_word_matching == "lemma":
                if lemma_text is None:
                    raise ValueError(
                        "text_lemma column is required for marker_word_matching='lemma'. "
                        "Run preprocess to regenerate the dataset with lemmatized tokens."
                    )
                df[f"marker_word_{key}"] = lemma_text.apply(
                    lambda value: self._count_lemma_hits(value, marker)
                )
            else:
                df[f"marker_word_{key}"] = text.apply(
                    lambda value: self._count_exact_word_hits(value, marker)
                )

        marker_cols = [col for col in df.columns if col.startswith("marker_")]
        if marker_cols:
            df["marker_density"] = df[marker_cols].sum(axis=1) / words
        else:
            df["marker_density"] = 0.0

        if lemma_text is not None:
            hedge_terms = set(self.hedges)
            certainty_terms = set(self.certainty_terms)
            df["hedge_count"] = lemma_text.apply(lambda value: self._count_lemma_terms(value, hedge_terms))
            df["certainty_count"] = lemma_text.apply(
                lambda value: self._count_lemma_terms(value, certainty_terms)
            )
        else:
            df["hedge_count"] = text.apply(
                lambda value: sum(len(pattern.findall(value)) for pattern in self._hedge_patterns)
            )
            df["certainty_count"] = text.apply(
                lambda value: sum(len(pattern.findall(value)) for pattern in self._certainty_patterns)
            )
        df["hedge_ratio"] = df["hedge_count"] / words
        df["certainty_ratio"] = df["certainty_count"] / words

        return df

    def _count_exact_word_hits(self, text: str, target: str) -> int:
        tokens = _TOKEN_PATTERN.findall(text.lower())
        return sum(1 for token in tokens if token == target)

    def _count_lemma_hits(self, lemma_text: str, target: str) -> int:
        if not lemma_text:
            return 0

        tokens = lemma_text.split()
        return sum(1 for token in tokens if token == target)

    def _count_lemma_terms(self, lemma_text: str, terms: set[str]) -> int:
        if not lemma_text:
            return 0

        tokens = lemma_text.split()
        return sum(1 for token in tokens if token in terms)
