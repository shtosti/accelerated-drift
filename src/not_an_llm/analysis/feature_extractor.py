from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Callable

import pandas as pd


_TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


@dataclass(slots=True)
class FeatureExtractor:
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
    _phrase_patterns: dict[str, re.Pattern[str]] = field(default_factory=dict, init=False, repr=False)
    _stem_fn: Callable[[str], str] = field(default=lambda token: token, init=False, repr=False)
    _stemmed_marker_words: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        allowed_modes = {"exact", "stem"}
        if self.marker_word_matching not in allowed_modes:
            raise ValueError("marker_word_matching must be one of: exact, stem")

        self.marker_phrases = [item.strip().lower() for item in self.marker_phrases if item.strip()]
        self.marker_words = [item.strip().lower() for item in self.marker_words if item.strip()]

        self._phrase_patterns = {}
        for phrase in self.marker_phrases:
            escaped = re.escape(phrase).replace("\\ ", r"\\s+")
            self._phrase_patterns[phrase] = re.compile(r"\b" + escaped + r"\b")

        self._stem_fn = lambda token: token
        if self.marker_word_matching == "stem":
            self._stem_fn = _simple_stem
            self._stemmed_marker_words = {self._stem_fn(word) for word in self.marker_words}
        else:
            self._stemmed_marker_words = set(self.marker_words)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        text = df["text_clean"].fillna("").astype(str)
        words = df["word_count"].fillna(0).astype(float) + 1.0

        df["em_dash_count"] = text.str.count(r"--")
        df["em_dashes_per_1k_words"] = df["em_dash_count"] / words * 1000.0

        df["semicolon_count"] = text.str.count(";")
        df["semicolon_per_1k"] = df["semicolon_count"] / words * 1000.0

        for marker in self.marker_phrases:
            key = _slugify(marker)
            pattern = self._phrase_patterns[marker]
            df[f"marker_phrase_{key}"] = text.apply(lambda value: len(pattern.findall(value)))

        for marker in self.marker_words:
            key = _slugify(marker)
            df[f"marker_word_{key}"] = text.apply(
                lambda value: self._count_word_hits(value, marker)
            )

        marker_cols = [col for col in df.columns if col.startswith("marker_")]
        df["marker_density"] = df[marker_cols].sum(axis=1) / words

        df["hedge_count"] = text.apply(lambda value: sum(value.count(term) for term in self.hedges))
        df["certainty_count"] = text.apply(
            lambda value: sum(value.count(term) for term in self.certainty_terms)
        )
        df["hedge_ratio"] = df["hedge_count"] / words
        df["certainty_ratio"] = df["certainty_count"] / words

        return df

    def _count_word_hits(self, text: str, target: str) -> int:
        tokens = _TOKEN_PATTERN.findall(text.lower())
        if self.marker_word_matching == "exact":
            return sum(1 for token in tokens if token == target)

        return sum(1 for token in tokens if self._stem_fn(token) == self._stem_fn(target))


def _simple_stem(word: str) -> str:
    # Lightweight suffix-based stemmer to group inflections (e.g., "delving" -> "delv").
    value = word.lower().strip()
    if len(value) <= 3:
        return value

    suffix_rules: list[tuple[str, str]] = [
        ("izations", "ize"),
        ("ization", "ize"),
        ("ational", "ate"),
        ("fulness", "ful"),
        ("ousness", "ous"),
        ("iveness", "ive"),
        ("ments", "ment"),
        ("ingly", ""),
        ("edly", ""),
        ("ing", ""),
        ("ed", ""),
        ("ies", "y"),
        ("sses", "ss"),
        ("s", ""),
    ]

    for suffix, replacement in suffix_rules:
        if value.endswith(suffix) and len(value) > len(suffix) + 2:
            return value[: -len(suffix)] + replacement

    return value
