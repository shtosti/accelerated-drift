"""Unified linguistic feature extraction from spaCy Doc objects.

Single-pass feature computation covering:
- Structural (tokens, sentences, word counts)
- Syntactic (patterns like em-dash, semicolon)
- Lexical/semantic (hedges, certainty, LLM markers with POS awareness)
- Readability (Flesch, Kincaid, Dale-Chall, SMOG, ARI, Fog, syllables)

This replaces fragmented regex/spacy/textstat passes with one consistent pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import textstat
from typing import Any

import spacy
from spacy.language import Language
from spacy.tokens import Doc


_SLUGIFY = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    return _SLUGIFY.sub("_", value.strip().lower()).strip("_")


@dataclass(slots=True)
class UnifiedLinguisticFeatures:
    """Extract all features from a single spaCy pass."""

    syntactic_features: dict[str, str] = field(default_factory=lambda: {
        "em_dash": r"--",
        "semicolon": r";",
    })
    marker_words: list[str] = field(default_factory=lambda: [
        "unparalleled",
        "invaluable",
        "delve",
    ])
    marker_word_matching: str = "lemma"  # "exact" or "lemma"
    hedge_terms: list[str] = field(default_factory=lambda: [
        "may", "might", "could", "suggest", "indicate"
    ])
    certainty_terms: list[str] = field(default_factory=lambda: [
        "demonstrate", "prove", "show", "confirm"
    ])
    readability_metrics: list[str] = field(default_factory=lambda: [
        "avg_words_per_sentence",
        "avg_syllables_per_word",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "dale_chall",
        "smog_index",
        "automated_readability_index",
        "gunning_fog",
    ])

    _nlp: Language | None = field(default=None, init=False, repr=False)
    _syntactic_patterns: dict[str, re.Pattern[str]] = field(default_factory=dict, init=False, repr=False)
    _phrase_patterns: dict[str, re.Pattern[str]] = field(default_factory=dict, init=False, repr=False)
    _marker_phrase_set: set[str] = field(default_factory=set, init=False, repr=False)
    _marker_word_set: set[str] = field(default_factory=set, init=False, repr=False)
    _hedge_set: set[str] = field(default_factory=set, init=False, repr=False)
    _certainty_set: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        # Normalize term lists
        self.marker_words = [w.strip().lower() for w in self.marker_words if w.strip()]
        self.hedge_terms = [h.strip().lower() for h in self.hedge_terms if h.strip()]
        self.certainty_terms = [c.strip().lower() for c in self.certainty_terms if c.strip()]

        # Build sets for O(1) lookup
        self._marker_word_set = set(self.marker_words)
        self._hedge_set = set(self.hedge_terms)
        self._certainty_set = set(self.certainty_terms)

        # Compile syntactic patterns (on text, not tokenized)
        self._syntactic_patterns = {}
        for name, pattern in self.syntactic_features.items():
            normalized_name = _slugify(str(name))
            if normalized_name and pattern.strip():
                self._syntactic_patterns[normalized_name] = re.compile(pattern.strip())

    def extract(self, text: str, doc: Doc) -> dict[str, Any]:
        """Extract all linguistic features from text and spaCy Doc.
        
        Args:
            text: Original (normalized) text for regex patterns
            doc: Processed spaCy Doc object
            
        Returns:
            Dictionary of all computed features
        """
        result = {}

        # ============= STRUCTURAL =============
        alpha_tokens = [t for t in doc if t.is_alpha]
        sentences = list(doc.sents)

        result["word_count"] = len(alpha_tokens)
        result["sentence_count"] = max(1, len(sentences))

        if not alpha_tokens:
            return self._empty_features(result)

        # ============= SYNTACTIC PATTERNS =============
        for pattern_name, pattern in self._syntactic_patterns.items():
            count = len(pattern.findall(text))
            result[f"{pattern_name}_count"] = count
            result[f"{pattern_name}_per_1k_words"] = count / len(alpha_tokens) * 1000.0

        # ============= MARKER WORDS (POS-aware) =============
        for marker in self.marker_words:
            key = _slugify(marker)
            if self.marker_word_matching == "lemma":
                count = self._count_marker_with_pos(alpha_tokens, marker)
            else:
                count = sum(1 for t in alpha_tokens if t.text.lower() == marker)
            result[f"marker_word_{key}"] = count

        # ============= MARKER DENSITY =============
        marker_cols = [v for k, v in result.items() if k.startswith("marker_")]
        result["marker_density"] = sum(marker_cols) / len(alpha_tokens) if marker_cols else 0.0

        # ============= HEDGES & CERTAINTY (lemma + POS-aware) =============
        result["hedge_count"] = sum(
            1 for t in alpha_tokens
            if t.lemma_.lower() in self._hedge_set and t.pos_ in {"ADV", "AUX", "VERB"}
        )
        result["certainty_count"] = sum(
            1 for t in alpha_tokens
            if t.lemma_.lower() in self._certainty_set and t.pos_ == "VERB"
        )
        result["hedge_ratio"] = result["hedge_count"] / len(alpha_tokens)
        result["certainty_ratio"] = result["certainty_count"] / len(alpha_tokens)

        # ============= READABILITY =============
        readability = self._compute_readability(text, len(alpha_tokens), result["sentence_count"])
        result.update(readability)

        return result

    def _count_marker_with_pos(self, tokens, marker: str) -> int:
        """Count marker occurrences using lemma + lenient POS filter."""
        # Accept broader POS to catch "delve" as NOUN or VERB, "demonstrate" as VERB or AUX
        allowed_pos = {"NOUN", "PROPN", "VERB", "AUX", "ADV", "ADJ"}
        return sum(
            1 for t in tokens
            if t.lemma_.lower() == marker and t.pos_ in allowed_pos
        )

    def _compute_readability(self, text: str, word_count: int, sentence_count: int) -> dict[str, float]:
        """Compute readability metrics."""
        result = {}
        safe_word_count = max(1, word_count)
        safe_sentence_count = max(1, sentence_count)

        for metric in self.readability_metrics:
            if metric == "avg_words_per_sentence":
                result[metric] = safe_word_count / safe_sentence_count
            elif metric == "avg_syllables_per_word":
                syllables = textstat.syllable_count(text)
                result[metric] = syllables / safe_word_count
            elif metric == "flesch_reading_ease":
                result[metric] = textstat.flesch_reading_ease(text)
            elif metric == "flesch_kincaid_grade":
                result[metric] = textstat.flesch_kincaid_grade(text)
            elif metric == "dale_chall":
                result[metric] = textstat.dale_chall_readability_score(text)
            elif metric == "smog_index":
                result[metric] = textstat.smog_index(text)
            elif metric == "automated_readability_index":
                result[metric] = textstat.automated_readability_index(text)
            elif metric == "gunning_fog":
                result[metric] = textstat.gunning_fog(text)

        return result

    def _empty_features(self, base: dict[str, Any]) -> dict[str, Any]:
        """Return zero-filled feature dict when text is empty."""
        for pattern_name in self._syntactic_patterns.keys():
            base[f"{pattern_name}_count"] = 0
            base[f"{pattern_name}_per_1k_words"] = 0.0

        for marker in self.marker_words:
            key = _slugify(marker)
            base[f"marker_word_{key}"] = 0

        base["marker_density"] = 0.0
        base["hedge_count"] = 0
        base["certainty_count"] = 0
        base["hedge_ratio"] = 0.0
        base["certainty_ratio"] = 0.0

        for metric in self.readability_metrics:
            base[metric] = 0.0

        return base
