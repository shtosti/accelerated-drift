from __future__ import annotations

from dataclasses import dataclass, field
import re
import math
from collections import Counter
import pandas as pd


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


@dataclass(slots=True)
class FeatureExtractor:
    nlp: any
    
    syntactic_features: dict[str, str] = field(
        default_factory=lambda: {
            "em_dash": r"--",
            "semicolon": r";",
        }
    )

    marker_words: list[str] = field(
        default_factory=lambda: [
            "unparalleled",
            "invaluable",
            "delve",
        ]
    )

    enable_list_of_three_marker: bool = True
    marker_word_matching: str = "exact"

    hedges: list[str] = field(
        default_factory=lambda: ["may", "might", "could", "suggest", "indicate"]
    )
    certainty_terms: list[str] = field(
        default_factory=lambda: ["demonstrate", "prove", "show", "confirm"]
    )

    _hedge_set: set[str] = field(default_factory=set, init=False, repr=False)
    _certainty_set: set[str] = field(default_factory=set, init=False, repr=False)
    _hedge_patterns: list[re.Pattern[str]] = field(default_factory=list, init=False, repr=False)
    _certainty_patterns: list[re.Pattern[str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.marker_word_matching not in {"exact", "lemma"}:
            raise ValueError("marker_word_matching must be 'exact' or 'lemma'")

        self.marker_words = [w.strip().lower() for w in self.marker_words if w.strip()]
        self.hedges = [w.strip().lower() for w in self.hedges if w.strip()]
        self.certainty_terms = [w.strip().lower() for w in self.certainty_terms if w.strip()]

        self.syntactic_features = {
            _slugify(name): pattern.strip()
            for name, pattern in self.syntactic_features.items()
            if name and pattern
        }

        self._hedge_patterns = [
            re.compile(rf"\b{re.escape(t)}\b") for t in self.hedges
        ]
        self._certainty_patterns = [
            re.compile(rf"\b{re.escape(t)}\b") for t in self.certainty_terms
        ]

        self._hedge_set = set(self.hedges)
        self._certainty_set = set(self.certainty_terms)

    # =========================================================
    # MAIN
    # =========================================================
    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()

        if "text_clean" not in df.columns:
            raise ValueError("Missing 'text_clean' column")

        if "word_count" not in df.columns:
            df["word_count"] = df["text_clean"].fillna("").astype(str).str.split().str.len()

        text = df["text_clean"].fillna("").astype(str)
        words = df["word_count"].fillna(0).astype(float) + 1.0

        has_lemma = "text_lemma" in df.columns
        lemma_text = df["text_lemma"].fillna("").astype(str) if has_lemma else None

        if not hasattr(self, "nlp"):
            raise ValueError("FeatureExtractor requires .nlp spaCy model")

        docs = list(self.nlp.pipe(text.tolist(), batch_size=16, n_process=8))

        # sentence check
        if not any(hasattr(doc, "sents") for doc in docs[:1]):
            raise ValueError("spaCy pipeline missing sentencizer")

        # ========================================================
        # regex features
        # ========================================================
        for name, pattern in self.syntactic_features.items():
            df[f"{name}_count"] = text.str.count(pattern)
            df[f"{name}_per_1k_words"] = df[f"{name}_count"] / words * 1000.0

        # ========================================================
        # spaCy structural features
        # ========================================================
        df["marker_list_of_three"] = [self._count_list_of_three_doc(doc) for doc in docs]
        df["coordination_count"] = [self._coordination_count_doc(doc) for doc in docs]

        df["sentence_count"] = [len(list(doc.sents)) for doc in docs]

        df["coordination_density"] = (
            df["coordination_count"] / (df["sentence_count"] + 1)
        )

        df["clause_depth"] = [self._clause_depth_doc(doc) for doc in docs]
        df["dependency_entropy"] = [self._dependency_entropy_doc(doc) for doc in docs]

        # ========================================================
        # marker words
        # ========================================================
        for marker in self.marker_words:
            key = _slugify(marker)

            if self.marker_word_matching == "lemma":
                if lemma_text is None:
                    raise ValueError("text_lemma required for lemma mode")

                df[f"marker_word_{key}"] = lemma_text.apply(
                    lambda v: self._count_lemma_hits(v, marker)
                )
            else:
                df[f"marker_word_{key}"] = text.apply(
                    lambda v: self._count_exact_word_hits(v, marker)
                )

        # ========================================================
        # marker density (safe)
        # ========================================================
        marker_cols = [c for c in df.columns if c.startswith("marker_")]
        if marker_cols:
            df["marker_density"] = df[marker_cols].sum(axis=1) / words
        else:
            df["marker_density"] = 0.0

        # ========================================================
        # hedges / certainty
        # ========================================================
        if lemma_text is not None:
            df["hedge_count"] = lemma_text.apply(
                lambda v: self._count_lemma_terms(v, self._hedge_set)
            )
            df["certainty_count"] = lemma_text.apply(
                lambda v: self._count_lemma_terms(v, self._certainty_set)
            )
        else:
            df["hedge_count"] = text.apply(
                lambda v: sum(len(p.findall(v)) for p in self._hedge_patterns)
            )
            df["certainty_count"] = text.apply(
                lambda v: sum(len(p.findall(v)) for p in self._certainty_patterns)
            )

        df["hedge_ratio"] = df["hedge_count"] / words
        df["certainty_ratio"] = df["certainty_count"] / words

        return df

    # =========================================================
    # HELPERS
    # =========================================================
    def _count_exact_word_hits(self, text: str, target: str) -> int:
        return text.lower().split().count(target)

    def _count_lemma_hits(self, lemma_text: str, target: str) -> int:
        return sum(1 for t in lemma_text.split() if t == target) if lemma_text else 0

    def _count_lemma_terms(self, lemma_text: str, terms: set[str]) -> int:
        return sum(1 for t in lemma_text.split() if t in terms) if lemma_text else 0

    # =========================================================
    # spaCy FEATURES
    # =========================================================
    def _count_list_of_three_doc(self, doc) -> int:
        seen = set()
        count = 0

        for token in doc:
            if token.dep_ == "cc" and token.text.lower() in {"and", "or"}:
                head = token.head

                if head.i in seen:
                    continue

                conjuncts = [head] + list(head.conjuncts)

                if len(conjuncts) >= 3:
                    count += 1
                    seen.add(head.i)

        return count

    def _coordination_count_doc(self, doc) -> int:
        return sum(
            1
            for token in doc
            if token.dep_ == "cc"
            and len([token.head] + list(token.head.conjuncts)) >= 2
        )

    def _max_depth(self, token) -> int:
        children = list(token.children)
        if not children:
            return 1
        return 1 + max(self._max_depth(c) for c in children)

    def _clause_depth_doc(self, doc) -> int:
        return max((self._max_depth(sent.root) for sent in doc.sents), default=0)

    def _dependency_entropy_doc(self, doc) -> float:
        deps = [t.dep_ for t in doc if t.dep_ != "punct"]
        if not deps:
            return 0.0

        counts = Counter(deps)
        total = sum(counts.values())

        return -sum((c / total) * math.log(c / total) for c in counts.values())