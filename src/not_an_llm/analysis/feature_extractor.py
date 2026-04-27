from __future__ import annotations

from dataclasses import dataclass, field
import re
import math
import json
from collections import Counter
from pathlib import Path

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

    marker_verbs: list[str] = field(
        default_factory=lambda: [
            "delve",
            "underscore",
            "showcase",
            "enhance",
            "exhibit",
            "garner",
            "align",
        ]
    )

    marker_adjectives: list[str] = field(
        default_factory=lambda: [
            "crucial",
            "pivotal",
            "comprehensive",
            "intricate",
            "potential",
        ]
    )

    marker_phrases: list[str] = field(
        default_factory=lambda: [
            "meticulously delve",
            "intricate web",
            "comprehensive chapter",
            "deep dive",
            "intricate interplay",
            "essential insight",
        ]
    )

    sequential_markers: list[str] = field(
        default_factory=lambda: ["additionally", "furthermore", "moreover", "subsequently", "further"]
    )

    causal_markers: list[str] = field(
        default_factory=lambda: ["hence", "thus", "consequently", "accordingly", "thereby"]
    )

    contrast_markers: list[str] = field(
        default_factory=lambda: ["however", "nonetheless", "nevertheless", "conversely", "alternatively"]
    )

    emphasis_markers: list[str] = field(
        default_factory=lambda: ["notably", "crucially", "remarkably", "particularly", "importantly"]
    )

    summary_markers: list[str] = field(
        default_factory=lambda: ["overall", "collectively", "ultimately", "in summary", "taken together"]
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
        self.marker_words = [w.strip().lower() for w in self.marker_words if w.strip()]
        self.marker_verbs = [w.strip().lower() for w in self.marker_verbs if w.strip()]
        self.marker_adjectives = [w.strip().lower() for w in self.marker_adjectives if w.strip()]
        self.marker_phrases = [w.strip().lower() for w in self.marker_phrases if w.strip()]
        self.sequential_markers = [w.strip().lower() for w in self.sequential_markers if w.strip()]
        self.causal_markers = [w.strip().lower() for w in self.causal_markers if w.strip()]
        self.contrast_markers = [w.strip().lower() for w in self.contrast_markers if w.strip()]
        self.emphasis_markers = [w.strip().lower() for w in self.emphasis_markers if w.strip()]
        self.summary_markers = [w.strip().lower() for w in self.summary_markers if w.strip()]
        self.hedges = [w.strip().lower() for w in self.hedges if w.strip()]
        self.certainty_terms = [w.strip().lower() for w in self.certainty_terms if w.strip()]

        self._hedge_patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in self.hedges]
        self._certainty_patterns = [re.compile(rf"\b{re.escape(t)}\b") for t in self.certainty_terms]

        self._hedge_set = set(self.hedges)
        self._certainty_set = set(self.certainty_terms)

    # =========================================================
    # MAIN
    # =========================================================
    def transform(self, frame: pd.DataFrame, debug_path: str | None = None) -> pd.DataFrame:
        df = frame.copy()

        if "text_clean" not in df.columns:
            raise ValueError("Missing 'text_clean' column")

        if "word_count" not in df.columns:
            df["word_count"] = df["text_clean"].fillna("").astype(str).str.split().str.len()

        text = df["text_clean"].fillna("").astype(str)
        text_lower = text.str.lower()
        words = df["word_count"].fillna(0).astype(float) + 1.0

        lemma_text = df["text_lemma"].fillna("").astype(str) if "text_lemma" in df.columns else None
        lemma_lower = lemma_text.str.lower() if lemma_text is not None else None

        if not hasattr(self, "nlp"):
            raise ValueError("FeatureExtractor requires spaCy model")

        docs = list(self.nlp.pipe(text.tolist(), batch_size=128, n_process=16))

        # ========================================================
        # regex features
        # ========================================================
        for name, pattern in self.syntactic_features.items():
            df[f"{_slugify(name)}_count"] = text.str.count(pattern)
            df[f"{_slugify(name)}_per_1k_words"] = df[f"{_slugify(name)}_count"] / words * 1000.0

        # ========================================================
        # syntactic structure (raw)
        # ========================================================
        df["sentence_count"] = [len(list(doc.sents)) for doc in docs]
        df["clause_depth"] = [self._clause_depth_doc(doc) for doc in docs]
        df["dependency_entropy"] = [self._dependency_entropy_doc(doc) for doc in docs]
        df["dependency_length"] = [self._dependency_length_doc(doc) for doc in docs]
        df["dependency_length_norm"] = [self._dependency_length_doc_normalized(doc) for doc in docs]
        df["dependency_distribution"] = [self._dependency_distribution_doc(doc) for doc in docs]
        df["coordination_count"] = [self._coordination_count_doc(doc) for doc in docs]
        df["coordination_density"] = df["coordination_count"] / (df["sentence_count"] + 1)
        df["sentence_depth_std"] = [self._sentence_depth_std_doc(doc) for doc in docs]
        df["list_of_three"] = [self._count_list_of_three_doc(doc) for doc in docs]

        # ========================================================
        # syntactic structure (normalized for scale-invariance)
        # ========================================================
        df["clause_depth_per_sentence"] = df["clause_depth"] / (df["sentence_count"] + 1.0)
        df["dependency_entropy_normalized"] = [self._normalize_dependency_entropy_doc(doc) for doc in docs]
        df["coordination_count_per_1k_words"] = df["coordination_count"] / words * 1000.0
        sentence_depth_means = [self._sentence_depth_mean_doc(doc) for doc in docs]
        df["sentence_depth_cv"] = df["sentence_depth_std"] / (pd.Series(sentence_depth_means) + 1e-8)
        df["list_of_three_per_1k_words"] = df["list_of_three"] / words * 1000.0

        # ========================================================
        # marker words
        # ========================================================
        group_specs = [
            ("marker_words", "word", self.marker_words, self.marker_word_matching),
            ("marker_verbs", "verb", self.marker_verbs, self.marker_word_matching),
            ("marker_adjectives", "adjective", self.marker_adjectives, self.marker_word_matching),
            ("marker_phrases", "phrase", self.marker_phrases, "exact"),
            ("sequential_markers", "sequential_marker", self.sequential_markers, self.marker_word_matching),
            ("causal_markers", "causal_marker", self.causal_markers, self.marker_word_matching),
            ("contrast_markers", "contrast_marker", self.contrast_markers, self.marker_word_matching),
            ("emphasis_markers", "emphasis_marker", self.emphasis_markers, self.marker_word_matching),
            ("summary_markers", "summary_marker", self.summary_markers, self.marker_word_matching),
        ]

        for group_name, prefix, terms, matching_mode in group_specs:
            group_total = pd.Series(0, index=df.index, dtype="float64")
            for term in terms:
                key = _slugify(term)
                column_name = f"{prefix}_{key}"
                counts = self._count_term_occurrences(
                    text_lower,
                    lemma_lower,
                    term,
                    matching_mode,
                )
                df[column_name] = counts
                group_total = group_total + counts

            df[f"{group_name}_total"] = group_total
            df[f"{group_name}_total_per_1k_words"] = group_total / words * 1000.0

        df["marker_density"] = df["marker_words_total_per_1k_words"]

        # ========================================================
        # hedges / certainty
        # ========================================================
        if lemma_text is not None:
            df["hedge_count"] = lemma_text.apply(lambda v: sum(t in self._hedge_set for t in v.split()))
            df["certainty_count"] = lemma_text.apply(lambda v: sum(t in self._certainty_set for t in v.split()))
        else:
            df["hedge_count"] = text.apply(lambda v: sum(len(p.findall(v)) for p in self._hedge_patterns))
            df["certainty_count"] = text.apply(lambda v: sum(len(p.findall(v)) for p in self._certainty_patterns))

        df["hedge_ratio"] = df["hedge_count"] / words
        df["certainty_ratio"] = df["certainty_count"] / words

        # ========================================================
        # DEBUG EXPORT (JSONL)
        # ========================================================
        debug_objects = [
            self._build_debug_object(doc, i)
            for i, doc in enumerate(docs)
        ]

        if debug_path:
            self._write_jsonl(debug_objects, debug_path)

        return df

    def _count_term_occurrences(
        self,
        text_lower: pd.Series,
        lemma_lower: pd.Series | None,
        term: str,
        matching_mode: str,
    ) -> pd.Series:
        if " " in term:
            pattern = re.compile(rf"\b{re.escape(term)}\b")
            return text_lower.apply(lambda value: len(pattern.findall(value)))

        if matching_mode == "lemma" and lemma_lower is not None:
            return lemma_lower.apply(lambda value: value.split().count(term))

        return text_lower.apply(lambda value: value.split().count(term))

    # =========================================================
    # DEBUG OBJECTS
    # =========================================================
    def _build_debug_object(self, doc, doc_id: int):
        return {
            "doc_id": doc_id,
            "dependency_entropy": self._dependency_entropy_doc(doc),
            "clause_depth": self._clause_depth_doc(doc),
            "dependency_distribution": self._dependency_distribution_doc(doc),
            "dependency_length": self._dependency_length_doc(doc),
            "dependency_length_norm": self._dependency_length_doc_normalized(doc),
            "deepest_sentence": self._deepest_sentence(doc),
            "long_dependency_examples": self._long_dependency_examples(doc),
            "list_of_three_examples": self._list_of_three_examples(doc),
        }

    def _write_jsonl(self, objects, path: str):
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            for obj in objects:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # =========================================================
    # SENTENCE INSPECTION
    # =========================================================
    def _deepest_sentence(self, doc):
        def depth(token):
            children = list(token.children)
            return 1 + max((depth(c) for c in children), default=0)

        best = (0, "")
        for sent in doc.sents:
            d = depth(sent.root)
            if d > best[0]:
                best = (d, sent.text)
        return {"depth": best[0], "sentence": best[1]}

    def _long_dependency_examples(self, doc, k=3):
        ex = []
        for sent in doc.sents:
            for t in sent:
                if t.head != t:
                    ex.append({
                        "length": abs(t.i - t.head.i),
                        "token": t.text,
                        "head": t.head.text,
                        "sentence": sent.text
                    })
        return sorted(ex, key=lambda x: x["length"], reverse=True)[:k]

    def _list_of_three_examples(self, doc):
        out = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ == "cc" and token.text.lower() in {"and", "or"}:
                    head = token.head
                    conj = [head] + list(head.conjuncts)
                    if len(conj) >= 3:
                        out.append({
                            "sentence": sent.text,
                            "conjuncts": [t.text for t in conj]
                        })
                        break
        return out

    # =========================================================
    # DEPENDENCY METRICS
    # =========================================================
    def _dependency_entropy_doc(self, doc):
        deps = [t.dep_ for t in doc if t.dep_ != "punct"]
        if not deps:
            return 0.0
        c = Counter(deps)
        total = sum(c.values())
        return -sum((v / total) * math.log(v / total) for v in c.values())

    def _dependency_length_doc(self, doc):
        lengths = [abs(t.i - t.head.i) for t in doc if t.head != t]
        return sum(lengths) / len(lengths) if lengths else 0.0

    def _dependency_length_doc_normalized(self, doc):
        vals = []
        for sent in doc.sents:
            lengths = [abs(t.i - t.head.i) for t in sent if t.head != t]
            if lengths:
                vals.append(sum(lengths) / len(lengths))
        return sum(vals) / len(vals) if vals else 0.0

    def _clause_depth_doc(self, doc):
        def depth(t):
            return 1 + max((depth(c) for c in t.children), default=0)
        return max((depth(s.root) for s in doc.sents), default=0)

    def _sentence_depth_std_doc(self, doc):
        """Compute standard deviation of dependency tree depth across sentences.
        Low variance = homogeneous complexity (LLM-like).
        High variance = spiky complexity (human-like).
        """
        def depth(t):
            return 1 + max((depth(c) for c in t.children), default=0)
        
        sentence_depths = [depth(s.root) for s in doc.sents]
        if len(sentence_depths) < 2:
            return 0.0
        return float(pd.Series(sentence_depths).std())

    def _sentence_depth_mean_doc(self, doc):
        """Compute mean dependency tree depth across sentences.
        Used for calculating coefficient of variation.
        """
        def depth(t):
            return 1 + max((depth(c) for c in t.children), default=0)
        
        sentence_depths = [depth(s.root) for s in doc.sents]
        if not sentence_depths:
            return 0.0
        return float(pd.Series(sentence_depths).mean())
    
    def _normalize_dependency_entropy_doc(self, doc):
        """Normalize Shannon entropy to [0, 1] by dividing by log(# dep types).
        This accounts for documents with different syntactic diversity ceilings.
        """
        deps = [t.dep_ for t in doc if t.dep_ != "punct"]
        if not deps:
            return 0.0
        
        c = Counter(deps)
        n_types = len(c)
        if n_types <= 1:
            return 0.0
        
        raw_entropy = self._dependency_entropy_doc(doc)
        max_entropy = math.log(n_types)
        return raw_entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _dependency_distribution_doc(self, doc):
        deps = [t.dep_ for t in doc if t.dep_ != "punct"]
        return dict(Counter(deps))

    def _coordination_count_doc(self, doc):
        return sum(
            1 for t in doc
            if t.dep_ == "cc"
        )
    


    def _count_list_of_three_doc(self, doc):
        seen = set()
        c = 0
        for t in doc:
            if t.dep_ == "cc" and t.text.lower() in {"and", "or"}:
                h = t.head
                if h.i in seen:
                    continue
                conj = [h] + list(h.conjuncts)
                if len(conj) >= 3:
                    c += 1
                    seen.add(h.i)
        return c