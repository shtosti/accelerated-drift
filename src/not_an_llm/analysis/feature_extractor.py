from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class FeatureExtractor:
    markers: list[str] = field(
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
    hedges: list[str] = field(default_factory=lambda: ["may", "might", "could", "suggest", "indicate"])
    certainty_terms: list[str] = field(default_factory=lambda: ["demonstrate", "prove", "show", "confirm"])

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        text = df["text_clean"].fillna("").astype(str)
        words = df["word_count"].fillna(0).astype(float) + 1.0

        df["em_dash_count"] = text.str.count(r"--")
        df["em_dashes_per_1k_words"] = df["em_dash_count"] / words * 1000.0

        df["semicolon_count"] = text.str.count(";")
        df["semicolon_per_1k"] = df["semicolon_count"] / words * 1000.0

        for marker in self.markers:
            key = marker.replace(" ", "_")
            df[f"marker_{key}"] = text.str.count(marker)

        marker_cols = [col for col in df.columns if col.startswith("marker_")]
        df["marker_density"] = df[marker_cols].sum(axis=1) / words

        df["hedge_count"] = text.apply(lambda value: sum(value.count(term) for term in self.hedges))
        df["certainty_count"] = text.apply(
            lambda value: sum(value.count(term) for term in self.certainty_terms)
        )
        df["hedge_ratio"] = df["hedge_count"] / words
        df["certainty_ratio"] = df["certainty_count"] / words

        return df
