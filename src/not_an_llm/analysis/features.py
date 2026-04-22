from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FeatureShiftHypothesis:
    name: str
    rationale: str



def default_hypotheses() -> list[FeatureShiftHypothesis]:
    return [
        FeatureShiftHypothesis(
            name="style_shift_post_llm_intro",
            rationale="Test whether writing features shift after llm_introduction_year.",
        ),
        FeatureShiftHypothesis(
            name="backlash_trend",
            rationale="Test whether initial LLM-like markers later decline as avoidance behavior grows.",
        ),
    ]
