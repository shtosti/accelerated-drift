# Research Plan (Brief)

## Core Question

Does the LLM era produce a measurable writing-feature shift, followed by a backlash trend where LLM-like patterns become less visible?

## Design

1. Collect papers over a broad year range via Semantic Scholar.
2. Aggregate writing features by month.
3. Fit interrupted time-series models around the ChatGPT release date.
4. Test whether post-release slopes differ from pre-release slopes for confirmatory marker, syntax, and readability families.
5. Use placebo intervention years as robustness checks.
6. Use pre/post differences and topic-level plots as exploratory context, not as the main hypothesis test.

## Notes

- The primary inferential output is `data/analysis/<stem>_its_stats.csv`.
- The primary estimand is `slope_change_per_year`, with family-level FDR correction in `slope_change_q`.
- Yearly pre/post tests are retained only as exploratory summaries.
