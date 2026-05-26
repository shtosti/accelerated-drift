# Architecture (Brief)

## Principles

1. Config-first: experiment controls live in root configuration files.
2. Small modules: each file owns one responsibility.
3. Pipeline clarity: ingest first, analyze later.
4. Analysis stages are explicit: preprocess -> feature extraction/readability -> trend analysis.

## Module Map

- src/not_an_llm/config.py
  - Loads and validates typed settings from config.toml.
- src/not_an_llm/clients/arxiv.py
  - Handles arXiv API requests and month-split collection.
- src/not_an_llm/clients/medarxiv.py
  - Handles medRxiv API requests and date-range pagination.
- src/not_an_llm/pipelines/collect.py
  - Orchestrates collection and writes JSONL.
- src/not_an_llm/preprocessing/text.py
  - Text preprocessor class for normalization and core text fields.
- src/not_an_llm/pipelines/preprocess.py
  - Runs preprocessing and persists preprocessed JSONL.
- src/not_an_llm/analysis/feature_extractor.py
  - Extracts stylistic and marker-based features from preprocessed text.
- src/not_an_llm/analysis/feature_selection.py
  - Owns shared feature inclusion rules and marker group summaries for pipelines.
- src/not_an_llm/analysis/feature_groups.py
  - Defines canonical grouped feature lists for diff and stack plots.
- src/not_an_llm/analysis/readability.py
  - Computes readability metrics (Flesch, FK grade, Dale-Chall, SMOG, ARI, Fog).
- src/not_an_llm/analysis/trends.py
  - Aggregates yearly/monthly feature trends and saves exploratory visualizations.
- src/not_an_llm/analysis/interrupted_time_series.py
  - Fits monthly segmented regressions with HAC standard errors, FDR correction, and placebo checks.
- src/not_an_llm/analysis/topic_modeling/
  - Runs BERTopic topic assignment with configurable HDBSCAN or KMeans clustering, records topic summaries and optional hierarchy review candidates, and saves topic-level reports.
- src/not_an_llm/pipelines/analyze.py
  - Orchestrates analysis modules and writes trend artifacts.
- src/not_an_llm/cli.py
  - CLI routing for collection, preprocessing, analysis, and visualization.

## Data Contract (Current)

1. Raw JSONL from source collection.
2. Preprocessed JSONL with normalized text fields.
3. Feature-enriched JSONL plus yearly/monthly trend CSVs.
4. Monthly interrupted time-series statistics and placebo checks.
5. Topic summary/prevalence CSVs under data/analysis.
6. Exploratory trend plots under data/visuals.
