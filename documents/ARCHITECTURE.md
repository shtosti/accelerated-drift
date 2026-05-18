# Architecture (Brief)

## Principles

1. Config-first: all experiment controls live in root config.toml.
2. Small modules: each file owns one responsibility.
3. Pipeline clarity: ingest first, analyze later.
4. Analysis stages are explicit: preprocess -> feature extraction/readability -> trend analysis.

## Module Map

- src/not_an_llm/config.py
  - Loads and validates typed settings from config.toml.
- src/not_an_llm/clients/semantic_scholar.py
  - Handles Semantic Scholar API requests and pagination.
- src/not_an_llm/pipelines/collect.py
  - Orchestrates collection and writes JSONL.
- src/not_an_llm/preprocessing/text.py
  - Text preprocessor class for normalization and core text fields.
- src/not_an_llm/pipelines/preprocess.py
  - Runs preprocessing and persists preprocessed JSONL.
- src/not_an_llm/analysis/feature_extractor.py
  - Extracts stylistic and marker-based features from preprocessed text.
- src/not_an_llm/analysis/readability.py
  - Computes readability metrics (Flesch, FK grade, Fog, SMOG).
- src/not_an_llm/analysis/trends.py
  - Aggregates yearly feature trends and saves visualizations.
- src/not_an_llm/analysis/topic_modeling/
  - Runs BERTopic topic assignment, records hierarchical merge candidates, filters topics by minimum abstract count/share, and saves topic-level reports.
- src/not_an_llm/pipelines/analyze.py
  - Orchestrates analysis modules and writes trend artifacts.
- src/not_an_llm/analysis/features.py
  - Defines high-level hypotheses.
- src/not_an_llm/cli.py
  - CLI routing for collection, preprocessing, and analysis.

## Data Contract (Current)

1. Raw JSONL from source collection.
2. Preprocessed JSONL with normalized text fields.
3. Feature-enriched JSONL plus per-year trend CSV and trend plots.
