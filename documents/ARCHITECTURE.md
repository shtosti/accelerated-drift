# Architecture (Brief)

## Principles

1. Config-first: all experiment controls live in root config.toml.
2. Small modules: each file owns one responsibility.
3. Pipeline clarity: ingest first, analyze later.

## Module Map

- src/not_an_llm/config.py
  - Loads and validates typed settings from config.toml.
- src/not_an_llm/clients/semantic_scholar.py
  - Handles Semantic Scholar API requests and pagination.
- src/not_an_llm/pipelines/collect.py
  - Orchestrates collection and writes JSONL.
- src/not_an_llm/analysis/features.py
  - Defines initial hypotheses and feature placeholders.
- src/not_an_llm/cli.py
  - CLI routing for collection and diagnostics.

## Data Contract (Current)

Output is JSONL with one paper per line, containing the fields configured in config.toml.
