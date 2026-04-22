# not-an-llm

Modular research pipeline for studying writing-feature changes before and after broad LLM adoption.

## Goal

1. Download Semantic Scholar papers with rich metadata, title, and abstract.
2. Build a longitudinal dataset around a configurable LLM introduction year.
3. Test for two effects:
	- post-introduction writing-feature shift
	- later backlash trend where LLM-like markers decline
4. Track readability and stylistic trends over time from preprocessed text.

## Single Source of Experiment Control

All experiment moderation is centralized in [config.toml](config.toml).

Edit only this file to control:

1. query, year range, fields, output path
2. LLM introduction year and pre/post windows
3. analysis feature toggles

## Run

1. Collect papers:
	uv run python main.py --config config.toml collect
2. Preprocess text (separate pipeline stage):
	uv run python main.py --config config.toml preprocess
3. Analyze features/readability and save yearly trends:
	uv run python main.py --config config.toml analyze
4. Show built-in hypotheses:
	uv run python main.py --config config.toml show-hypotheses

Optional: set S2_API_KEY for higher Semantic Scholar API limits.

## Structure

- [config.toml](config.toml): root moderation file
- [main.py](main.py): root entrypoint
- [src/not_an_llm/config.py](src/not_an_llm/config.py): typed config loading
- [src/not_an_llm/clients/semantic_scholar.py](src/not_an_llm/clients/semantic_scholar.py): API client
- [src/not_an_llm/pipelines/collect.py](src/not_an_llm/pipelines/collect.py): ingestion pipeline
- [src/not_an_llm/pipelines/preprocess.py](src/not_an_llm/pipelines/preprocess.py): preprocessing pipeline
- [src/not_an_llm/pipelines/analyze.py](src/not_an_llm/pipelines/analyze.py): feature/readability analysis pipeline
- [src/not_an_llm/preprocessing/text.py](src/not_an_llm/preprocessing/text.py): text preprocessing class
- [src/not_an_llm/analysis/feature_extractor.py](src/not_an_llm/analysis/feature_extractor.py): style feature extraction class
- [src/not_an_llm/analysis/readability.py](src/not_an_llm/analysis/readability.py): readability metrics class
- [src/not_an_llm/analysis/trends.py](src/not_an_llm/analysis/trends.py): yearly trend aggregation and plotting
- [src/not_an_llm/analysis/features.py](src/not_an_llm/analysis/features.py): analysis hypotheses scaffold
- [documents/ARCHITECTURE.md](documents/ARCHITECTURE.md): brief module map
- [documents/RESEARCH_PLAN.md](documents/RESEARCH_PLAN.md): brief study logic