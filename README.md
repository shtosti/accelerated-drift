# not-an-llm

Modular research pipeline for studying writing-feature changes before and after broad LLM adoption.

## Goal

1. Download Semantic Scholar papers with rich metadata, title, and abstract.
2. Build a longitudinal dataset around a configurable LLM introduction year.
3. Test for two effects:
	- post-introduction writing-feature shift
	- later backlash trend where LLM-like markers decline
4. Track readability and stylistic trends over time from preprocessed text.

## Experiment Control

The full run is controlled by [config.toml](config.toml).
The mini run is controlled by [config_mini.toml](config_mini.toml).

Edit the matching file to control:

1. query, year range, fields, output path
2. LLM introduction year and pre/post windows
3. analysis feature toggles

## Run

Full run:
1. Collect papers:
	uv run python main.py --config config.toml collect
2. Preprocess text:
	uv run python main.py --config config.toml preprocess
3. Analyze features/readability and save yearly trends:
	uv run python main.py --config config.toml analyze

Mini run:
1. Preprocess the mini dataset:
	uv run python main.py --config config_mini.toml preprocess
2. Analyze the mini dataset:
	uv run python main.py --config config_mini.toml analyze

Show built-in hypotheses:
	uv run python main.py --config config.toml show-hypotheses

Separate external dataset pipelines (kept independent from the main collect/preprocess/analyze flow):
1. Build paired external JSONL with raw+lemmatized text:
	uv run python main.py --config config_external.toml external-preprocess
2. Analyze paired external records without temporal trends:
	uv run python main.py --config config_external.toml external-analyze

To switch between HC3 and MAGE, edit the `dataset` value in [config_external.toml](config_external.toml) and adjust the matching output paths there. The Slurm wrappers point at this file directly.

Optional: set S2_API_KEY for higher Semantic Scholar API limits.

Collection sources:
1. `semantic_scholar`: query-based Semantic Scholar API.
2. `arxiv`: arXiv API with month-split full collection to avoid 10k offset failures.
3. `medarxiv`: medRxiv API with date-range cursor pagination and the same month-split full collection strategy.

## Structure

- [config.toml](config.toml): full-run moderation file
- [config_mini.toml](config_mini.toml): mini-run moderation file
- [main.py](main.py): root entrypoint
- [src/not_an_llm/config.py](src/not_an_llm/config.py): typed config loading
- [src/not_an_llm/clients/semantic_scholar.py](src/not_an_llm/clients/semantic_scholar.py): API client
- [src/not_an_llm/clients/arxiv.py](src/not_an_llm/clients/arxiv.py): arXiv API client
- [src/not_an_llm/clients/medarxiv.py](src/not_an_llm/clients/medarxiv.py): medRxiv API client
- [src/not_an_llm/pipelines/collect.py](src/not_an_llm/pipelines/collect.py): ingestion pipeline
- [src/not_an_llm/pipelines/preprocess.py](src/not_an_llm/pipelines/preprocess.py): preprocessing pipeline
- [src/not_an_llm/pipelines/analyze.py](src/not_an_llm/pipelines/analyze.py): feature/readability analysis pipeline
- [src/not_an_llm/pipelines/external_preprocess.py](src/not_an_llm/pipelines/external_preprocess.py): HC3/MAGE paired human/AI preprocessing pipelines
- [src/not_an_llm/pipelines/external_analyze.py](src/not_an_llm/pipelines/external_analyze.py): non-temporal external human-vs-AI comparison pipeline
- [src/not_an_llm/preprocessing/text.py](src/not_an_llm/preprocessing/text.py): text preprocessing class
- [src/not_an_llm/analysis/feature_extractor.py](src/not_an_llm/analysis/feature_extractor.py): style feature extraction class
- [src/not_an_llm/analysis/readability.py](src/not_an_llm/analysis/readability.py): readability metrics class
- [src/not_an_llm/analysis/trends.py](src/not_an_llm/analysis/trends.py): yearly trend aggregation and plotting
- [src/not_an_llm/analysis/features.py](src/not_an_llm/analysis/features.py): analysis hypotheses scaffold
- [documents/ARCHITECTURE.md](documents/ARCHITECTURE.md): brief module map
- [documents/RESEARCH_PLAN.md](documents/RESEARCH_PLAN.md): brief study logic