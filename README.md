# Accelerated Drift

Research pipeline for studying changes in scientific writing around broad LLM adoption. It collects arXiv and medRxiv titles and abstracts, extracts lexical, readability, and syntactic features, fits interrupted time-series (ITS) models, and generates figures.


## Quick start

Use Python 3.11 or newer and [uv](https://docs.astral.sh/uv/). Run commands from the repository root. Collection and first-time model downloads require internet access.

```bash
git clone https://github.com/shtosti/accelerated-drift.git
cd accelerated-drift
uv sync --locked
uv run python -m spacy download en_core_web_sm
```

For a small pipeline check, use Bash (including Git Bash on Windows):

```bash
bash scripts/reproduce.sh --config config_mini.toml collect preprocess analyze visualize
```

The mini configuration currently samples arXiv quantitative biology titles from 2015–2016. It checks pipeline execution; estimating post-2022 effects requires a date range spanning the intervention. Topic modeling is enabled by default and can still require substantial memory and model downloads. Set `[analysis].topic_modeling_enabled = false` for a lighter check.

Raw, preprocessed, and document-level datasets are not bundled as a complete reproduction dataset. Some aggregate tables and figures are tracked; other generated outputs must be regenerated. Run `collect` before `preprocess` unless you already have the configured raw JSONL.

## Configure and run

[config.toml](config.toml) controls full runs; [config_mini.toml](config_mini.toml) controls small checks. Set:

- `[collection].source`: `arxiv_ai`, `arxiv_qbio`, `arxiv_stat`, or `medarxiv`.
- Collection queries, years, cutoff date, and full/monthly sampling settings.
- `[analysis].text_source`: `title`, `abstract`, or `title_abstract`.
- Feature selection, topic modeling, output paths, and statistical settings.

The full config currently selects arXiv statistics abstracts, with collection ending on April 30, 2026. One invocation processes the selected source and text mode; it does not automatically reproduce every corpus and title/abstract combination. Review the config before running, and use distinct output paths when retaining multiple runs.

Run stages individually on any supported shell:

```bash
uv run python main.py --config config.toml collect
uv run python main.py --config config.toml preprocess
uv run python main.py --config config.toml analyze
uv run python main.py --config config.toml visualize
```

`analyze` extracts features and fits models; `visualize` regenerates figures from saved analysis tables. Existing inputs are needed when skipping earlier stages.

The Bash wrapper synchronizes dependencies from `uv.lock` and installs the spaCy model if missing:

```bash
bash scripts/reproduce.sh all
```

`all` runs `collect`, `preprocess`, `analyze`, `visualize`, and both comparison scripts. Comparisons need the relevant corpus outputs, which may require separate runs. Additional wrapper stages are `additional-analysis`, `topic-compare`, `comparisons`, and `corpus-compare`. See `bash scripts/reproduce.sh --help` for options. To preview commands without dependency setup, use `--skip-sync --dry-run` together.

## Outputs and methods

Outputs are organized by dataset stem under `data/analysis/` and `data/visuals/` (subject to configured paths).

| Output | Contents |
| --- | --- |
| `data/analysis/<stem>/features.jsonl` | Document-level features |
| `trends_by_month.csv`, `trends_by_year.csv` | Aggregated feature trends in each analysis directory |
| `its_stats.csv`, `its_placebo_stats.csv` | Monthly ITS estimates and placebo checks |
| `first_post_year_counterfactual_excess.csv`, `first_two_year_counterfactual_excess.csv` | Early post-intervention excess over the pre-intervention trend |
| `topic_*.csv`, `topics/topic_*/` | Topic summaries, prevalence, and topic-level trends |
| `additional_analysis/` | Determiner decomposition and dependency-bigram analyses |
| `data/visuals/<stem>/` | Rendered figures |

The primary monthly model estimates baseline trend, intervention level shift, and post-intervention slope change:

```text
feature_t = beta0 + beta1*time + beta2*post + beta3*time_after + error_t
```

Models use monthly paper-count weights and HAC standard errors. Main inferential outputs include annualized slope changes, confidence intervals, p-values, and feature-family Benjamini–Hochberg q-values. Counterfactual analyses assess excess over the pre-intervention trend. Pre/post percentage changes are descriptive summaries. Temporal associations alone do not establish that LLM use caused a change.

Optional BERTopic analysis uses sentence embeddings, UMAP, and HDBSCAN or KMeans, with topic-level feature analyses. See [the output and visualization reference](docs/VISUALS_AND_ANALYSIS.md) for details.

## Follow-up analyses

Run targeted dependency analyses from existing feature or preprocessed data:

```bash
uv run python main.py --config config.toml additional-analysis
```

Refresh counterfactual tables from existing monthly trends and regenerate comparisons:

```bash
uv run python main.py --config config.toml additional-analysis --counterfactual-only --comparisons
```

Other scripts require their corresponding saved inputs:

| Command after `uv run python` | Purpose / input |
| --- | --- |
| `scripts/compare_title_abstract_trends.py` | Compare completed title and abstract analyses |
| `scripts/compare_corpus_trends.py` | Compare completed corpus ITS profiles |
| `scripts/decompose_readability_abstracts.py` | Decompose ARI/FKGL using abstract monthly trends |
| `scripts/analyze_syntax_readability_abstracts.py` | Model readability from document features; `--skip-bigrams` avoids dependency-bigram extraction |

Comparison tables and figures go into named directories such as `title_abstract_comparison/` and `corpus_comparison/` under `data/analysis/` and `data/visuals/`.

Separate HC3/MAGE human–AI comparisons are available through `external-preprocess` and `external-analyze`, configured in `[external]`. They are separate from the temporal preprint workflow. For topic comparisons, use `topic-compare` with domain stems matching the outputs you have generated. Explore commands with:

```bash
uv run python main.py --help
uv run python main.py topic-compare --help
```

## Code layout

- `main.py`: repository entry point.
- `src/not_an_llm/cli.py`, `config.py`: CLI and typed configuration.
- `src/not_an_llm/clients/`: collection APIs.
- `src/not_an_llm/pipelines/`: collection, preprocessing, analysis, and visualization workflows.
- `src/not_an_llm/analysis/`: feature extraction, readability, ITS, and topic modeling.
- `scripts/`: reproduction and follow-up analyses.

Large generated datasets, caches, and exploratory figures should stay out of Git. Check `.gitignore` before adding outputs; some aggregate-output directories are intentionally allowed.

## License

Project code is licensed under the [BSD-3-Clause License](LICENSE). Copyright (c) 2026, University of Bern. Third-party datasets, preprint text, models, and dependencies remain subject to their respective terms; this code license does not replace those terms.
