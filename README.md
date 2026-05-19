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
3. analysis feature toggles and the explicit feature list used for plots/statistics

## Run

### Full pipeline (with plots)

Full run:
1. Collect papers:
	uv run python main.py --config config.toml collect
2. Preprocess text:
	uv run python main.py --config config.toml preprocess
3. Analyze features/readability and save yearly trends with plots:
	uv run python main.py --config config.toml analyze

### Separated analysis and plotting workflow

The analysis step can be separated from plot generation for efficiency. This is useful when:
- Running analysis on a compute cluster without graphics support
- Iterating on plots without recomputing analysis
- Distributing analysis and visualization across different machines

**Key configuration:**
- Set `generate_plots = false` in config to skip plotting during analysis (already set in config.toml)
- The `visualize` command regenerates plots from previously computed analysis data

**Workflow:**

1. **Run analysis only (no plots):**
	uv run python main.py --config config.toml analyze
	
   This produces:
   - Feature dataset: `data/analyzed/arxiv.jsonl` (features for each document)
   - Yearly trends: `data/analysis/arxiv_year.csv`
   - Monthly trends: `data/analysis/arxiv_month.csv`
   - Monthly interrupted time-series statistics: `data/analysis/<stem>_its_stats.csv`
   - Placebo interrupted time-series checks: `data/analysis/<stem>_its_placebo_stats.csv`

2. **Generate plots separately:**
	uv run python main.py --config config.toml visualize
	
   This reads the precomputed trend CSVs and generates all plots in `data/visuals/<dataset_stem>/`.
   
   Benefits:
   - No need to rerun expensive feature extraction
   - Plots can be regenerated with different styling by modifying `src/not_an_llm/analysis/trends.py`
   - Can be run on a machine with graphics libraries after analysis completes

### Mini dataset workflow

1. Preprocess the mini dataset:
	uv run python main.py --config config_mini.toml preprocess
2. Analyze the mini dataset (no plots):
	uv run python main.py --config config_mini.toml analyze
3. Generate plots for mini dataset:
	uv run python main.py --config config_mini.toml visualize

### Other utilities

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
- [src/not_an_llm/cli.py](src/not_an_llm/cli.py): CLI commands (collect, preprocess, analyze, visualize)
- [src/not_an_llm/clients/semantic_scholar.py](src/not_an_llm/clients/semantic_scholar.py): API client
- [src/not_an_llm/clients/arxiv.py](src/not_an_llm/clients/arxiv.py): arXiv API client
- [src/not_an_llm/clients/medarxiv.py](src/not_an_llm/clients/medarxiv.py): medRxiv API client
- [src/not_an_llm/pipelines/collect.py](src/not_an_llm/pipelines/collect.py): ingestion pipeline
- [src/not_an_llm/pipelines/preprocess.py](src/not_an_llm/pipelines/preprocess.py): preprocessing pipeline
- [src/not_an_llm/pipelines/analyze.py](src/not_an_llm/pipelines/analyze.py): feature/readability analysis pipeline (respects generate_plots flag)
- [src/not_an_llm/pipelines/visualize.py](src/not_an_llm/pipelines/visualize.py): plot generation pipeline (reads precomputed analysis data)
- [src/not_an_llm/pipelines/external_preprocess.py](src/not_an_llm/pipelines/external_preprocess.py): HC3/MAGE paired human/AI preprocessing pipelines
- [src/not_an_llm/pipelines/external_analyze.py](src/not_an_llm/pipelines/external_analyze.py): non-temporal external human-vs-AI comparison pipeline
- [src/not_an_llm/preprocessing/text.py](src/not_an_llm/preprocessing/text.py): text preprocessing class
- [src/not_an_llm/analysis/feature_extractor.py](src/not_an_llm/analysis/feature_extractor.py): style feature extraction class
- [src/not_an_llm/analysis/readability.py](src/not_an_llm/analysis/readability.py): readability metrics class
- [src/not_an_llm/analysis/feature_selection.py](src/not_an_llm/analysis/feature_selection.py): shared feature selection rules used by analyze, visualize, and external analysis
- [src/not_an_llm/analysis/feature_groups.py](src/not_an_llm/analysis/feature_groups.py): canonical grouped feature lists for diff and stack plots
- [src/not_an_llm/analysis/trends.py](src/not_an_llm/analysis/trends.py): yearly/monthly trend aggregation and exploratory plotting
- [src/not_an_llm/analysis/interrupted_time_series.py](src/not_an_llm/analysis/interrupted_time_series.py): monthly segmented regression, HAC standard errors, FDR correction, and placebo checks
- [src/not_an_llm/analysis/topic_modeling/](src/not_an_llm/analysis/topic_modeling/): BERTopic/HDBSCAN topic assignment, topic summaries, hierarchical merge candidates for review, and topic-level reports
- [src/not_an_llm/analysis/features.py](src/not_an_llm/analysis/features.py): analysis hypotheses scaffold

### Statistical Design

The paper-facing temporal test is monthly interrupted time series, not the older yearly pre/post mean comparison.

For each feature, the model estimates:

`feature_rate_t = beta0 + beta1 * time + beta2 * post_chatgpt + beta3 * time_after_chatgpt + error_t`

Interpretation:

1. `beta1` is the pre-ChatGPT slope.
2. `beta2` is the immediate level shift after the intervention date.
3. `beta3` is the slope change after the intervention date.
4. `beta1 + beta3` is the post-ChatGPT slope.

The main hypothesis tests use `slope_change_per_year`, its 95% confidence interval, `slope_change_p`, and family-level Benjamini-Hochberg `slope_change_q` from `data/analysis/<stem>_its_stats.csv`. Models are weighted by monthly paper count and use HAC/Newey-West style standard errors to reduce overconfidence from autocorrelated monthly residuals. Placebo intervention years are written to `data/analysis/<stem>_its_placebo_stats.csv`.

The pre/post diff plots and `feature_stats.csv` are retained as exploratory summaries for visual scanning; they are not the primary inferential result and do not carry the paper's hypothesis-test annotations. The inferential plot is `its_slope_changes.png`, annotated with delta slope/year, 95% CI, and q-value.

### Topic Modeling Control

Topic modeling is a single BERTopic/HDBSCAN assignment pass:

1. SentenceTransformer embeddings are encoded for each document.
2. UMAP projects embeddings to `topic_modeling_umap_n_components` dimensions for clustering.
3. HDBSCAN forms topic clusters using `topic_modeling_min_cluster_ratio` to derive `min_cluster_size` and `topic_modeling_hdbscan_min_samples` to control outlier strictness.
4. BERTopic labels topics with `topic_modeling_top_n_terms` and reduces the final topic count with `topic_modeling_max_final_topics`.
5. A separate `topic_modeling_plot_umap_n_components` projection is used for `topic_clusters.png`.
6. HDBSCAN outliers are preserved as `topic_id = -1` with label `-1 (outliers)` instead of being remapped to an ordinary topic.
7. Topic labels are saved in `data/analysis/<stem>_topic_labels.csv`, topic counts in `data/analysis/<stem>_topic_summary.csv`, optional hierarchy review data in `data/analysis/<stem>_topic_merge_candidates.csv`, and dataset/topic counts in `data/analysis/<stem>_topic_modeling_stats.csv`.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): brief module map
- [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md): brief study logic
