Modular research pipeline for studying changes in scientific writing before and after broad LLM adoption.

This repository supports the paper's arXiv/medRxiv analysis of post-ChatGPT stylistic drift in scientific preprint titles and abstracts. The primary workflow collects preprint metadata, preprocesses text, extracts lexical/readability/syntactic features, fits monthly interrupted time-series models, and generates publication figures.

## Experiment Control

The full run is controlled by [config.toml](config.toml).
The mini run is controlled by [config_mini.toml](config_mini.toml).

Edit the matching file to control:

1. collection source, categories, date range, and output paths
2. intervention date and pre/post periods
3. feature lists used for plots and statistics
4. topic modeling settings

## Run

### Full pipeline

1. Collect records:
   ```bash
   uv run python main.py --config config.toml collect
   ```

2. Preprocess text:
   ```bash
   uv run python main.py --config config.toml preprocess
   ```

3. Analyze features and write statistics:
   ```bash
   uv run python main.py --config config.toml analyze
   ```

4. Regenerate plots from saved analysis tables:
   ```bash
   uv run python main.py --config config.toml visualize
   ```

The analysis and visualization steps are separated so that feature extraction and topic modeling can be run once, while figures can be regenerated quickly after styling changes.

### Mini Dataset

The mini configuration is intended for quick checks of the pipeline.

```bash
uv run python main.py --config config_mini.toml preprocess
uv run python main.py --config config_mini.toml analyze
uv run python main.py --config config_mini.toml visualize
```

## Main Outputs

- `data/analyzed/<stem>_features.jsonl`: document-level feature output
- `data/analysis/<stem>_year.csv`: yearly feature means
- `data/analysis/<stem>_month.csv`: monthly feature means
- `data/analysis/<stem>_its_stats.csv`: primary interrupted time-series statistics
- `data/analysis/<stem>_its_placebo_stats.csv`: placebo interrupted time-series checks
- `data/analysis/<stem>_topic_*.csv`: topic labels, prevalence, and summaries
- `data/analysis/<stem>_topics/topic_*/`: topic-level trend tables
- `data/visuals/<stem>/`: rendered figures

The paper-facing inferential tables are the monthly interrupted time-series outputs in `data/analysis/<stem>_its_stats.csv`. Pre/post percentage-change plots are retained as descriptive summaries.

## Statistical Design

For each feature, the primary model is a monthly interrupted time series:

```text
feature_t = beta0 + beta1 * time + beta2 * post_chatgpt + beta3 * time_after_chatgpt + error_t
```

Interpretation:

1. `beta0` is the fitted baseline level.
2. `beta1` is the pre-intervention monthly slope.
3. `beta2` is the post-intervention level shift.
4. `beta3` is the post-intervention slope change.
5. `beta1 + beta3` is the post-intervention monthly slope.

The main tests use `slope_change_per_year`, its 95% confidence interval, `slope_change_p`, and family-level Benjamini-Hochberg `slope_change_q`. Models are weighted by monthly paper count and use HAC/Newey-West style standard errors for autocorrelated monthly residuals. Standardized effect sizes divide the annualized slope change by the pre-intervention monthly standard deviation.

## Topic Modeling

Topic modeling uses BERTopic with configurable clustering:

1. SentenceTransformer embeddings are encoded for each document.
2. UMAP projects embeddings for clustering.
3. HDBSCAN or KMeans assigns topic labels, depending on configuration.
4. BERTopic labels topics with top c-TF-IDF terms.
5. Topic-level ITS models are fit on the same feature set used in the corpus-level analysis.

Outputs include topic labels, yearly prevalence, topic-level feature trends, and cross-topic standardized ITS heatmaps.

## Repository Structure

- [config.toml](config.toml): full-run configuration
- [config_mini.toml](config_mini.toml): mini-run configuration
- [main.py](main.py): root entry point
- [src/not_an_llm/config.py](src/not_an_llm/config.py): typed config loading
- [src/not_an_llm/cli.py](src/not_an_llm/cli.py): CLI commands
- [src/not_an_llm/clients/arxiv.py](src/not_an_llm/clients/arxiv.py): arXiv API client
- [src/not_an_llm/clients/medarxiv.py](src/not_an_llm/clients/medarxiv.py): medRxiv API client
- [src/not_an_llm/pipelines/collect.py](src/not_an_llm/pipelines/collect.py): collection pipeline
- [src/not_an_llm/pipelines/preprocess.py](src/not_an_llm/pipelines/preprocess.py): preprocessing pipeline
- [src/not_an_llm/pipelines/analyze.py](src/not_an_llm/pipelines/analyze.py): feature extraction, trend aggregation, ITS, and topic analysis
- [src/not_an_llm/pipelines/visualize.py](src/not_an_llm/pipelines/visualize.py): plot generation from saved analysis tables
- [src/not_an_llm/analysis/feature_extractor.py](src/not_an_llm/analysis/feature_extractor.py): lexical, punctuation, discourse, and syntactic feature extraction
- [src/not_an_llm/analysis/readability.py](src/not_an_llm/analysis/readability.py): readability metrics
- [src/not_an_llm/analysis/feature_selection.py](src/not_an_llm/analysis/feature_selection.py): shared feature-selection rules
- [src/not_an_llm/analysis/feature_groups.py](src/not_an_llm/analysis/feature_groups.py): canonical feature families
- [src/not_an_llm/analysis/trends.py](src/not_an_llm/analysis/trends.py): yearly/monthly trend aggregation and plotting
- [src/not_an_llm/analysis/interrupted_time_series.py](src/not_an_llm/analysis/interrupted_time_series.py): segmented regression, HAC standard errors, FDR correction, and placebo checks
- [src/not_an_llm/analysis/topic_modeling/](src/not_an_llm/analysis/topic_modeling/): BERTopic assignment, topic summaries, and topic-level reports
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): brief module map
- [docs/VISUALS_AND_ANALYSIS.md](docs/VISUALS_AND_ANALYSIS.md): output and visualization reference

<!-- ## Anonymous Release Notes

The repository is prepared for anonymous review. Runtime logs, local paths, raw/preprocessed JSONL data, and exploratory notebooks should not be included in the public artifact. Generated aggregate analysis tables and rendered figures may be included when needed for reproducibility. -->
