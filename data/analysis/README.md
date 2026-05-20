# Analysis Tables

This directory is the only home for CSV analysis artifacts. Plot directories under `data/visuals/` should contain rendered figures only.

The feature-enriched document JSONL files live in `data/analyzed/` because they are large intermediate datasets rather than summary statistics.

## Core Temporal Tables

### `<stem>_year.csv`

Year-level feature trends. Each row is one publication year.

- `year`: publication year.
- `paper_count`: number of abstracts in that year.
- `<feature>_yearly_mean`: mean value of a selected analysis feature across abstracts in that year.
- `<feature>_yearly_std`: across-abstract standard deviation for that feature in that year.
- `<feature>_yearly_n`: number of non-missing abstracts contributing to that feature mean.

Use this table for descriptive trend inspection and yearly plots. It is not the main inferential test because yearly samples are few and the ChatGPT release happened within a year.

### `<stem>_month.csv`

Month-level feature trends. Each row is one publication month.

- `month_ts`: first day of the publication month.
- `paper_count`: number of abstracts in that month.
- `<feature>_monthly_mean`: mean value of a selected analysis feature across abstracts in that month.
- `year`, `month`: numeric calendar fields derived from `month_ts`.

Use this table for higher-resolution visual trends and as the input to the interrupted time-series model.

## Primary Statistical Tables

### `<stem>_its_stats.csv`

Primary hypothesis-test table. Each row is one configured feature in one feature family.

- `feature`: measured feature name from the configured analysis feature list.
- `family`: feature family used for within-family false-discovery-rate correction.
- `intervention_date`: intervention date used by the model, currently the ChatGPT release date.
- `n_months`, `n_pre_months`, `n_post_months`: total, pre-intervention, and post-intervention monthly observations used.
- `pre_mean`, `post_mean`: average monthly feature value before and after the intervention.
- `pre_slope_per_month`, `pre_slope_per_year`: estimated pre-intervention trend.
- `level_shift`: immediate post-intervention jump or drop in the feature level.
- `level_shift_se`, `level_shift_ci_low`, `level_shift_ci_high`, `level_shift_p`, `level_shift_q`: uncertainty, confidence interval, p-value, and FDR-adjusted q-value for the immediate level shift.
- `slope_change_per_month`, `slope_change_per_year`: change in trend after the intervention. This is the main estimate for temporal acceleration or deceleration.
- `slope_change_se`, `slope_change_per_year_se`, `slope_change_per_year_ci_low`, `slope_change_per_year_ci_high`, `slope_change_p`, `slope_change_q`: uncertainty, confidence interval, p-value, and FDR-adjusted q-value for the slope change.
- `post_slope_per_month`, `post_slope_per_year`: estimated post-intervention trend, equal to the pre-slope plus the slope change.
- `r_squared`: descriptive model fit.
- `model`: model specification label.
- `hac_lags`: number of HAC/Newey-West lags used for autocorrelation-robust standard errors.

Read `slope_change_per_year` with its confidence interval and `slope_change_q` as the main result. Positive values mean the feature increased faster after the intervention; negative values mean it slowed or declined relative to the pre-intervention trend.

### `<stem>_its_placebo_stats.csv`

Robustness table using the same interrupted time-series model with earlier placebo intervention years.

It has the same interpretation as `<stem>_its_stats.csv`, plus:

- `placebo_year`: artificial intervention year being tested.

Use this table to check whether the ChatGPT-era slope change is unusual relative to arbitrary earlier breaks. Placebo results are supporting evidence, not the main estimate.

## Topic Tables

### `<stem>_topic_labels.csv`

Topic id to topic label lookup.

- `topic_id`: numeric topic assignment. `-1` is the outlier bucket when HDBSCAN leaves documents unassigned.
- `topic_label`: human-readable label built from top topic terms.

Use this table to decode topic ids in all topic outputs.

### `<stem>_topic_summary.csv`

Dataset-level topic size summary.

- `topic_id`: numeric topic assignment.
- `abstract_count`: number of abstracts assigned to the topic.
- `abstract_share`: share of all analyzed abstracts assigned to the topic.
- `topic_label`: readable topic label.

Use this table to understand topic prevalence across the whole dataset.

### `<stem>_topic_modeling_stats.csv`

Compact topic-model health and size table.

- `scope`: `dataset` for whole-dataset metrics or `topic` for per-topic metrics.
- `stage`: modeling stage label.
- `topic_id`, `topic_label`: topic identifiers when the metric is topic-specific.
- `metric`: statistic name, such as `total_abstracts`, `topics`, `outlier_abstract_count`, `outlier_abstract_share`, `abstract_count`, or `abstract_share`.
- `value`: metric value.

Use this table to report how many topics were produced, how large the outlier bucket is, and how large each topic is.

### `<stem>_topic_merge_candidates.csv`

Optional hierarchy-review table for candidate topic merges.

- `Parent_ID`, `Child_Left_ID`, `Child_Right_ID`: hierarchy node ids from the topic model.
- `Distance`: semantic distance between merged branches; lower values are closer.
- `child_left_abstract_count`, `child_right_abstract_count`, `parent_abstract_count`: topic sizes for review.
- `child_left_abstract_share`, `child_right_abstract_share`, `parent_abstract_share`: corresponding corpus shares.

Use this only to review whether semantically close topics should be merged or relabeled.

### `<stem>_topic_prevalence_yearly.csv`

Yearly topic prevalence.

- `year`: publication year.
- `topic_id`: topic assignment.
- `count`: abstracts in that year and topic.
- `total`: all analyzed abstracts in that year.
- `pct`: `count / total * 100`.
- `topic_label`: readable topic label.

Use this table to read yearly topic-mix changes.

### `<stem>_topic_prevalence_monthly.csv`

Monthly topic prevalence, written when month-level timestamps are available.

- `month_ts`: first day of the publication month.
- `topic_id`: topic assignment.
- `count`: abstracts in that month and topic.
- `total`: all analyzed abstracts in that month.
- `pct`: `count / total * 100`.
- `topic_label`: readable topic label.

Use this table for monthly topic-mix plots and short-term shifts.

## Topic Subdirectories

`<stem>_topics/topic_<id>/` contains per-topic trend tables:

- `trends_by_year.csv`: same structure and interpretation as `<stem>_year.csv`, but restricted to one topic.
- `trends_by_month.csv`: same structure and interpretation as `<stem>_month.csv`, but restricted to one topic.

These files are descriptive topic slices. They do not include separate `feature_stats.csv` files because the retained inferential statistics are the dataset-level interrupted time-series tables.
