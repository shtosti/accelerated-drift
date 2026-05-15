from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor

from not_an_llm.analysis.feature_extractor import FeatureExtractor, _slugify
from not_an_llm.analysis.readability import ReadabilityAnalyzer
from not_an_llm.analysis.trends import TrendAnalyzer
from not_an_llm.config import AppConfig

from .feature_groups import FEATURE_GROUPS
from .label_map import LABEL_MAP


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AnalysisArtifacts:
    feature_dataset_jsonl: Path
    trends_csv: Path
    monthly_trends_csv: Path
    trends_plot_paths: list[Path]


# =========================================================
# PATH RESOLUTION
# =========================================================
def _resolve_analysis_paths(config: AppConfig):
    # Derive output names from the preprocessed input file basename to distinguish mini/full runs
    input_stem = config.analysis.preprocessed_jsonl.stem
    analysis_dir = Path(config.data_dir) / "analysis"
    visuals_dir = Path(config.data_dir) / "visuals"

    analysis_dir.mkdir(parents=True, exist_ok=True)

    def _maybe(path_value: Path, default: Path) -> Path:
        # Use default if path_value is empty or matches the old analysis/plots default
        if not path_value or str(path_value).startswith("data/visuals"):
            return default
        return path_value

    feature_dataset = _maybe(
        config.analysis.feature_dataset_jsonl,
        analysis_dir / f"{input_stem}.jsonl",
    )

    trends_csv = _maybe(
        config.analysis.trends_csv,
        analysis_dir / f"{input_stem}_trends_by_year.csv",
    )

    monthly_csv = _maybe(
        config.analysis.monthly_trends_csv,
        analysis_dir / f"{input_stem}_trends_by_month.csv",
    )

    plot_dir = _maybe(
        config.analysis.trends_plot_dir,
        visuals_dir / input_stem,
    )

    return feature_dataset, trends_csv, monthly_csv, plot_dir


# =========================================================
# MARKER GROUPS
# =========================================================
def _build_marker_group_specs(config: AppConfig):
    group_specs: dict[str, dict[str, object]] = {}
    summary_features: set[str] = set()

    group_definitions = {
        "marker_words": (
            "Marker words",
            "word",
            config.analysis.llm_marker_words,
            "marker_words_total_per_1k_words",
            "marker_words_total",
        ),
        "marker_verbs": (
            "Marker verbs",
            "verb",
            config.analysis.llm_marker_verbs,
            "marker_verbs_total_per_1k_words",
            "marker_verbs_total",
        ),
        "marker_adjectives": (
            "Marker adjectives",
            "adjective",
            config.analysis.llm_marker_adjectives,
            "marker_adjectives_total_per_1k_words",
            "marker_adjectives_total",
        ),
        "marker_phrases": (
            "Marker phrases",
            "phrase",
            config.analysis.llm_marker_phrases,
            "marker_phrases_total_per_1k_words",
            "marker_phrases_total",
        ),
        "sequential_markers": (
            "Sequential markers",
            "sequential_marker",
            config.analysis.sequential_markers,
            "sequential_markers_total_per_1k_words",
            "sequential_markers_total",
        ),
        "causal_markers": (
            "Causal markers",
            "causal_marker",
            config.analysis.causal_markers,
            "causal_markers_total_per_1k_words",
            "causal_markers_total",
        ),
        "contrast_markers": (
            "Contrast markers",
            "contrast_marker",
            config.analysis.contrast_markers,
            "contrast_markers_total_per_1k_words",
            "contrast_markers_total",
        ),
        "emphasis_markers": (
            "Emphasis markers",
            "emphasis_marker",
            config.analysis.emphasis_markers,
            "emphasis_markers_total_per_1k_words",
            "emphasis_markers_total",
        ),
        "summary_markers": (
            "Summary markers",
            "summary_marker",
            config.analysis.summary_markers,
            "summary_markers_total_per_1k_words",
            "summary_markers_total",
        ),
    }

    for group_name, (label, prefix, terms, rate_feature, count_feature) in group_definitions.items():
        group_specs[group_name] = {
            "label": label,
            "rate_feature": rate_feature,
            "count_feature": count_feature,
        }
        summary_features.add(rate_feature)
        summary_features.add(count_feature)

    summary_features.add("marker_density")
    return group_specs, summary_features


def _format_xticks(ax):
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def _assign_topics(
    enriched: pd.DataFrame,
    config: AppConfig,
    plot_dir: Path,
) -> tuple[pd.DataFrame, dict[int, str], pd.DataFrame | None]:
    if not config.analysis.topic_modeling_enabled:
        return enriched, {}, None

    if "text_clean" not in enriched.columns:
        raise ValueError("Topic modeling requires a 'text_clean' column.")

    texts = enriched["text_clean"].fillna("").astype(str).tolist()
    num_topics = config.analysis.topic_modeling_num_topics
    top_terms = config.analysis.topic_modeling_top_n_terms

    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    import torch
    import time
    import umap
    import hdbscan
    import multiprocessing

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = config.analysis.topic_modeling_num_workers or multiprocessing.cpu_count()
    num_workers = max(1, min(num_workers, multiprocessing.cpu_count()))
    
    # UMAP/numba has a hard limit of 16 threads
    umap_n_jobs = min(num_workers, 16)

    logger.info(
        f"Using device: {device} for topic modeling ({len(texts)} documents, {num_topics} topics)"
    )
    logger.info(f"Using {umap_n_jobs} threads for UMAP, {num_workers} workers for HDBSCAN")

    start_time = time.time()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    logger.info("Encoding document embeddings...")
    embeddings = embedding_model.encode(
        texts,
        batch_size=config.analysis.topic_modeling_embedding_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    vectorizer_model = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=1,
    )

    umap_model = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.0,
        metric='cosine',
        random_state=42,
        n_jobs=umap_n_jobs,
    )
    hdbscan_model = hdbscan.HDBSCAN(
        min_cluster_size=5,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
        core_dist_n_jobs=num_workers,
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        language="english",
        nr_topics="auto",
        top_n_words=top_terms,
        min_topic_size=1,  # Minimum documents per topic
        calculate_probabilities=True,
        verbose=False,
    )

    try:
        topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
        topic_model.reduce_topics(texts, nr_topics=num_topics)
        topics = topic_model.topics_
            
        embedding_time = time.time() - start_time
        logger.info(f"Topic modeling completed in {embedding_time:.2f} seconds")
        
        original_topics = sorted(set(topics))
        has_noise = -1 in original_topics
        topic_id_map = {topic_id: topic_id for topic_id in original_topics if topic_id != -1}
        if has_noise:
            next_topic_id = max(topic_id_map.keys(), default=-1) + 1
            topic_id_map[-1] = next_topic_id
        actual_topics = sorted(topic_id_map.values())

        logger.info(f"Created {len(actual_topics)} topics (requested {num_topics})")
        if len(actual_topics) != num_topics:
            logger.warning(f"BERTopic created {len(actual_topics)} topics instead of requested {num_topics}")
            if len(actual_topics) < num_topics:
                logger.warning("Consider reducing nr_topics or adjusting min_topic_size")

        # Extract topic labels from fitted model, including remapped outliers if present
        stop_words = set(vectorizer_model.get_stop_words() or [])
        topic_labels: dict[int, str] = {}
        for target_topic_id in actual_topics:
            source_topic_id = -1 if has_noise and topic_id_map.get(-1) == target_topic_id else target_topic_id
            terms = [
                term for term, _ in topic_model.get_topic(source_topic_id)
                if term.isalpha() and term.lower() not in stop_words
            ]
            if not terms:
                terms = [term for term, _ in topic_model.get_topic(source_topic_id)]
            topic_labels[target_topic_id] = ", ".join(terms[:top_terms]) or f"topic_{target_topic_id}"

    except Exception as e:
        import traceback
        logger.error(f"Topic modeling failed: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        # Fallback: assign all documents to topic 0
        topics = [0] * len(texts)
        topic_labels = {0: "fallback_topic"}
        topic_id_map = {0: 0}
        logger.warning("Using fallback topic assignment")

    labels = [topic_id_map.get(int(t), int(t)) for t in topics]

    enriched = enriched.copy()
    enriched["topic_id"] = labels
    enriched["topic_label"] = [topic_labels.get(int(t), "noise") for t in labels]

    analysis_dir = Path(config.data_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    topic_labels_path = analysis_dir / f"{config.analysis.preprocessed_jsonl.stem}_topic_labels.csv"
    pd.DataFrame(
        {
            "topic_id": list(topic_labels.keys()),
            "topic_label": list(topic_labels.values()),
        }
    ).to_csv(topic_labels_path, index=False)

    # Extract 2D embeddings for cluster visualization
    embeddings_2d = None
    try:
        if hasattr(topic_model, 'umap_model') and hasattr(topic_model.umap_model, 'embedding_'):
            embeddings_2d = pd.DataFrame(
                topic_model.umap_model.embedding_,
                columns=['x', 'y']
            )
            embeddings_2d['topic_id'] = labels
            embeddings_2d['topic_label'] = [topic_labels.get(int(t), "noise") for t in labels]
    except Exception as e:
        logger.warning(f"Could not extract 2D embeddings: {e}")

    return enriched, topic_labels, embeddings_2d


from pathlib import Path
import matplotlib.pyplot as plt


def save_legend_only(ax, output_path: Path, ncol: int = 1):
    """
    Save legend from an axis as a standalone image.
    """

    handles, labels = ax.get_legend_handles_labels()

    if not handles:
        return

    # Create separate legend figure
    fig_legend = plt.figure(figsize=(4, 2))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis("off")

    legend = ax_legend.legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        ncol=ncol,
        fontsize=8,
    )

    # Resize figure tightly around legend
    fig_legend.canvas.draw()
    bbox = legend.get_window_extent().transformed(
        fig_legend.dpi_scale_trans.inverted()
    )

    fig_legend.savefig(
        output_path,
        dpi=250,
        bbox_inches=bbox
    )

    plt.close(fig_legend)


def _save_topic_prevalence(
    enriched: pd.DataFrame,
    plot_dir: Path,
    analysis_dir: Path,
    input_stem: str,
    topic_labels: dict[int, str],
) -> list[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if "year" not in enriched.columns:
        raise ValueError("Topic prevalence requires a 'year' column.")

    df = enriched.copy()
    if "month_ts" not in df.columns and "year" in df.columns and "month" in df.columns:
        df["month_ts"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-01",
            errors="coerce",
        )

    df["count"] = 1
    yearly = (
        df.groupby(["year", "topic_id"], as_index=False)["count"].sum()
    )
    totals = df.groupby("year", as_index=False)["count"].sum().rename(columns={"count": "total"})
    yearly = yearly.merge(totals, on="year")
    yearly["pct"] = yearly["count"] / yearly["total"] * 100
    yearly["topic_label"] = yearly["topic_id"].map(topic_labels)

    yearly_csv = analysis_dir / f"{input_stem}_topic_prevalence_yearly.csv"
    yearly.to_csv(yearly_csv, index=False)
    paths.append(yearly_csv)

    yearly_pivot = yearly.pivot(index="year", columns="topic_label", values="pct").fillna(0.0)
    
    # Line plot
    fig, ax = plt.subplots(figsize=(5, 3))
    for label in yearly_pivot.columns:
        ax.plot(yearly_pivot.index, yearly_pivot[label], marker="o", label=label)
    ax.set_xlabel("Year")
    ax.set_ylabel("Topic prevalence (%)")
    # ax.set_title("Topic prevalence by year (line plot)")
    legend = ax.legend(loc="best", fontsize=8)
    legend_path = plot_dir / "topic_prevalence_legend.png"
    save_legend_only(ax, legend_path)
    legend.remove()

    ax.grid(alpha=0.3)
    _format_xticks(ax)
    trend_path = plot_dir / "topic_prevalence_yearly.png"
    fig.tight_layout()
    fig.savefig(trend_path, dpi=150)
    plt.close(fig)
    paths.append(trend_path)

    # Stacked area plot
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.stackplot(
        yearly_pivot.index,
        *[yearly_pivot[col] for col in yearly_pivot.columns],
        labels=yearly_pivot.columns,
        alpha=0.8,
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Topic prevalence (%)")
    # ax.set_title("Topic evolution over time (stacked area)")
    legend = ax.legend(loc="best", fontsize=8)
    legend_path = plot_dir / "topic_evolution_stacked_legend.png"
    save_legend_only(ax, legend_path)
    legend.remove()

    ax.grid(alpha=0.3, axis="y")
    _format_xticks(ax)
    stacked_path = plot_dir / "topic_evolution_stacked_yearly.png"
    fig.tight_layout()
    fig.savefig(stacked_path, dpi=150)
    plt.close(fig)
    paths.append(stacked_path)

    if "month_ts" in enriched.columns:
        monthly = (
            df.groupby(["month_ts", "topic_id"], as_index=False)["count"].sum()
        )
        month_totals = df.groupby("month_ts", as_index=False)["count"].sum().rename(columns={"count": "total"})
        monthly = monthly.merge(month_totals, on="month_ts")
        monthly["pct"] = monthly["count"] / monthly["total"] * 100
        monthly["topic_label"] = monthly["topic_id"].map(topic_labels)

        monthly_csv = analysis_dir / f"{input_stem}_topic_prevalence_monthly.csv"
        monthly.to_csv(monthly_csv, index=False)
        paths.append(monthly_csv)

        monthly_pivot = monthly.pivot(index="month_ts", columns="topic_label", values="pct").fillna(0.0)
        fig, ax = plt.subplots(figsize=(5, 3))
        for label in monthly_pivot.columns:
            ax.plot(pd.to_datetime(monthly_pivot.index), monthly_pivot[label], marker=".", label=label)
        ax.set_xlabel("Month")
        ax.set_ylabel("Topic prevalence (%)")
        # ax.set_title("Topic prevalence by month")
        legend = ax.legend(loc="best", fontsize=8)
        legend_path = plot_dir / "topic_prevalence_monthly_legend.png"
        save_legend_only(ax, legend_path)
        legend.remove()
        ax.grid(alpha=0.3)
        _format_xticks(ax)
        monthly_path = plot_dir / "topic_prevalence_monthly.png"
        fig.tight_layout()
        fig.savefig(monthly_path, dpi=150)
        plt.close(fig)
        paths.append(monthly_path)

    return paths


def _save_topic_trend_plots(
    enriched: pd.DataFrame,
    plot_dir: Path,
    topic_labels: dict[int, str],
    events: dict[str, str],
) -> list[Path]:
    """Create trend plots for each feature showing lines for each topic."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if "year" not in enriched.columns:
        return paths

    # Get all numeric feature columns (excluding metadata)
    feature_columns = [
        col for col in enriched.columns
        if col.endswith("_per_1k_words") and pd.api.types.is_numeric_dtype(enriched[col])
    ]

    if not feature_columns:
        return paths

    # Group by year and topic, compute means
    grouped = enriched.groupby(["year", "topic_id"])[feature_columns].mean().reset_index()
    grouped["topic_label"] = grouped["topic_id"].map(topic_labels)

    event_dates = {k: pd.to_datetime(v) for k, v in events.items()}

    for feature in feature_columns:
        fig, ax = plt.subplots(figsize=(5, 4))

        # Plot line for each topic
        for topic_id, topic_data in grouped.groupby("topic_id"):
            topic_label = topic_labels.get(topic_id, f"topic_{topic_id}")
            topic_years = topic_data["year"]
            topic_values = topic_data[feature]

            ax.plot(
                topic_years,
                topic_values,
                marker="o",
                linewidth=1.5,
                label=topic_label,
                alpha=0.8
            )

        # Add event lines
        for event_name, event_date in event_dates.items():
            if event_date.year in grouped["year"].values:
                ax.axvline(x=event_date.year, color="red", linestyle="--", alpha=0.7, label=event_name)

        ax.set_xlabel("Year")
        ax.set_ylabel(f"{feature.replace('_per_1k_words', '').replace('_', ' ').replace('Word', '').title()}")
        # ax.set_title(f"{feature.replace('_per_1k_words', '').replace('_', ' ').title()} by Topic")
        legend = ax.legend(loc="best", fontsize=8)
        legend_path = plot_dir / f"{feature}_by_topic_legend.png"
        save_legend_only(ax, legend_path)
        legend.remove()

        ax.grid(alpha=0.3)
        _format_xticks(ax)

        plot_path = plot_dir / f"{feature}_by_topic.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        paths.append(plot_path)

    return paths


def _save_topic_cluster_plot(
    embeddings_2d: pd.DataFrame,
    plot_dir: Path,
) -> Path | None:
    """Create a 2D cluster plot showing spatial distribution of topics."""
    if embeddings_2d is None or embeddings_2d.empty:
        return None

    plot_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Get unique topics (excluding noise)
    topics = sorted(embeddings_2d['topic_id'].unique())
    topics = [t for t in topics if t != -1]  # Exclude noise
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(topics)))
    color_map = dict(zip(topics, colors))
    
    # Plot noise points first (light gray)
    noise_data = embeddings_2d[embeddings_2d['topic_id'] == -1]
    if not noise_data.empty:
        ax.scatter(
            noise_data['x'], 
            noise_data['y'], 
            c='lightgray', 
            alpha=0.5, 
            s=20, 
            label='noise'
        )
    
    # Plot each topic
    for topic_id in topics:
        topic_data = embeddings_2d[embeddings_2d['topic_id'] == topic_id]
        if not topic_data.empty:
            topic_label = topic_data['topic_label'].iloc[0]
            ax.scatter(
                topic_data['x'], 
                topic_data['y'], 
                c=[color_map[topic_id]], 
                alpha=0.7, 
                s=30, 
                label=f'{topic_id}: {topic_label}',
                edgecolors='black',
                linewidth=0.5
            )
    
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    # ax.set_title('Topic Clusters (UMAP 2D Projection)')
    legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    legend_path = plot_dir / "topic_clusters_legend.png"
    save_legend_only(ax, legend_path)
    legend.remove()
    
    ax.grid(alpha=0.3)
    
    plot_path = plot_dir / "topic_clusters.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return plot_path


def _run_topic_analysis(
    enriched: pd.DataFrame,
    config: AppConfig,
    plot_dir: Path,
    trend_analyzer: TrendAnalyzer,
    group_specs: dict[str, dict[str, object]],
    summary_features: set[str],
    events: dict[str, str],
    embeddings_2d: pd.DataFrame | None = None,
) -> list[Path]:
    logger.info("Starting topic analysis...")
    if not config.analysis.topic_modeling_enabled:
        logger.info("Topic modeling disabled, skipping topic analysis")
        return []
    
    if "topic_id" not in enriched.columns:
        logger.error("topic_id column missing from enriched data, skipping topic analysis")
        return []

    logger.info(f"Running topic analysis for {enriched['topic_id'].nunique()} topics")

    plot_dir = plot_dir / "topics"
    plot_dir.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(enriched["topic_id"].dropna().unique())
    topic_labels = {
        int(t): enriched.loc[enriched["topic_id"] == int(t), "topic_label"].iloc[0]
        for t in unique_labels
    }

    analysis_dir = Path(config.data_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_topic_base = analysis_dir / f"{config.analysis.preprocessed_jsonl.stem}_topics"
    analysis_topic_base.mkdir(parents=True, exist_ok=True)

    paths = _save_topic_prevalence(
        enriched,
        plot_dir,
        analysis_dir,
        config.analysis.preprocessed_jsonl.stem,
        topic_labels,
    )

    # Add topic trend plots
    paths.extend(_save_topic_trend_plots(enriched, plot_dir, topic_labels, events))

    # Add cluster plot
    cluster_plot_path = _save_topic_cluster_plot(embeddings_2d, plot_dir)
    if cluster_plot_path:
        paths.append(cluster_plot_path)

    for topic_id in unique_labels:
        plot_topic_dir = plot_dir / f"topic_{topic_id}"
        plot_topic_dir.mkdir(parents=True, exist_ok=True)
        analysis_topic_dir = analysis_topic_base / f"topic_{topic_id}"
        analysis_topic_dir.mkdir(parents=True, exist_ok=True)

        topic_subset = enriched[enriched["topic_id"] == topic_id]
        if topic_subset.empty:
            continue

        topic_yearly = trend_analyzer.aggregate_yearly(topic_subset)
        topic_monthly = trend_analyzer.aggregate_monthly(topic_subset)
        topic_yearly.to_csv(analysis_topic_dir / "trends_by_year.csv", index=False)
        topic_monthly.to_csv(analysis_topic_dir / "trends_by_month.csv", index=False)

        stats_df = trend_analyzer.compute_all_stats(topic_yearly)
        stats_df.to_csv(analysis_topic_dir / "feature_stats.csv", index=False)
        paths.append(analysis_topic_dir / "feature_stats.csv")

        # Always generate topic-level visualizations for cross-topic comparison
        paths.extend(
            trend_analyzer.save_plots(
                topic_yearly,
                topic_monthly,
                plot_topic_dir,
                events,
                exclude_features=summary_features,
            )
        )
        if hasattr(trend_analyzer, "save_grouped_feature_diffs"):
            paths.extend(
                trend_analyzer.save_grouped_feature_diffs(
                    topic_yearly,
                    output_dir=plot_topic_dir,
                ).values()
            )
        paths.extend(
            trend_analyzer.save_grouped_word_plots(
                yearly=topic_yearly,
                monthly=topic_monthly,
                output_dir=plot_topic_dir,
                group_specs=group_specs,
                events=events,
                smoothing_window=3,
            )
        )
        paths.append(
            trend_analyzer.save_word_prefix_stack_plot(
                yearly=topic_yearly,
                monthly=topic_monthly,
                output_dir=plot_topic_dir,
                events=events,
                smoothing_window=3,
            )
        )
        paths.extend(
            trend_analyzer.save_dependency_distribution_plot(
                df=topic_subset,
                output_dir=plot_topic_dir,
                events=events,
            )
        )
        paths.append(
            trend_analyzer.save_stacked_word_plots(
                yearly=topic_yearly,
                monthly=topic_monthly,
                output_dir=plot_topic_dir,
                events=events,
                smoothing_window=3,
                exclude_features=summary_features,
            )
        )

    return paths


# =========================================================
# WORKER (PARALLEL CHUNK PROCESSING)
# =========================================================
def _process_chunk_worker(args):
    chunk, config = args

    import spacy
    from not_an_llm.analysis.feature_extractor import FeatureExtractor
    from not_an_llm.analysis.readability import ReadabilityAnalyzer

    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    nlp.max_length = 2_000_000

    feature_extractor = FeatureExtractor(
        nlp=nlp,
        syntactic_features=config.analysis.syntactic_features,
        marker_words=config.analysis.llm_marker_words,
        marker_verbs=config.analysis.llm_marker_verbs,
        marker_adjectives=config.analysis.llm_marker_adjectives,
        marker_phrases=config.analysis.llm_marker_phrases,
        sequential_markers=config.analysis.sequential_markers,
        causal_markers=config.analysis.causal_markers,
        contrast_markers=config.analysis.contrast_markers,
        emphasis_markers=config.analysis.emphasis_markers,
        summary_markers=config.analysis.summary_markers,
        enable_list_of_three_marker=config.analysis.enable_list_of_three_marker,
        marker_word_matching=config.analysis.llm_marker_word_matching,
        hedges=config.analysis.hedge_terms,
        certainty_terms=config.analysis.certainty_terms,
    )

    readability = None
    if config.analysis.include_readability:
        readability = ReadabilityAnalyzer(metrics=config.analysis.readability_metrics)

    enriched = feature_extractor.transform(chunk)

    if readability:
        enriched = readability.transform(enriched)

    return enriched


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_analysis(config: AppConfig) -> AnalysisArtifacts:
    input_path = config.analysis.preprocessed_jsonl
    if not input_path.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found at {input_path}. Run preprocess first."
        )

    (
        enriched_output_path,
        trends_csv,
        monthly_trends_csv,
        plot_dir,
    ) = _resolve_analysis_paths(config)

    print("Saving feature dataset to:", enriched_output_path)
    print("Saving yearly trends to:", trends_csv)
    print("Saving monthly trends to:", monthly_trends_csv)
    print("Saving plots to:", plot_dir)

    # =========================
    # CHUNK LOADING
    # =========================
    chunks = pd.read_json(input_path, lines=True, chunksize=2000)
    chunks_list = list(chunks)

    enriched_output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if enriched_output_path.exists():
        enriched_output_path.unlink()

    # =========================
    # PARALLEL EXECUTION
    # =========================
    import multiprocessing

    num_workers = config.analysis.topic_modeling_num_workers or multiprocessing.cpu_count()
    num_workers = max(1, min(num_workers, multiprocessing.cpu_count()))
    logger.info(f"Using {num_workers} workers")

    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, enriched_chunk in enumerate(
            executor.map(_process_chunk_worker, [(c, config) for c in chunks_list])
        ):
            enriched_chunk.to_json(
                enriched_output_path,
                orient="records",
                lines=True,
                force_ascii=False,
                mode="w" if i == 0 else "a",
            )
            results.append(enriched_chunk)

    # =========================
    # CONCAT
    # =========================
    enriched = pd.concat(results, ignore_index=True)
    for col in enriched.columns:
        if col.endswith("_per_1k_words"):
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce").fillna(0.0)

    enriched, topic_labels, embeddings_2d = _assign_topics(enriched, config, plot_dir)

    enriched.to_json(
        enriched_output_path,
        orient="records",
        lines=True,
        force_ascii=False,
        mode="w",
    )

    feature_columns = _resolve_feature_columns(config, enriched)

    trend_analyzer = TrendAnalyzer(feature_columns)
    marker_group_specs, summary_features = _build_marker_group_specs(config)

    yearly = trend_analyzer.aggregate_yearly(enriched)

    # =========================
    # STATISTICAL ANALYSIS
    # =========================
    logger.info("Computing statistical summaries...")

    stats_df = trend_analyzer.compute_all_stats(yearly)

    stats_path = plot_dir / "feature_stats.csv"
    stats_df.to_csv(stats_path, index=False)

    logger.info("Saved statistical summary to %s", stats_path)

    # =========================
    # PRE/POST DIFF PLOTS
    # =========================
    if config.analysis.generate_plots:
        logger.info("Generating grouped pre/post diff plot...")
        diff_plot_path = trend_analyzer.save_pre_post_diff_plot(
            yearly,
            output_dir=plot_dir,
        )
        logger.info("Saved grouped pre/post diff plot to %s", diff_plot_path)

        logger.info("Generating per-group pre/post diff plots...")
        if hasattr(trend_analyzer, "save_grouped_feature_diffs"):
            grouped_diff_plots = trend_analyzer.save_grouped_feature_diffs(
                yearly,
                output_dir=plot_dir,
            )
            logger.info("Saved %d per-group diff plots", len(grouped_diff_plots))
        else:
            # Backwards-compatible fallback: compute grouped diffs inline and save plots
            logger.warning("TrendAnalyzer lacks 'save_grouped_feature_diffs'; using fallback implementation.")
            from not_an_llm.analysis.trends import save_grouped_difference_plot

            grouped_diff_plots = {}
            pre = yearly[yearly["year"] <= 2022]
            post = yearly[yearly["year"] >= 2023]

            for group_name, members in FEATURE_GROUPS.items():
                rows = []
                valid_cols = []
                for m in members:
                    col = f"{m}_yearly_mean"
                    if col not in yearly.columns:
                        continue
                    valid_cols.append(col)
                    pre_vals = pd.to_numeric(pre[col], errors="coerce")
                    post_vals = pd.to_numeric(post[col], errors="coerce")
                    if pre_vals.dropna().empty or post_vals.dropna().empty:
                        continue
                    pre_mean = float(pre_vals.mean())
                    post_mean = float(post_vals.mean())
                    scale = abs(pre_mean) + abs(post_mean)
                    pct_change = 0.0 if scale == 0.0 else 200.0 * (post_mean - pre_mean) / scale
                    rows.append({"feature": m, "diff": pct_change})

                if not rows:
                    continue

                df = pd.DataFrame(rows)

                df = df.sort_values("diff")
                out_path = plot_dir / f"{group_name}_diff.png"
                save_grouped_difference_plot(
                    df,
                    out_path,
                    "feature",
                    "diff",
                    "Percent change (%)",
                    LABEL_MAP,
                )
                grouped_diff_plots[group_name] = out_path
            logger.info("Saved %d per-group diff plots (fallback)", len(grouped_diff_plots))

    monthly = trend_analyzer.aggregate_monthly(enriched)

    # =========================
    # SAVE TABLES
    # =========================
    trends_csv.parent.mkdir(parents=True, exist_ok=True)
    yearly.to_csv(trends_csv, index=False)

    monthly_trends_csv.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(monthly_trends_csv, index=False)

    # =========================
    # PLOTS
    # =========================
    llm_events = {
        "ChatGPT release": "2022-11-30",
    }

    trend_plots = []
    if config.analysis.generate_plots:
        trend_plots = trend_analyzer.save_plots(
            yearly,
            monthly,
            plot_dir,
            llm_events,
            exclude_features=summary_features,
        )

        trend_plots.extend(
            trend_analyzer.save_grouped_word_plots(
                yearly=yearly,
                monthly=monthly,
                output_dir=plot_dir,
                group_specs=marker_group_specs,
                events=llm_events,
                smoothing_window=3,
            )
        )

        word_stack_plot_path = trend_analyzer.save_word_prefix_stack_plot(
            yearly=yearly,
            monthly=monthly,
            output_dir=plot_dir,
            events=llm_events,
            smoothing_window=3,
        )
        trend_plots.append(word_stack_plot_path)

        dep_plot_paths = trend_analyzer.save_dependency_distribution_plot(
            df=enriched,
            output_dir=plot_dir,
            events=llm_events,
        )

        trend_plots.extend(dep_plot_paths)

        stacked_plot_path = trend_analyzer.save_stacked_word_plots(
            yearly=yearly,
            monthly=monthly,
            output_dir=plot_dir,
            events=llm_events,
            smoothing_window=3,
            exclude_features=summary_features,
        )

        trend_plots.append(stacked_plot_path)

    if config.analysis.topic_modeling_enabled:
        topic_paths = _run_topic_analysis(
            enriched=enriched,
            config=config,
            plot_dir=plot_dir,
            trend_analyzer=trend_analyzer,
            group_specs=marker_group_specs,
            summary_features=summary_features,
            events=llm_events,
            embeddings_2d=embeddings_2d,
        )
        trend_plots.extend(topic_paths)

    return AnalysisArtifacts(
        feature_dataset_jsonl=enriched_output_path,
        trends_csv=trends_csv,
        monthly_trends_csv=monthly_trends_csv,
        trends_plot_paths=trend_plots,
    )


# =========================================================
# FEATURE SELECTION (UNCHANGED)
# =========================================================
def _resolve_feature_columns(config: AppConfig, frame: pd.DataFrame) -> list[str]:
    metadata_exclusions = {
        "year",
        "paperId",
        "citationCount",
        "influentialCitationCount",
        "isOpenAccess",
    }

    numeric_cols = [
        column
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and column not in metadata_exclusions
    ]

    requested = [item.strip() for item in config.analysis.features if item.strip()]
    if not requested or requested == ["all"]:
        return [column for column in numeric_cols if _is_canonical_analysis_feature(column)]

    return [
        column
        for column in requested
        if column in frame.columns and _is_canonical_analysis_feature(column)
    ]


def _is_canonical_analysis_feature(column_name: str) -> bool:
    excluded = {
        "word_count",
        "sentence_count",
        "paper_count",
        "marker_density",
        # "coordination_density",
        # "clause_depth_per_sentence",
        # "dependency_entropy_normalized",
        # "dependency_length_norm",
        # "sentence_depth_cv",
        # "coordination_count_per_1k_words",
        # "list_of_three_per_1k_words",
    }

    if column_name in excluded:
        return False

    if column_name in {
        "clause_depth",
        "clause_depth_std",
        "dependency_entropy",
        "dependency_length",
        "dependency_length_std",
        "coordination_count",
        "coordination_per_sentence_std",
        "sentence_depth_std",
        "list_of_three",
        "avg_words_per_sentence",
        "avg_syllables_per_word",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "dale_chall",
        "hedge_ratio",
        "certainty_ratio",
    }:
        return True

    if column_name.endswith("_total_per_1k_words"):
        return True

    if column_name.endswith("_per_1k_words") and not column_name.endswith("_count_per_1k_words"):
        return True
    
    if column_name.startswith("verb_") or column_name.startswith("adjective_") or column_name.startswith("word_") or column_name.startswith("phrase_"):
        return True

    return False


def _is_metadata_like(column_name: str) -> bool:
    lowered = column_name.lower()
    blocked_tokens = {
        "citation",
        "openaccess",
        "paperid",
        "publication",
        "externalid",
        "fields_of_study",
        "fieldsofstudy",
        "authors",
        "venue",
        "journal",
        "url",
        "tldr",
    }

    normalized = lowered.replace("_", "")
    return any(token in lowered or token in normalized for token in blocked_tokens)