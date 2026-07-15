from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import logging
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import pandas as pd
import spacy

from not_an_llm.config import AppConfig
from not_an_llm.analysis.feature_selection import resolve_feature_columns
from not_an_llm.analysis.interrupted_time_series import (
    compute_first_post_year_counterfactual_excess,
    compute_first_two_year_counterfactual_excess,
    save_first_post_year_excess_plot,
    save_first_post_year_grouped_excess_plots,
    save_first_two_year_excess_plot,
    save_first_two_year_grouped_excess_plots,
)
from not_an_llm.analysis.label_map import LABEL_MAP
from not_an_llm.analysis.trends import DEPENDENCY_ROLE_COLORS
from not_an_llm.pipelines.analyze import _resolve_analysis_paths


LOGGER = logging.getLogger(__name__)


def run_comparison_analyses() -> None:
    """Regenerate cross-corpus and title-vs-abstract comparisons."""

    root = Path(__file__).resolve().parents[3]
    scripts = (
        root / "scripts" / "compare_corpus_trends.py",
        root / "scripts" / "compare_title_abstract_trends.py",
    )
    for script in scripts:
        LOGGER.info("Running comparison analysis %s", script.name)
        subprocess.run([sys.executable, str(script)], cwd=root, check=True)

@dataclass(slots=True)
class AdditionalAnalysisArtifacts:
    per_document_csv: Path | None
    yearly_csv: Path | None
    plot_path: Path | None
    dependency_bigram_yearly_csv: Path | None
    dependency_bigram_change_csv: Path | None
    dependency_bigram_plot_path: Path | None
    first_post_year_counterfactual_csv: Path | None
    first_two_year_counterfactual_csv: Path | None
    counterfactual_plot_paths: list[Path]


def run_additional_analysis(
    config: AppConfig,
    *,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    chunk_size: int = 2000,
    counterfactual_only: bool = False,
) -> AdditionalAnalysisArtifacts:
    """Run small, targeted follow-up analyses without rerunning the full pipeline."""

    analysis_dir, _, _, monthly_csv, plot_dir = _resolve_analysis_paths(config)
    if monthly_csv.exists():
        counterfactual_paths = _run_counterfactual_analysis(config, monthly_csv, analysis_dir, plot_dir)
    elif counterfactual_only:
        raise FileNotFoundError(
            f"Monthly trends not found at {monthly_csv}. Run analyze once before "
            "running additional-analysis --counterfactual-only."
        )
    else:
        LOGGER.warning("Skipping counterfactual refresh because %s does not exist", monthly_csv)
        counterfactual_paths = (None, None, [])
    if counterfactual_only:
        return AdditionalAnalysisArtifacts(
            None,
            None,
            None,
            None,
            None,
            None,
            *counterfactual_paths,
        )

    source_path = _resolve_additional_input(config, input_path)
    out_dir = Path(output_dir) if output_dir is not None else analysis_dir / "additional_analysis"
    visual_out_dir = _resolve_additional_visual_output_dir(config, output_dir, plot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    visual_out_dir.mkdir(parents=True, exist_ok=True)

    per_doc_path = out_dir / "determiner_decomposition_documents.csv"
    yearly_path = out_dir / "determiner_decomposition_yearly.csv"
    plot_path = visual_out_dir / "determiner_decomposition_yearly.png"
    bigram_yearly_path = out_dir / "dependency_bigram_yearly.csv"
    bigram_change_path = out_dir / "dependency_bigram_change.csv"
    bigram_plot_path = visual_out_dir / "dependency_bigram_trends.png"

    LOGGER.info("Running additional dependency analysis from %s", source_path)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    nlp.max_length = 2_000_000

    wrote_any = False
    yearly_parts = []
    bigram_parts = []
    chunks = pd.read_json(source_path, lines=True, chunksize=chunk_size)

    for chunk_index, chunk in enumerate(chunks):
        required = {"year", "text_clean"}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(
                f"Input {source_path} is missing required columns: {', '.join(sorted(missing))}. "
                "Use a feature_dataset.jsonl or preprocessed JSONL that includes text_clean and year."
            )

        docs = list(nlp.pipe(chunk["text_clean"].fillna("").astype(str).tolist(), batch_size=128, n_process=1))
        records = [
            _dependency_decomposition_record(row, doc)
            for row, doc in zip(chunk.to_dict("records"), docs, strict=False)
        ]
        bigram_records = [
            record
            for row_index, (row, doc) in enumerate(zip(chunk.to_dict("records"), docs, strict=False))
            for record in _dependency_bigram_records(row, doc, document_key=row.get("paperId") or f"{chunk_index}:{row_index}")
        ]
        result = pd.DataFrame.from_records(records)
        result.to_csv(
            per_doc_path,
            index=False,
            mode="w" if chunk_index == 0 else "a",
            header=chunk_index == 0,
        )
        yearly_parts.append(result)
        if bigram_records:
            bigram_parts.append(pd.DataFrame.from_records(bigram_records))
        wrote_any = True

    if not wrote_any:
        empty = pd.DataFrame()
        empty.to_csv(per_doc_path, index=False)
        empty.to_csv(yearly_path, index=False)
        empty.to_csv(bigram_yearly_path, index=False)
        empty.to_csv(bigram_change_path, index=False)
        return AdditionalAnalysisArtifacts(
            per_doc_path,
            yearly_path,
            plot_path,
            bigram_yearly_path,
            bigram_change_path,
            bigram_plot_path,
            *counterfactual_paths,
        )

    per_doc = pd.concat(yearly_parts, ignore_index=True)
    yearly = _aggregate_determiner_decomposition(per_doc)
    yearly.to_csv(yearly_path, index=False)
    _save_determiner_decomposition_plot(yearly, plot_path)
    if bigram_parts:
        bigram_yearly = _aggregate_dependency_bigrams(pd.concat(bigram_parts, ignore_index=True))
    else:
        bigram_yearly = _empty_dependency_bigram_yearly()
    bigram_yearly.to_csv(bigram_yearly_path, index=False)
    bigram_change = _dependency_bigram_change_table(bigram_yearly)
    bigram_change.to_csv(bigram_change_path, index=False)
    _save_dependency_bigram_trend_plot(bigram_yearly, bigram_change, bigram_plot_path)

    return AdditionalAnalysisArtifacts(
        per_doc_path,
        yearly_path,
        plot_path,
        bigram_yearly_path,
        bigram_change_path,
        bigram_plot_path,
        *counterfactual_paths,
    )


def _run_counterfactual_analysis(
    config: AppConfig,
    monthly_csv: Path,
    analysis_dir: Path,
    plot_dir: Path,
) -> tuple[Path, Path, list[Path]]:
    """Refresh strict pre-intervention counterfactual outputs from monthly trends."""

    monthly = pd.read_csv(monthly_csv)
    features = resolve_feature_columns(config, monthly)
    if not features:
        raise ValueError(f"No configured analysis features were found in {monthly_csv}")

    LOGGER.info("Computing strict counterfactual analyses from %s", monthly_csv)
    first_post_year = compute_first_post_year_counterfactual_excess(monthly, features)
    first_two_year = compute_first_two_year_counterfactual_excess(monthly, features)

    first_post_year_csv = analysis_dir / "first_post_year_counterfactual_excess.csv"
    first_two_year_csv = analysis_dir / "first_two_year_counterfactual_excess.csv"
    first_post_year.to_csv(first_post_year_csv, index=False)
    first_two_year.to_csv(first_two_year_csv, index=False)

    plot_paths: list[Path] = []
    first_plot = save_first_post_year_excess_plot(
        first_post_year,
        plot_dir / "first_post_year_counterfactual_excess" / "overall.png",
        label_map=LABEL_MAP,
    )
    if first_plot is not None:
        plot_paths.append(first_plot)
    plot_paths.extend(
        save_first_post_year_grouped_excess_plots(
            first_post_year,
            plot_dir / "first_post_year_counterfactual_excess" / "groups",
            label_map=LABEL_MAP,
        )
    )

    second_plot = save_first_two_year_excess_plot(
        first_two_year,
        plot_dir / "first_two_year_counterfactual_excess" / "overall.png",
        label_map=LABEL_MAP,
    )
    if second_plot is not None:
        plot_paths.append(second_plot)
    plot_paths.extend(
        save_first_two_year_grouped_excess_plots(
            first_two_year,
            plot_dir / "first_two_year_counterfactual_excess" / "groups",
            label_map=LABEL_MAP,
        )
    )
    return first_post_year_csv, first_two_year_csv, plot_paths


def _resolve_additional_input(config: AppConfig, input_path: str | Path | None) -> Path:
    if input_path is not None:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Additional-analysis input not found at {path}")
        return path

    _, feature_dataset, _, _, _ = _resolve_analysis_paths(config)
    candidates = [
        feature_dataset,
        config.analysis.feature_dataset_jsonl,
        config.analysis.preprocessed_jsonl,
    ]
    for path in candidates:
        if path and Path(path).exists():
            return Path(path)

    raise FileNotFoundError(
        "Could not find a feature dataset or preprocessed JSONL. "
        "Pass --input to additional-analysis."
    )


def _resolve_additional_visual_output_dir(
    config: AppConfig,
    output_dir: str | Path | None,
    plot_dir: Path,
) -> Path:
    if output_dir is None:
        return plot_dir / "additional_analysis"

    output_path = Path(output_dir)
    dataset_name = output_path.parent.name if output_path.name == "additional_analysis" else output_path.name
    return Path(config.data_dir) / "visuals" / dataset_name / "additional_analysis"


def _dependency_decomposition_record(row: dict, doc) -> dict[str, float | int | str | None]:
    det_pobj_count = 0
    det_non_pobj_count = 0
    prep_count = 0
    pobj_count = 0
    amod_count = 0
    compound_count = 0
    nonpunct_count = 0

    for token in doc:
        if token.dep_ == "punct":
            continue
        nonpunct_count += 1
        if token.dep_ == "det":
            if token.head.dep_ == "pobj":
                det_pobj_count += 1
            else:
                det_non_pobj_count += 1
        elif token.dep_ == "prep":
            prep_count += 1
        elif token.dep_ == "pobj":
            pobj_count += 1
        elif token.dep_ == "amod":
            amod_count += 1
        elif token.dep_ == "compound":
            compound_count += 1

    word_count = float(row.get("word_count") or nonpunct_count or 0.0)
    denominator = word_count if word_count > 0 else 1.0
    dep_denominator = float(nonpunct_count or 1)
    det_total = det_pobj_count + det_non_pobj_count
    prep_phrase_total = prep_count + pobj_count
    prenominal_total = amod_count + compound_count

    return {
        "paperId": row.get("paperId"),
        "year": row.get("year"),
        "publicationDate": row.get("publicationDate"),
        "word_count": word_count,
        "nonpunct_dependency_count": nonpunct_count,
        "det_pobj_count": det_pobj_count,
        "det_non_pobj_count": det_non_pobj_count,
        "det_total_count": det_total,
        "prep_count": prep_count,
        "pobj_count": pobj_count,
        "amod_count": amod_count,
        "compound_count": compound_count,
        "prepositional_modifier_count": prep_phrase_total,
        "prenominal_modifier_count": prenominal_total,
        "det_pobj_per_1k_words": det_pobj_count / denominator * 1000.0,
        "det_non_pobj_per_1k_words": det_non_pobj_count / denominator * 1000.0,
        "det_total_per_1k_words": det_total / denominator * 1000.0,
        "prepositional_modifier_per_1k_words": prep_phrase_total / denominator * 1000.0,
        "prenominal_modifier_per_1k_words": prenominal_total / denominator * 1000.0,
        "det_pobj_share_of_det": det_pobj_count / det_total if det_total else 0.0,
        "prepositional_modifier_role_prop": prep_phrase_total / dep_denominator,
        "prenominal_modifier_role_prop": prenominal_total / dep_denominator,
        "prenominal_to_prepositional_ratio": prenominal_total / prep_phrase_total if prep_phrase_total else float("nan"),
    }


def _dependency_bigram_records(
    row: dict,
    doc,
    *,
    document_key: str,
) -> list[dict[str, float | int | str | None]]:
    counts: Counter[str] = Counter()
    total_edges = 0
    for token in doc:
        if token.dep_ == "punct" or token.head == token or token.head.dep_ == "punct":
            continue
        total_edges += 1
        counts[f"{token.head.dep_}->{token.dep_}"] += 1

    word_count = float(row.get("word_count") or sum(1 for token in doc if token.dep_ != "punct") or 0.0)
    return [
        {
            "paperId": row.get("paperId"),
            "document_key": document_key,
            "year": row.get("year"),
            "publicationDate": row.get("publicationDate"),
            "word_count": word_count,
            "dependency_edge_count": total_edges,
            "dependency_bigram": bigram,
            "count": count,
        }
        for bigram, count in counts.items()
    ]


def _aggregate_determiner_decomposition(per_doc: pd.DataFrame) -> pd.DataFrame:
    df = per_doc.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    count_columns = [
        "word_count",
        "nonpunct_dependency_count",
        "det_pobj_count",
        "det_non_pobj_count",
        "det_total_count",
        "prep_count",
        "pobj_count",
        "amod_count",
        "compound_count",
        "prepositional_modifier_count",
        "prenominal_modifier_count",
    ]
    yearly = df.groupby("year", as_index=False).agg(
        paper_count=("year", "size"),
        **{column: (column, "sum") for column in count_columns},
    )

    word_den = yearly["word_count"].where(yearly["word_count"] > 0, 1.0)
    dep_den = yearly["nonpunct_dependency_count"].where(yearly["nonpunct_dependency_count"] > 0, 1.0)
    det_den = yearly["det_total_count"].where(yearly["det_total_count"] > 0, 1.0)
    prep_den = yearly["prepositional_modifier_count"].where(
        yearly["prepositional_modifier_count"] > 0,
        float("nan"),
    )

    yearly["det_pobj_per_1k_words"] = yearly["det_pobj_count"] / word_den * 1000.0
    yearly["det_non_pobj_per_1k_words"] = yearly["det_non_pobj_count"] / word_den * 1000.0
    yearly["det_total_per_1k_words"] = yearly["det_total_count"] / word_den * 1000.0
    yearly["prepositional_modifier_per_1k_words"] = yearly["prepositional_modifier_count"] / word_den * 1000.0
    yearly["prenominal_modifier_per_1k_words"] = yearly["prenominal_modifier_count"] / word_den * 1000.0
    yearly["det_pobj_share_of_det"] = yearly["det_pobj_count"] / det_den
    yearly["prepositional_modifier_role_prop"] = yearly["prepositional_modifier_count"] / dep_den
    yearly["prenominal_modifier_role_prop"] = yearly["prenominal_modifier_count"] / dep_den
    yearly["prenominal_to_prepositional_ratio"] = yearly["prenominal_modifier_count"] / prep_den
    return yearly.sort_values("year").reset_index(drop=True)


def _aggregate_dependency_bigrams(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return _empty_dependency_bigram_yearly()

    df = records.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "dependency_bigram"]).copy()
    if df.empty:
        return _empty_dependency_bigram_yearly()
    df["year"] = df["year"].astype(int)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0.0)
    df["dependency_edge_count"] = pd.to_numeric(df["dependency_edge_count"], errors="coerce").fillna(0.0)

    document_totals = df.drop_duplicates(["year", "document_key"])
    totals = document_totals.groupby("year", as_index=False).agg(
        paper_count=("document_key", "nunique"),
        word_count=("word_count", "sum"),
        dependency_edge_count=("dependency_edge_count", "sum"),
    )
    yearly = df.groupby(["year", "dependency_bigram"], as_index=False).agg(count=("count", "sum"))
    yearly = yearly.merge(totals, on="year", how="left")
    edge_den = yearly["dependency_edge_count"].where(yearly["dependency_edge_count"] > 0, 1.0)
    word_den = yearly["word_count"].where(yearly["word_count"] > 0, 1.0)
    yearly["proportion"] = yearly["count"] / edge_den
    yearly["per_1k_words"] = yearly["count"] / word_den * 1000.0
    yearly[["head_dep", "child_dep"]] = yearly["dependency_bigram"].str.split("->", n=1, expand=True)
    return yearly[
        [
            "year",
            "dependency_bigram",
            "head_dep",
            "child_dep",
            "count",
            "proportion",
            "per_1k_words",
            "paper_count",
            "word_count",
            "dependency_edge_count",
        ]
    ].sort_values(["dependency_bigram", "year"]).reset_index(drop=True)


def _dependency_bigram_change_table(yearly: pd.DataFrame, *, pre_cut: int = 2022, post_cut: int = 2023) -> pd.DataFrame:
    if yearly.empty:
        return _empty_dependency_bigram_change()

    keys = ["dependency_bigram", "head_dep", "child_dep"]
    pre_totals = yearly.loc[yearly["year"] <= pre_cut].groupby(keys, as_index=False)["count"].sum()
    post_totals = yearly.loc[yearly["year"] >= post_cut].groupby(keys, as_index=False)["count"].sum()
    grouped = pre_totals.merge(post_totals, on=keys, how="outer", suffixes=("_pre", "_post"))
    grouped = grouped.rename(columns={"count_pre": "pre_count", "count_post": "post_count"})
    grouped["pre_count"] = grouped["pre_count"].fillna(0.0)
    grouped["post_count"] = grouped["post_count"].fillna(0.0)
    all_pre_edges = float(yearly.loc[yearly["year"] <= pre_cut].drop_duplicates("year")["dependency_edge_count"].sum())
    all_post_edges = float(yearly.loc[yearly["year"] >= post_cut].drop_duplicates("year")["dependency_edge_count"].sum())
    grouped["pre_proportion"] = grouped["pre_count"] / (all_pre_edges or 1.0)
    grouped["post_proportion"] = grouped["post_count"] / (all_post_edges or 1.0)
    grouped["proportion_change"] = grouped["post_proportion"] - grouped["pre_proportion"]
    grouped["abs_proportion_change"] = grouped["proportion_change"].abs()
    grouped["relative_change_pct"] = grouped.apply(
        lambda row: 100.0 * row["proportion_change"] / abs(row["pre_proportion"])
        if row["pre_proportion"]
        else float("nan"),
        axis=1,
    )
    return grouped.sort_values("abs_proportion_change", ascending=False).reset_index(drop=True)


def _save_dependency_bigram_trend_plot(
    yearly: pd.DataFrame,
    change: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int = 10,
) -> None:
    if yearly.empty or change.empty:
        return

    top_bigrams = change.head(top_n)["dependency_bigram"].astype(str).tolist()
    plot_df = yearly[yearly["dependency_bigram"].isin(top_bigrams)].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(3, 2.5))
    for index, bigram in enumerate(top_bigrams):
        series = plot_df[plot_df["dependency_bigram"] == bigram].sort_values("year")
        if series.empty:
            continue
        ax.plot(
            series["year"],
            series["proportion"],
            marker="o",
            markersize=2.4,
            linewidth=0.9,
            color=DEPENDENCY_ROLE_COLORS[index % len(DEPENDENCY_ROLE_COLORS)],
            label=bigram,
        )

    ax.axvline(2022.92, color="#333333", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("Year", fontsize=7)
    ax.set_ylabel("Edge share", fontsize=7)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
    ax.tick_params(axis="both", labelsize=6)
    ax.tick_params(axis="x", rotation=45, pad=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.03)
    _save_dependency_bigram_legend(ax, output_path.with_name(f"{output_path.stem}_legend{output_path.suffix}"))
    plt.close(fig)


def _save_dependency_bigram_legend(ax: plt.Axes, output_path: Path) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ncol = 2
    rows = (len(handles) + ncol - 1) // ncol
    fig, legend_ax = plt.subplots(figsize=(2.65, 0.14 * rows + 0.08))
    legend_ax.axis("off")
    legend_ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0, 0.5, 1, 0.01),
        mode="expand",
        ncol=ncol,
        fontsize=5.2,
        frameon=False,
        handlelength=1.0,
        columnspacing=0.25,
        handletextpad=0.35,
        borderaxespad=0,
        labelspacing=0.25,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def _empty_dependency_bigram_yearly() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "year",
            "dependency_bigram",
            "head_dep",
            "child_dep",
            "count",
            "proportion",
            "per_1k_words",
            "paper_count",
            "word_count",
            "dependency_edge_count",
        ]
    )


def _empty_dependency_bigram_change() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "dependency_bigram",
            "head_dep",
            "child_dep",
            "pre_count",
            "post_count",
            "pre_proportion",
            "post_proportion",
            "proportion_change",
            "abs_proportion_change",
            "relative_change_pct",
        ]
    )


def _save_determiner_decomposition_plot(yearly: pd.DataFrame, output_path: Path) -> None:
    if yearly.empty:
        return

    x = yearly["year"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    axes[0].plot(x, yearly["det_pobj_per_1k_words"], marker="o", label="det attached to pobj")
    axes[0].plot(x, yearly["det_non_pobj_per_1k_words"], marker="o", label="other det")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Determiners per 1k words")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(x, yearly["prepositional_modifier_role_prop"], marker="o", label="prep + pobj")
    axes[1].plot(x, yearly["prenominal_modifier_role_prop"], marker="o", label="amod + compound")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Role proportion")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
        ax.tick_params(axis="x", rotation=45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
