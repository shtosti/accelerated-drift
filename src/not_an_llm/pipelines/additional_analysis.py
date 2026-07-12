from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import spacy

from not_an_llm.config import AppConfig
from not_an_llm.pipelines.analyze import _resolve_analysis_paths


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AdditionalAnalysisArtifacts:
    per_document_csv: Path
    yearly_csv: Path
    plot_path: Path


def run_additional_analysis(
    config: AppConfig,
    *,
    input_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    chunk_size: int = 2000,
) -> AdditionalAnalysisArtifacts:
    """Run small, targeted follow-up analyses without rerunning the full pipeline."""

    source_path = _resolve_additional_input(config, input_path)
    analysis_dir, _, _, _, _ = _resolve_analysis_paths(config)
    out_dir = Path(output_dir) if output_dir is not None else analysis_dir / "additional_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_doc_path = out_dir / "determiner_decomposition_documents.csv"
    yearly_path = out_dir / "determiner_decomposition_yearly.csv"
    plot_path = out_dir / "determiner_decomposition_yearly.png"

    LOGGER.info("Running additional dependency analysis from %s", source_path)
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    nlp.max_length = 2_000_000

    wrote_any = False
    yearly_parts = []
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
        result = pd.DataFrame.from_records(records)
        result.to_csv(
            per_doc_path,
            index=False,
            mode="w" if chunk_index == 0 else "a",
            header=chunk_index == 0,
        )
        yearly_parts.append(result)
        wrote_any = True

    if not wrote_any:
        empty = pd.DataFrame()
        empty.to_csv(per_doc_path, index=False)
        empty.to_csv(yearly_path, index=False)
        return AdditionalAnalysisArtifacts(per_doc_path, yearly_path, plot_path)

    per_doc = pd.concat(yearly_parts, ignore_index=True)
    yearly = _aggregate_determiner_decomposition(per_doc)
    yearly.to_csv(yearly_path, index=False)
    _save_determiner_decomposition_plot(yearly, plot_path)

    return AdditionalAnalysisArtifacts(per_doc_path, yearly_path, plot_path)


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
        ax.tick_params(axis="x", rotation=45)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
