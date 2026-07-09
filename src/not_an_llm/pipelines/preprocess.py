from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd

from not_an_llm.config import AppConfig
from not_an_llm.preprocessing.text import TextPreprocessor


LOGGER = logging.getLogger(__name__)
DEFAULT_CHUNK_SIZE = 5000


def run_preprocessing(config: AppConfig) -> Path:
    use_enriched = (
        config.analysis.text_mode == "full_text"
        or config.analysis.paired_full_text_only
    )
    input_path = (
        config.collection.enriched_output_jsonl
        if use_enriched
        else config.collection.output_jsonl
    )
    output_path = config.analysis.preprocessed_jsonl

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found at {input_path}. "
            f"Run {'enrich' if use_enriched else 'collect'} first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    preprocessor = TextPreprocessor(
        keep_case=False,
        text_mode=config.analysis.text_mode,
    )
    chunk_iter = pd.read_json(input_path, lines=True, chunksize=DEFAULT_CHUNK_SIZE)

    wrote_any = False
    total_rows = 0
    for chunk_index, raw_chunk in enumerate(chunk_iter):
        if use_enriched:
            if "full_text_status" not in raw_chunk.columns:
                raise ValueError(
                    f"Enriched dataset {input_path} has no full_text_status column"
                )
            raw_chunk = raw_chunk.loc[
                raw_chunk["full_text_status"].eq("available")
            ].copy()
            if raw_chunk.empty:
                continue
        preprocessed_chunk = preprocessor.preprocess_dataframe(raw_chunk)
        if "doc" in preprocessed_chunk.columns:
            preprocessed_chunk = preprocessed_chunk.drop(columns=["doc"])
        preprocessed_chunk.to_json(
            output_path,
            orient="records",
            lines=True,
            force_ascii=False,
            mode="w" if not wrote_any else "a",
        )
        wrote_any = True
        total_rows += len(preprocessed_chunk)

    if not wrote_any:
        output_path.write_text("", encoding="utf-8")

    LOGGER.info("Preprocessed %s rows into %s", total_rows, output_path)

    return output_path
