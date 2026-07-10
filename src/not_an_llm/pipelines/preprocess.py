from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd

from not_an_llm.config import AppConfig
from not_an_llm.preprocessing.text import TextPreprocessor


LOGGER = logging.getLogger(__name__)
DEFAULT_CHUNK_SIZE = 5000


def run_preprocessing(config: AppConfig) -> Path:
    input_path = config.collection.output_jsonl
    output_path = config.analysis.preprocessed_jsonl

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {input_path}. Run collection first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Starting preprocessing: input=%s output=%s text_source=%s chunk_size=%s",
        input_path,
        output_path,
        config.analysis.text_source,
        DEFAULT_CHUNK_SIZE,
    )
    LOGGER.info("Loading text preprocessor and spaCy model...")
    preprocessor = TextPreprocessor(
        keep_case=False,
        text_source=config.analysis.text_source,
    )
    LOGGER.info("Text preprocessor ready")
    chunk_iter = pd.read_json(input_path, lines=True, chunksize=DEFAULT_CHUNK_SIZE)

    wrote_any = False
    total_rows = 0
    for chunk_index, raw_chunk in enumerate(chunk_iter):
        chunk_number = chunk_index + 1
        LOGGER.info(
            "Preprocessing chunk %s: rows=%s total_rows_before=%s",
            chunk_number,
            len(raw_chunk),
            total_rows,
        )
        preprocessed_chunk = preprocessor.preprocess_dataframe(raw_chunk)
        if "doc" in preprocessed_chunk.columns:
            preprocessed_chunk = preprocessed_chunk.drop(columns=["doc"])
        if config.analysis.text_source == "title" and chunk_number == 1:
            LOGGER.info(
                "Title-only preprocessing output columns: %s",
                ", ".join(preprocessed_chunk.columns),
            )
        preprocessed_chunk.to_json(
            output_path,
            orient="records",
            lines=True,
            force_ascii=False,
            mode="w" if chunk_index == 0 else "a",
        )
        wrote_any = True
        total_rows += len(preprocessed_chunk)
        LOGGER.info(
            "Finished preprocessing chunk %s: rows=%s total_rows=%s output=%s",
            chunk_number,
            len(preprocessed_chunk),
            total_rows,
            output_path,
        )

    if not wrote_any:
        output_path.write_text("", encoding="utf-8")
        LOGGER.warning("No rows found in %s; wrote empty output to %s", input_path, output_path)

    LOGGER.info("Preprocessed %s rows into %s", total_rows, output_path)

    return output_path
