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

    preprocessor = TextPreprocessor(keep_case=False)
    chunk_iter = pd.read_json(input_path, lines=True, chunksize=DEFAULT_CHUNK_SIZE)

    wrote_any = False
    total_rows = 0
    for chunk_index, raw_chunk in enumerate(chunk_iter):
        preprocessed_chunk = preprocessor.preprocess_dataframe(raw_chunk)
        preprocessed_chunk.to_json(
            output_path,
            orient="records",
            lines=True,
            force_ascii=False,
            mode="w" if chunk_index == 0 else "a",
        )
        wrote_any = True
        total_rows += len(preprocessed_chunk)

    if not wrote_any:
        output_path.write_text("", encoding="utf-8")

    LOGGER.info("Preprocessed %s rows into %s", total_rows, output_path)

    return output_path
