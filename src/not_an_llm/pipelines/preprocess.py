from __future__ import annotations

from pathlib import Path

import pandas as pd

from not_an_llm.config import AppConfig
from not_an_llm.preprocessing.text import TextPreprocessor


def run_preprocessing(config: AppConfig) -> Path:
    input_path = config.collection.output_jsonl
    output_path = config.analysis.preprocessed_jsonl

    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {input_path}. Run collection first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_json(input_path, lines=True)
    preprocessor = TextPreprocessor(keep_case=False)
    preprocessed = preprocessor.preprocess_dataframe(raw)

    preprocessed.to_json(output_path, orient="records", lines=True, force_ascii=False)

    return output_path
