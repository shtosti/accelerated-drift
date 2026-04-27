from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import random

from datasets import load_dataset

from not_an_llm.config import AppConfig
from not_an_llm.preprocessing.text import TextPreprocessor


@dataclass(slots=True)
class ExternalPreprocessArtifacts:
    output_jsonl: Path
    records_written: int


def _first_answer(value: object) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    return ""


def _preprocess_text_pairs(
    *,
    output_path: Path,
    records: list[dict[str, object]],
    human_texts: list[str],
    ai_texts: list[str],
) -> ExternalPreprocessArtifacts:
    preprocessor = TextPreprocessor(keep_case=False)

    all_raw_texts = human_texts + ai_texts
    all_clean_texts = [preprocessor.normalize_text(text) for text in all_raw_texts]
    docs = list(preprocessor.nlp.pipe(all_clean_texts, batch_size=128, n_process=16))

    human_docs = docs[: len(human_texts)]
    ai_docs = docs[len(human_texts) :]

    human_lemma = preprocessor._extract_lemmas(human_docs)
    ai_lemma = preprocessor._extract_lemmas(ai_docs)
    human_word_count = preprocessor._word_counts(human_docs)
    ai_word_count = preprocessor._word_counts(ai_docs)
    human_sentence_count = preprocessor._sentence_counts(human_docs)
    ai_sentence_count = preprocessor._sentence_counts(ai_docs)

    with output_path.open("w", encoding="utf-8") as handle:
        for idx, base in enumerate(records):
            payload = {
                **base,
                "human": {
                    "text_raw": human_texts[idx],
                    "text_clean": all_clean_texts[idx],
                    "text_lemma": human_lemma[idx],
                    "word_count": int(human_word_count[idx]),
                    "sentence_count": int(human_sentence_count[idx]),
                },
                "ai": {
                    "text_raw": ai_texts[idx],
                    "text_clean": all_clean_texts[len(human_texts) + idx],
                    "text_lemma": ai_lemma[idx],
                    "word_count": int(ai_word_count[idx]),
                    "sentence_count": int(ai_sentence_count[idx]),
                },
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return ExternalPreprocessArtifacts(output_jsonl=output_path, records_written=len(records))


def run_hc3_preprocessing(
    *,
    subset: str,
    split: str,
    output_jsonl: str | Path,
    subsets: list[str] | None = None,
) -> ExternalPreprocessArtifacts:
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset_set = {value.strip() for value in (subsets or []) if value.strip()}

    dataset = load_dataset("Hello-SimpleAI/HC3", subset, split=split)

    records: list[dict[str, object]] = []
    human_texts: list[str] = []
    ai_texts: list[str] = []

    for row in dataset:
        row_subset = str(row.get("category", "")) if "category" in row else subset

        if subset_set and row_subset not in subset_set:
            continue

        human_raw = _first_answer(row.get("human_answers"))
        ai_raw = _first_answer(row.get("chatgpt_answers"))

        base = {
            "pair_id": str(row.get("id", "")),
            "subset": row_subset,
            "split": split,
            "question": str(row.get("question", "")),
        }
        records.append(base)
        human_texts.append(human_raw)
        ai_texts.append(ai_raw)

    return _preprocess_text_pairs(
        output_path=output_path,
        records=records,
        human_texts=human_texts,
        ai_texts=ai_texts,
    )


def run_mage_preprocessing(
    *,
    split: str,
    output_jsonl: str | Path,
    max_pairs: int = 0,
    seed: int = 42,
    domains: list[str] | None = None,
    include_sources: list[str] | None = None,
    exclude_sources: list[str] | None = None,
) -> ExternalPreprocessArtifacts:
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("yaful/MAGE", split=split)

    human_rows: list[dict[str, object]] = []
    ai_rows: list[dict[str, object]] = []

    domain_set = {value.strip() for value in (domains or []) if value.strip()}
    include_set = {value.strip() for value in (include_sources or []) if value.strip()}
    exclude_set = {value.strip() for value in (exclude_sources or []) if value.strip()}

    for row in dataset:
        src = str(row.get("src", ""))
        domain = src.split("_", 1)[0] if "_" in src else src

        if domain_set and domain not in domain_set:
            continue
        if include_set and src not in include_set:
            continue
        if exclude_set and src in exclude_set:
            continue

        text = str(row.get("text", ""))
        payload = {
            "src": src,
            "domain": domain,
            "label": int(row.get("label", -1)) if row.get("label") is not None else -1,
            "text": text,
        }
        if "human" in src:
            human_rows.append(payload)
        else:
            ai_rows.append(payload)

    rng = random.Random(seed)
    rng.shuffle(human_rows)
    rng.shuffle(ai_rows)

    pair_count = min(len(human_rows), len(ai_rows))
    if max_pairs and max_pairs > 0:
        pair_count = min(pair_count, max_pairs)

    records: list[dict[str, object]] = []
    human_texts: list[str] = []
    ai_texts: list[str] = []

    for idx in range(pair_count):
        h = human_rows[idx]
        a = ai_rows[idx]
        records.append(
            {
                "pair_id": f"mage_{idx}",
                "subset": "mage",
                "split": split,
                "question": "",
                "human_src": h["src"],
                "ai_src": a["src"],
                "human_domain": h["domain"],
                "ai_domain": a["domain"],
                "human_label": h["label"],
                "ai_label": a["label"],
            }
        )
        human_texts.append(str(h["text"]))
        ai_texts.append(str(a["text"]))

    return _preprocess_text_pairs(
        output_path=output_path,
        records=records,
        human_texts=human_texts,
        ai_texts=ai_texts,
    )


def run_configured_external_preprocessing(config: AppConfig) -> ExternalPreprocessArtifacts:
    if config.external is None:
        raise ValueError("Missing [external] section in config; use config_external.toml for external pipelines.")

    external = config.external
    if external.dataset == "hc3":
        return run_hc3_preprocessing(
            subset=external.hc3_subset,
            split=external.hc3_split,
            output_jsonl=external.hc3_pair_output_jsonl,
            subsets=external.hc3_subsets if external.hc3_subsets else None,
        )

    if external.dataset == "mage":
        return run_mage_preprocessing(
            split=external.mage_split,
            output_jsonl=external.mage_pair_output_jsonl,
            max_pairs=external.mage_max_pairs,
            domains=external.mage_domains,
            include_sources=external.mage_include_sources,
            exclude_sources=external.mage_exclude_sources,
        )

    raise ValueError(f"Unsupported external dataset: {external.dataset}")
