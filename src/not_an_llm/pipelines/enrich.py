from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import logging

from tqdm import tqdm

from not_an_llm.config import AppConfig
from not_an_llm.full_text import (
    FULL_TEXT_EXTRACTOR_VERSION,
    build_full_text_provider,
)


LOGGER = logging.getLogger(__name__)


def run_full_text_enrichment(config: AppConfig) -> Path:
    input_path = config.collection.output_jsonl
    output_path = config.collection.enriched_output_jsonl
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {input_path}. Run collection first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(output_path)
    provider = build_full_text_provider(
        config.collection.source,
        cache_dir=config.collection.full_text_cache_dir,
        timeout_seconds=config.collection.full_text_timeout_seconds,
        min_request_interval_seconds=(
            config.collection.min_request_interval_seconds
        ),
        max_retries=config.collection.max_retries,
        initial_backoff_seconds=config.collection.initial_backoff_seconds,
        max_backoff_seconds=config.collection.max_backoff_seconds,
    )

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    total = available = failed = 0
    with input_path.open("r", encoding="utf-8") as source_handle:
        with temporary_path.open("w", encoding="utf-8") as output_handle:
            for line in tqdm(source_handle, desc="Enriching full text", unit="paper"):
                if not line.strip():
                    continue
                raw_record = json.loads(line)
                total += 1
                prior = existing.get(_paper_key(raw_record))
                if (
                    prior
                    and prior.get("full_text_status") == "available"
                    and prior.get("full_text_extractor_version")
                    == FULL_TEXT_EXTRACTOR_VERSION
                ):
                    enriched = {**raw_record, **_enrichment_fields(prior)}
                    available += 1
                else:
                    enriched = _enrich_record(raw_record, provider)
                    if enriched["full_text_status"] == "available":
                        available += 1
                    else:
                        failed += 1
                output_handle.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    temporary_path.replace(output_path)
    LOGGER.info(
        "Full-text enrichment complete: total=%s available=%s failed=%s output=%s",
        total, available, failed, output_path,
    )
    return output_path


def _enrich_record(record, provider):
    enriched_at = datetime.now(timezone.utc).isoformat()
    try:
        result = provider.fetch(record)
        return {
            **record,
            "full_text_body": result.body,
            "full_text_status": "available",
            "full_text_source": result.source,
            "full_text_url": result.url,
            "full_text_content_type": result.content_type,
            "full_text_page_count": result.page_count,
            "full_text_word_count": len(result.body.split()),
            "full_text_extractor_version": FULL_TEXT_EXTRACTOR_VERSION,
            "full_text_removed_page_furniture_blocks": (
                result.removed_page_furniture_blocks
            ),
            "full_text_removed_table_blocks": result.removed_table_blocks,
            "full_text_error": None,
            "full_text_enriched_at": enriched_at,
        }
    except Exception as error:
        LOGGER.warning("Could not enrich %s: %s", _paper_key(record), error)
        return {
            **record,
            "full_text_body": "",
            "full_text_status": "failed",
            "full_text_source": None,
            "full_text_url": None,
            "full_text_content_type": None,
            "full_text_page_count": None,
            "full_text_word_count": 0,
            "full_text_extractor_version": FULL_TEXT_EXTRACTOR_VERSION,
            "full_text_removed_page_furniture_blocks": 0,
            "full_text_removed_table_blocks": 0,
            "full_text_error": f"{type(error).__name__}: {error}",
            "full_text_enriched_at": enriched_at,
        }


def _load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[_paper_key(record)] = record
    return records


def _paper_key(record: dict) -> str:
    return str(record.get("paperId") or record.get("url") or "")


def _enrichment_fields(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key.startswith("full_text_")
    }
