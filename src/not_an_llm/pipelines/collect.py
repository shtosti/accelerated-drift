from __future__ import annotations

from pathlib import Path
from typing import Iterator
import json
import logging
from tqdm import tqdm

from not_an_llm.clients.arxiv import ArxivClient
from not_an_llm.clients.semantic_scholar import SemanticScholarClient
from not_an_llm.config import AppConfig


logger = logging.getLogger(__name__)



def run_collection(config: AppConfig) -> Path:
    split_year = config.experiment.llm_introduction_year - 1
    total_limit = config.collection.max_results
    target_before = total_limit // 2
    target_after = total_limit - target_before

    logger.info(
        "Starting collection: queries=%s total_max_results=%s page_size=%s years=%s-%s split=%s/%s",
        len(config.collection.queries),
        total_limit,
        config.collection.page_size,
        config.collection.year_min,
        config.collection.year_max,
        target_before,
        target_after,
    )

    output_path = config.collection.output_jsonl
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys, existing_count, existing_before_count, existing_after_count = _load_existing_state(
        output_path,
        split_year=split_year,
    )
    if existing_count > 0:
        logger.info(
            "Resuming from existing output: path=%s existing_unique_papers=%s before=%s after=%s",
            output_path,
            existing_count,
            existing_before_count,
            existing_after_count,
        )

    needed_before = max(0, target_before - existing_before_count)
    needed_after = max(0, target_after - existing_after_count)
    total_needed = needed_before + needed_after
    available_slots = max(0, total_limit - existing_count)

    if total_needed > available_slots:
        raise ValueError(
            "Existing output already uses too many slots in one side of the year split. "
            f"Need before={needed_before}, after={needed_after}, but only {available_slots} slots are free. "
            "Start from a fresh output file (or remove overrepresented records) to enforce a 50/50 split."
        )

    if total_needed == 0:
        logger.info(
            "Output already satisfies balanced target: total=%s before=%s after=%s",
            total_limit,
            target_before,
            target_after,
        )
        return output_path

    client = _build_collection_client(config)

    new_count = 0
    with tqdm(total=total_limit, initial=existing_count, desc="Collecting papers", unit="paper") as progress:
        with output_path.open("a", encoding="utf-8") as handle:
            if needed_before > 0:
                progress.set_postfix_str("before split")
                for paper in _collect_papers_for_queries(
                    client,
                    config,
                    seen_keys=seen_keys,
                    target_count=needed_before,
                    year_min=config.collection.year_min,
                    year_max=split_year,
                    label="before",
                ):
                    handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
                    new_count += 1
                    progress.update(1)
                    handle.flush()

            if needed_after > 0:
                progress.set_postfix_str("after split")
                for paper in _collect_papers_for_queries(
                    client,
                    config,
                    seen_keys=seen_keys,
                    target_count=needed_after,
                    year_min=split_year + 1,
                    year_max=config.collection.year_max,
                    label="after",
                ):
                    handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
                    new_count += 1
                    progress.update(1)
                    handle.flush()

            handle.flush()

    logger.info(
        "Collection complete: new_papers=%s total_unique_papers=%s output=%s",
        new_count,
        existing_count + new_count,
        output_path,
    )

    return output_path


def _collect_papers_for_queries(
    client: SemanticScholarClient | ArxivClient,
    config: AppConfig,
    *,
    seen_keys: set[str],
    target_count: int,
    year_min: int,
    year_max: int,
    label: str,
) -> Iterator[dict[str, object]]:
    queries = config.collection.queries
    query_count = len(queries)

    if query_count == 0 or target_count <= 0:
        return

    base_limit = target_count // query_count
    extra = target_count % query_count

    collected_count = 0

    for index, query in enumerate(queries):
        query_limit = base_limit + (1 if index < extra else 0)
        if query_limit <= 0:
            continue

        logger.info(
            "[%s] Query %s/%s: target=%s years=%s-%s query=%s",
            label,
            index + 1,
            query_count,
            query_limit,
            year_min,
            year_max,
            query,
        )

        batch = client.search_papers(
            query=query,
            fields=config.collection.fields,
            year_min=year_min,
            year_max=year_max,
            limit=query_limit,
            page_size=config.collection.page_size,
        )

        added_count = 0
        duplicate_count = 0

        for paper in batch:
            key = _paper_key(paper)
            if key in seen_keys:
                duplicate_count += 1
                continue
            seen_keys.add(key)
            collected_count += 1
            added_count += 1
            yield paper

            if collected_count >= target_count:
                logger.info(
                    "[%s] Reached bucket target (%s). stopping early after query %s/%s.",
                    label,
                    target_count,
                    index + 1,
                    query_count,
                )
                return

        logger.info(
            "[%s] Query %s/%s done: fetched=%s added=%s duplicates=%s running_bucket_total=%s/%s",
            label,
            index + 1,
            query_count,
            len(batch),
            added_count,
            duplicate_count,
            collected_count,
            target_count,
        )


def _load_existing_state(output_path: Path, *, split_year: int) -> tuple[set[str], int, int, int]:
    if not output_path.exists():
        return set(), 0, 0, 0

    seen_keys: set[str] = set()
    before_count = 0
    after_count = 0

    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line %s in %s", line_number, output_path)
                continue

            if not isinstance(payload, dict):
                logger.warning("Skipping non-object JSONL line %s in %s", line_number, output_path)
                continue

            paper = payload
            key = _paper_key(paper)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            year = _paper_year(paper)
            if year is None:
                continue
            if year <= split_year:
                before_count += 1
            else:
                after_count += 1

    return seen_keys, len(seen_keys), before_count, after_count


def _paper_key(paper: dict[str, object]) -> str:
    paper_id = paper.get("paperId")
    if isinstance(paper_id, str) and paper_id.strip():
        return f"paperId:{paper_id.strip()}"

    title = paper.get("title")
    year = paper.get("year")
    return f"fallback:{title}|{year}"


def _paper_year(paper: dict[str, object]) -> int | None:
    raw_year = paper.get("year")
    if isinstance(raw_year, int):
        return raw_year
    if isinstance(raw_year, str):
        try:
            return int(raw_year)
        except ValueError:
            return None
    return None


def _build_collection_client(config: AppConfig) -> SemanticScholarClient | ArxivClient:
    common_kwargs = {
        "min_request_interval_seconds": config.collection.min_request_interval_seconds,
        "max_retries": config.collection.max_retries,
        "initial_backoff_seconds": config.collection.initial_backoff_seconds,
        "max_backoff_seconds": config.collection.max_backoff_seconds,
        "backoff_jitter_seconds": config.collection.backoff_jitter_seconds,
    }

    source = config.collection.source
    if source == "arxiv":
        return ArxivClient(**common_kwargs)
    return SemanticScholarClient(**common_kwargs)
