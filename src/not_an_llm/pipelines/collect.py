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
    if config.collection.source == "arxiv":
        return _run_monthly_arxiv_collection(config)

    return _run_query_based_collection(config)


def _run_monthly_arxiv_collection(config: AppConfig) -> Path:
    total_limit = _expected_monthly_total(config)
    output_path = config.collection.output_jsonl
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys, existing_total, existing_month_counts = _load_existing_monthly_state(
        output_path,
        year_min=config.collection.year_min,
        year_max=config.collection.year_max,
    )

    logger.info(
        "Starting arXiv monthly collection: years=%s-%s samples_per_month=%s total_target=%s",
        config.collection.year_min,
        config.collection.year_max,
        config.collection.samples_per_month,
        total_limit,
    )

    if existing_total > 0:
        logger.info(
            "Resuming from existing output: path=%s existing_unique_papers=%s",
            output_path,
            existing_total,
        )

    month_plan = _build_month_plan(config.collection.year_min, config.collection.year_max)
    for year, month in month_plan:
        existing_month_count = existing_month_counts.get((year, month), 0)
        if existing_month_count > config.collection.samples_per_month:
            raise ValueError(
                f"Existing output already has {existing_month_count} papers for {year}-{month:02d}, "
                f"which exceeds the target of {config.collection.samples_per_month}. "
                "Use a fresh output file to enforce an exact per-month sample."
            )

    if existing_total >= total_limit and all(
        existing_month_counts.get((year, month), 0) >= config.collection.samples_per_month
        for year, month in month_plan
    ):
        logger.info("Output already satisfies the monthly sampling target. Nothing to do.")
        return output_path

    client = _build_collection_client(config)

    new_count = 0
    with tqdm(total=total_limit, initial=existing_total, desc="Collecting papers", unit="paper") as progress:
        with output_path.open("a", encoding="utf-8") as handle:
            for year, month in month_plan:
                target_count = config.collection.samples_per_month
                existing_count = existing_month_counts.get((year, month), 0)
                needed = target_count - existing_count
                if needed <= 0:
                    continue

                progress.set_postfix_str(f"{year}-{month:02d}")
                logger.info(
                    "Month %s-%02s: target=%s existing=%s needed=%s",
                    year,
                    month,
                    target_count,
                    existing_count,
                    needed,
                )

                for paper in _collect_papers_for_month(
                    client,
                    config,
                    seen_keys=seen_keys,
                    year=year,
                    month=month,
                    target_count=needed,
                ):
                    handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
                    new_count += 1
                    progress.update(1)
                    handle.flush()

            handle.flush()

    logger.info(
        "Monthly collection complete: new_papers=%s total_unique_papers=%s output=%s",
        new_count,
        existing_total + new_count,
        output_path,
    )

    return output_path


def _run_query_based_collection(config: AppConfig) -> Path:
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


def _collect_papers_for_month(
    client: SemanticScholarClient | ArxivClient,
    config: AppConfig,
    *,
    seen_keys: set[str],
    year: int,
    month: int,
    target_count: int,
) -> Iterator[dict[str, object]]:
    queries = config.collection.queries
    if not queries or target_count <= 0:
        return

    collected_count = 0
    for index, query in enumerate(queries):
        if collected_count >= target_count:
            return

        query_limit = target_count - collected_count
        logger.info(
            "[month %s-%02s] Query %s/%s: target=%s query=%s",
            year,
            month,
            index + 1,
            len(queries),
            query_limit,
            query,
        )

        batch = client.search_papers(
            query=query,
            fields=config.collection.fields,
            year_min=config.collection.year_min,
            year_max=config.collection.year_max,
            limit=query_limit,
            page_size=config.collection.page_size,
            published_year=year,
            published_month=month,
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
                    "[month %s-%02s] Reached bucket target (%s).",
                    year,
                    month,
                    target_count,
                )
                return

        logger.info(
            "[month %s-%02s] Query %s/%s done: fetched=%s added=%s duplicates=%s running_bucket_total=%s/%s",
            year,
            month,
            index + 1,
            len(queries),
            len(batch),
            added_count,
            duplicate_count,
            collected_count,
            target_count,
        )


def _build_month_plan(year_min: int, year_max: int) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    for year in range(year_min, year_max + 1):
        for month in range(1, 13):
            months.append((year, month))
    return months


def _expected_monthly_total(config: AppConfig) -> int:
    month_count = len(_build_month_plan(config.collection.year_min, config.collection.year_max))
    return month_count * config.collection.samples_per_month


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


def _load_existing_monthly_state(
    output_path: Path,
    *,
    year_min: int,
    year_max: int,
) -> tuple[set[str], int, dict[tuple[int, int], int]]:
    if not output_path.exists():
        return set(), 0, {}

    seen_keys: set[str] = set()
    month_counts: dict[tuple[int, int], int] = {}

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

            year, month = _paper_year_month(paper)
            if year is None or month is None:
                continue
            if year < year_min or year > year_max:
                continue
            month_counts[(year, month)] = month_counts.get((year, month), 0) + 1

    return seen_keys, len(seen_keys), month_counts


def _paper_key(paper: dict[str, object]) -> str:
    paper_id = paper.get("paperId")
    if isinstance(paper_id, str) and paper_id.strip():
        return f"paperId:{paper_id.strip()}"

    title = paper.get("title")
    year = paper.get("year")
    return f"fallback:{title}|{year}"


def _paper_year_month(paper: dict[str, object]) -> tuple[int | None, int | None]:
    raw_year = paper.get("year")
    if isinstance(raw_year, int):
        year = raw_year
    elif isinstance(raw_year, str):
        try:
            year = int(raw_year)
        except ValueError:
            return None, None
    else:
        return None, None

    raw_publication_date = paper.get("publicationDate")
    if isinstance(raw_publication_date, str):
        try:
            month = int(raw_publication_date.split("-")[1])
            return year, month
        except (IndexError, ValueError):
            return year, None

    return year, None


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
