from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterator
import json
import logging
from tqdm import tqdm

from not_an_llm.clients.arxiv import ArxivClient
from not_an_llm.clients.medarxiv import MedarxivClient
from not_an_llm.config import AppConfig


logger = logging.getLogger(__name__)


def _get_output_path(config: AppConfig) -> Path:
    return config.collection.output_jsonl

def run_collection(config: AppConfig) -> Path:
    if config.collection.source == "arxiv_ai":
        mode = config.collection.arxiv_ai_collection_mode
        if mode == "monthly":
            return _run_monthly_arxiv_collection(config)
        return _run_full_arxiv_collection(config)

    if config.collection.source == "arxiv_qbio":
        mode = config.collection.arxiv_qbio_collection_mode
        if mode == "monthly":
            return _run_monthly_arxiv_collection(config)
        return _run_full_arxiv_collection(config)

    if config.collection.source == "arxiv_stat":
        mode = config.collection.arxiv_stat_collection_mode
        if mode == "monthly":
            return _run_monthly_arxiv_collection(config)
        return _run_full_arxiv_collection(config)

    if config.collection.source == "medarxiv":
        mode = config.collection.medarxiv_collection_mode
        if mode == "monthly":
            return _run_monthly_arxiv_collection(config)
        return _run_full_arxiv_collection(config)

    return _run_query_based_collection(config)


def _run_full_arxiv_collection(config: AppConfig) -> Path:
    output_path = _get_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys, existing_total = _load_existing_keys(output_path)

    logger.info(
        "Starting %s collection: queries=%s years=%s-%s cutoff=%s (month-split)",
        config.collection.source,
        len(config.collection.queries),
        config.collection.year_min,
        config.collection.year_max,
        config.collection.cutoff_date or "none",
    )

    if existing_total > 0:
        logger.info(
            "Resuming from existing output: path=%s existing_unique_papers=%s",
            output_path,
            existing_total,
        )

    if not config.collection.queries:
        logger.info("No queries configured. Nothing to do.")
        return output_path

    client = _build_collection_client(config)

    month_plan = _build_month_plan(
        config.collection.year_min,
        config.collection.year_max,
        cutoff_date=config.collection.cutoff_date,
    )

    new_count = 0
    with tqdm(desc=f"Collecting {config.collection.source} papers", initial=existing_total, unit="paper") as progress:
        with output_path.open("a", encoding="utf-8") as handle:
            for year, month in month_plan:
                progress.set_postfix_str(f"{year}-{month:02d}")
                logger.info("[%s] Collecting month bucket: %s-%02d", config.collection.source, year, month)

                for paper in _collect_all_papers_for_queries(
                    client,
                    config,
                    seen_keys=seen_keys,
                    year_min=year,
                    year_max=year,
                    published_year=year,
                    published_month=month,
                ):
                    handle.write(json.dumps(paper, ensure_ascii=False) + "\n")
                    new_count += 1
                    progress.update(1)
                    handle.flush()

            handle.flush()

    logger.info(
        "%s collection complete: new_papers=%s total_unique_papers=%s output=%s",
        config.collection.source,
        new_count,
        existing_total + new_count,
        output_path,
    )

    return output_path


def _run_monthly_arxiv_collection(config: AppConfig) -> Path:
    output_path = _get_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys, existing_total, month_counts = _load_existing_monthly_state(
        output_path,
        year_min=config.collection.year_min,
        year_max=config.collection.year_max,
        cutoff_date=config.collection.cutoff_date,
    )

    logger.info(
        "Starting monthly %s collection: queries=%s years=%s-%s cutoff=%s samples_per_month=%s",
        config.collection.source,
        len(config.collection.queries),
        config.collection.year_min,
        config.collection.year_max,
        config.collection.cutoff_date or "none",
        config.collection.samples_per_month,
    )

    if existing_total > 0:
        logger.info(
            "Resuming from existing output: path=%s existing_unique_papers=%s",
            output_path,
            existing_total,
        )

    if not config.collection.queries:
        logger.info("No queries configured. Nothing to do.")
        return output_path

    client = _build_collection_client(config)
    month_plan = _build_month_plan(
        config.collection.year_min,
        config.collection.year_max,
        cutoff_date=config.collection.cutoff_date,
    )

    new_count = 0
    with tqdm(desc=f"Collecting monthly {config.collection.source} papers", initial=existing_total, unit="paper") as progress:
        with output_path.open("a", encoding="utf-8") as handle:
            for year, month in month_plan:
                existing_month = month_counts.get((year, month), 0)
                needed = max(0, config.collection.samples_per_month - existing_month)
                if needed == 0:
                    continue

                progress.set_postfix_str(f"{year}-{month:02d}")
                for paper in _collect_month_bucket(
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
        "Monthly %s collection complete: new_papers=%s total_unique_papers=%s output=%s",
        config.collection.source,
        new_count,
        existing_total + new_count,
        output_path,
    )

    return output_path


def _collect_all_papers_for_queries(
    client: ArxivClient | MedarxivClient,
    config: AppConfig,
    *,
    seen_keys: set[str],
    year_min: int,
    year_max: int,
    published_year: int | None = None,
    published_month: int | None = None,
) -> Iterator[dict[str, object]]:
    queries = config.collection.queries
    if not queries:
        return

    for index, query in enumerate(queries):
        logger.info(
            "[%s] Query %s/%s: years=%s-%s month=%s query=%s",
            config.collection.source,
            index + 1,
            len(queries),
            year_min,
            year_max,
            (
                f"{published_year:04d}-{published_month:02d}"
                if published_year is not None and published_month is not None
                else "all"
            ),
            query,
        )

        search_kwargs: dict[str, object] = {
            "query": query,
            "fields": config.collection.fields,
            "year_min": year_min,
            "year_max": year_max,
            "limit": None,
            "page_size": config.collection.page_size,
        }
        if isinstance(client, (ArxivClient, MedarxivClient)):
            search_kwargs["published_year"] = published_year
            search_kwargs["published_month"] = published_month

        batch = client.search_papers(**search_kwargs)

        added_count = 0
        duplicate_count = 0

        for paper in batch:
            if not _is_within_collection_cutoff(paper, config.collection.cutoff_date):
                continue
            key = _paper_key(paper)
            if key in seen_keys:
                duplicate_count += 1
                continue

            seen_keys.add(key)
            added_count += 1

            # -----------------------------
            # INSERT METADATA
            # -----------------------------
            enriched_paper = dict(paper)

            enriched_paper["source_query"] = query

            # arXiv category extraction
            if config.collection.source in {"arxiv_ai", "arxiv_qbio", "arxiv_stat"}:
                arxiv_category = _extract_arxiv_category(query)
                enriched_paper["arxiv_category"] = arxiv_category
                enriched_paper["arxiv_domain"] = _extract_arxiv_domain(arxiv_category)

            yield enriched_paper

        logger.info(
            "[%s] Query %s/%s done: fetched=%s added=%s duplicates=%s",
            config.collection.source,
            index + 1,
            len(queries),
            len(batch),
            added_count,
            duplicate_count,
        )


def _run_query_based_collection(config: AppConfig) -> Path:
    logger.info(
        "Starting collection: queries=%s page_size=%s years=%s-%s cutoff=%s",
        len(config.collection.queries),
        config.collection.page_size,
        config.collection.year_min,
        config.collection.year_max,
        config.collection.cutoff_date or "none",
    )

    output_path = _get_output_path(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_keys, existing_count = _load_existing_keys(output_path)
    if existing_count > 0:
        logger.info(
            "Resuming from existing output: path=%s existing_unique_papers=%s",
            output_path,
            existing_count,
        )

    client = _build_collection_client(config)

    new_count = 0
    with tqdm(desc="Collecting papers", initial=existing_count, unit="paper") as progress:
        with output_path.open("a", encoding="utf-8") as handle:
            for paper in _collect_papers_for_queries(
                client,
                config,
                seen_keys=seen_keys,
                year_min=config.collection.year_min,
                year_max=config.collection.year_max,
                label="all",
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
    client:  ArxivClient | MedarxivClient,
    config: AppConfig,
    *,
    seen_keys: set[str],
    year_min: int,
    year_max: int,
    label: str,
) -> Iterator[dict[str, object]]:
    queries = config.collection.queries
    query_count = len(queries)

    if query_count == 0:
        return

    for index, query in enumerate(queries):
        logger.info(
            "[%s] Query %s/%s: years=%s-%s query=%s",
            label,
            index + 1,
            query_count,
            year_min,
            year_max,
            query,
        )

        batch = client.search_papers(
            query=query,
            fields=config.collection.fields,
            year_min=year_min,
            year_max=year_max,
            limit=None,
            page_size=config.collection.page_size,
        )

        added_count = 0
        duplicate_count = 0

        for paper in batch:
            if not _is_within_collection_cutoff(paper, config.collection.cutoff_date):
                continue
            key = _paper_key(paper)
            if key in seen_keys:
                duplicate_count += 1
                continue
            seen_keys.add(key)
            added_count += 1
            yield paper

        logger.info(
            "[%s] Query %s/%s done: fetched=%s added=%s duplicates=%s",
            label,
            index + 1,
            query_count,
            len(batch),
            added_count,
            duplicate_count,
        )


def _collect_month_bucket(
    client: ArxivClient | MedarxivClient,
    config: AppConfig,
    *,
    seen_keys: set[str],
    year: int,
    month: int,
    target_count: int,
) -> Iterator[dict[str, object]]:
    if target_count <= 0:
        return

    collected = 0
    for index, query in enumerate(config.collection.queries):
        if collected >= target_count:
            return

        query_limit = target_count - collected
        logger.info(
            "[month %s-%02d] Query %s/%s target=%s query=%s",
            year,
            month,
            index + 1,
            len(config.collection.queries),
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

        for paper in batch:
            if not _is_within_collection_cutoff(paper, config.collection.cutoff_date):
                continue
            key = _paper_key(paper)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            collected += 1
            yield paper
            if collected >= target_count:
                return


def _load_existing_keys(output_path: Path) -> tuple[set[str], int]:
    if not output_path.exists():
        return set(), 0

    seen_keys: set[str] = set()

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

    return seen_keys, len(seen_keys)


def _load_existing_monthly_state(
    output_path: Path,
    *,
    year_min: int,
    year_max: int,
    cutoff_date: date | None,
) -> tuple[set[str], int, dict[tuple[int, int], int]]:
    seen_keys, total = _load_existing_keys(output_path)
    month_counts: dict[tuple[int, int], int] = {}

    if not output_path.exists():
        return seen_keys, total, month_counts

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

            year, month = _paper_year_month(payload)
            if year is None or month is None:
                continue
            if year < year_min or year > year_max:
                continue
            if not _is_within_collection_cutoff(payload, cutoff_date):
                continue

            key = _paper_key(payload)
            if not key:
                continue
            month_counts[(year, month)] = month_counts.get((year, month), 0) + 1

    return seen_keys, total, month_counts


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


def _paper_year_month(paper: dict[str, object]) -> tuple[int | None, int | None]:
    year = _paper_year(paper)
    if year is None:
        return None, None

    raw_publication_date = paper.get("publicationDate")
    if not isinstance(raw_publication_date, str):
        return year, None

    try:
        month = int(raw_publication_date.split("-")[1])
    except (IndexError, ValueError):
        return year, None
    return year, month


def _is_within_collection_cutoff(paper: dict[str, object], cutoff_date: date | None) -> bool:
    if cutoff_date is None:
        return True

    raw_publication_date = paper.get("publicationDate")
    if isinstance(raw_publication_date, str):
        try:
            return date.fromisoformat(raw_publication_date[:10]) <= cutoff_date
        except ValueError:
            pass

    year, month = _paper_year_month(paper)
    if year is None:
        return True
    if month is None:
        return year <= cutoff_date.year
    return (year, month) <= (cutoff_date.year, cutoff_date.month)


def _paper_key(paper: dict[str, object]) -> str:
    paper_id = paper.get("paperId")
    if isinstance(paper_id, str) and paper_id.strip():
        return f"paperId:{paper_id.strip()}"

    title = paper.get("title")
    year = paper.get("year")
    return f"fallback:{title}|{year}"


def _build_month_plan(year_min: int, year_max: int, *, cutoff_date: date | None = None) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    for year in range(year_min, year_max + 1):
        for month in range(1, 13):
            if cutoff_date is not None and (year, month) > (cutoff_date.year, cutoff_date.month):
                continue
            months.append((year, month))
    return months


def _build_year_plan(year_min: int, year_max: int) -> list[int]:
    return list(range(year_min, year_max + 1))


def _build_collection_client(config: AppConfig) -> ArxivClient | MedarxivClient: 
    common_kwargs = {
        "min_request_interval_seconds": config.collection.min_request_interval_seconds,
        "max_retries": config.collection.max_retries,
        "initial_backoff_seconds": config.collection.initial_backoff_seconds,
        "max_backoff_seconds": config.collection.max_backoff_seconds,
        "backoff_jitter_seconds": config.collection.backoff_jitter_seconds,
    }

    source = config.collection.source
    if source in {"arxiv_ai", "arxiv_qbio", "arxiv_stat"}:
        return ArxivClient(**common_kwargs)
    if source == "medarxiv":
        return MedarxivClient(**common_kwargs)
    raise ValueError(f"Unsupported collection source: {source}")


def _extract_arxiv_category(query: str) -> str:
    for token in query.replace("(", " ").replace(")", " ").split():
        token = token.strip()
        if not token.startswith("cat:"):
            continue
        category = token.removeprefix("cat:").strip()
        return category or "other"
    return "other"


def _extract_arxiv_domain(category: str) -> str:
    if "." in category:
        return category.split(".", 1)[0]
    return category
