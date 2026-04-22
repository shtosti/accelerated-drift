from __future__ import annotations

from pathlib import Path
import json
import logging
from tqdm import tqdm

from not_an_llm.clients.semantic_scholar import SemanticScholarClient
from not_an_llm.config import AppConfig


logger = logging.getLogger(__name__)



def run_collection(config: AppConfig) -> Path:
    logger.info(
        "Starting collection: queries=%s total_max_results=%s page_size=%s years=%s-%s",
        len(config.collection.queries),
        config.collection.max_results,
        config.collection.page_size,
        config.collection.year_min,
        config.collection.year_max,
    )

    client = SemanticScholarClient(
        min_request_interval_seconds=config.collection.min_request_interval_seconds,
        max_retries=config.collection.max_retries,
        initial_backoff_seconds=config.collection.initial_backoff_seconds,
        max_backoff_seconds=config.collection.max_backoff_seconds,
        backoff_jitter_seconds=config.collection.backoff_jitter_seconds,
    )
    papers = _collect_papers_for_queries(client, config)
    logger.info("Collection complete: unique_papers=%s", len(papers))

    output_path = config.collection.output_jsonl
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper, ensure_ascii=False) + "\n")

    logger.info("Wrote output JSONL to %s", output_path)

    return output_path


def _collect_papers_for_queries(client: SemanticScholarClient, config: AppConfig) -> list[dict[str, object]]:
    queries = config.collection.queries
    total_limit = config.collection.max_results
    query_count = len(queries)

    if query_count == 0 or total_limit <= 0:
        return []

    base_limit = total_limit // query_count
    extra = total_limit % query_count

    papers: list[dict[str, object]] = []
    seen_keys: set[str] = set()

    with tqdm(total=total_limit, desc="Collecting papers", unit="paper") as progress:
        for index, query in enumerate(queries):
            query_limit = base_limit + (1 if index < extra else 0)
            if query_limit <= 0:
                continue

            progress.set_postfix_str(f"query {index + 1}/{query_count}")
            logger.info(
                "Query %s/%s: target=%s query=%s",
                index + 1,
                query_count,
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
            )

            before_count = len(papers)
            duplicate_count = 0

            for paper in batch:
                key = _paper_key(paper)
                if key in seen_keys:
                    duplicate_count += 1
                    continue
                seen_keys.add(key)
                papers.append(paper)

                if len(papers) >= total_limit:
                    added_count = len(papers) - before_count
                    if added_count > 0:
                        progress.update(added_count)
                    logger.info(
                        "Reached total cap (%s). stopping early after query %s/%s.",
                        total_limit,
                        index + 1,
                        query_count,
                    )
                    return papers

            added_count = len(papers) - before_count
            if added_count > 0:
                progress.update(added_count)
            logger.info(
                "Query %s/%s done: fetched=%s added=%s duplicates=%s running_total=%s/%s",
                index + 1,
                query_count,
                len(batch),
                added_count,
                duplicate_count,
                len(papers),
                total_limit,
            )

    return papers


def _paper_key(paper: dict[str, object]) -> str:
    paper_id = paper.get("paperId")
    if isinstance(paper_id, str) and paper_id.strip():
        return f"paperId:{paper_id.strip()}"

    title = paper.get("title")
    year = paper.get("year")
    return f"fallback:{title}|{year}"
