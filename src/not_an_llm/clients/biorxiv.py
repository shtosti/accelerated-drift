from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import calendar
import logging
import random
import time

import requests


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BiorxivClient:
    base_url: str = "https://api.biorxiv.org/details/biorxiv"
    timeout_seconds: int = 30
    min_request_interval_seconds: float = 3.0
    max_retries: int = 5
    initial_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 60.0
    backoff_jitter_seconds: float = 0.25
    _last_request_at: float = 0.0

    def search_papers(
        self,
        *,
        query: str,
        fields: list[str],
        year_min: int,
        year_max: int,
        limit: int | None,
        page_size: int,
        published_year: int | None = None,
        published_month: int | None = None,
    ) -> list[dict[str, Any]]:
        del fields
        del page_size  # API serves up to 100 records per call.

        start_date, end_date = _build_date_bounds(
            year_min=year_min,
            year_max=year_max,
            published_year=published_year,
            published_month=published_month,
        )

        papers: list[dict[str, Any]] = []
        cursor = 0

        while True:
            if limit is not None and len(papers) >= limit:
                break

            payload = self._get_feed(start_date=start_date, end_date=end_date, cursor=cursor)
            collection = payload.get("collection", [])
            if not isinstance(collection, list) or not collection:
                break

            for item in collection:
                if not isinstance(item, dict):
                    continue

                paper = _normalize_record(item)
                if paper is None:
                    continue

                paper_year = paper.get("year")
                publication_date = paper.get("publicationDate")
                if not isinstance(paper_year, int) or not (year_min <= paper_year <= year_max):
                    continue

                if published_year is not None and paper_year != published_year:
                    continue

                if published_month is not None:
                    month = _extract_month(publication_date)
                    if month != published_month:
                        continue

                if not _matches_query(query, paper):
                    continue

                papers.append(paper)
                if limit is not None and len(papers) >= limit:
                    break

            if not collection:
                break

            cursor += len(collection)

        return papers

    def _get_feed(self, *, start_date: str, end_date: str, cursor: int) -> dict[str, Any]:
        url = f"{self.base_url}/{start_date}/{end_date}/{cursor}/json"

        backoff_seconds = self.initial_backoff_seconds
        for attempt in range(self.max_retries + 1):
            self._respect_rate_limit()
            try:
                response = requests.get(url, timeout=self.timeout_seconds)
                self._last_request_at = time.monotonic()
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
                raise ValueError("Unexpected bioRxiv response payload type")
            except requests.HTTPError as error:
                status_code = error.response.status_code if error.response is not None else 0
                if status_code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise

                sleep_seconds = max(backoff_seconds, 10.0 if status_code == 429 else 0.0)
                logger.warning(
                    "Retryable HTTP %s from bioRxiv (attempt %s/%s). sleeping %.2fs",
                    status_code,
                    attempt + 1,
                    self.max_retries + 1,
                    sleep_seconds,
                )
            except requests.RequestException:
                if attempt >= self.max_retries:
                    raise
                sleep_seconds = backoff_seconds
                logger.warning(
                    "Network error from bioRxiv (attempt %s/%s). sleeping %.2fs",
                    attempt + 1,
                    self.max_retries + 1,
                    sleep_seconds,
                )

            sleep_seconds += random.uniform(0.0, self.backoff_jitter_seconds)
            time.sleep(sleep_seconds)
            backoff_seconds = min(backoff_seconds * 2.0, self.max_backoff_seconds)

        raise RuntimeError("bioRxiv request retry loop exhausted unexpectedly")

    def _respect_rate_limit(self) -> None:
        if self._last_request_at <= 0:
            return

        elapsed_seconds = time.monotonic() - self._last_request_at
        sleep_seconds = self.min_request_interval_seconds - elapsed_seconds
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def _build_date_bounds(
    *,
    year_min: int,
    year_max: int,
    published_year: int | None,
    published_month: int | None,
) -> tuple[str, str]:
    if published_year is not None and published_month is not None:
        last_day = calendar.monthrange(published_year, published_month)[1]
        return (
            f"{published_year:04d}-{published_month:02d}-01",
            f"{published_year:04d}-{published_month:02d}-{last_day:02d}",
        )

    return f"{year_min:04d}-01-01", f"{year_max:04d}-12-31"


def _normalize_record(item: dict[str, Any]) -> dict[str, Any] | None:
    doi = str(item.get("doi", "")).strip()
    title = str(item.get("title", "")).strip()
    abstract = str(item.get("abstract", "")).strip()

    date_text = str(item.get("date", "")).strip()
    if not date_text:
        return None

    try:
        published_at = datetime.strptime(date_text, "%Y-%m-%d")
    except Exception:
        return None  # still OK, but now explicit

    authors_text = str(item.get("authors", "")).strip()
    authors = [
        {"name": a.strip()}
        for a in authors_text.split(";")
        if a.strip()
    ]

    category = str(item.get("category", "")).strip()
    version = str(item.get("version", "")).strip()

    url = (
        f"https://www.biorxiv.org/content/{doi}v{version}"
        if doi and version
        else None
    )

    return {
        "paperId": doi or f"biorxiv:{title}|{published_at.date()}",
        "title": title,
        "abstract": abstract,
        "year": published_at.year,
        "month": published_at.month,
        "authors": authors,
        "venue": "bioRxiv",
        "publicationDate": published_at.date().isoformat(),
        "fieldsOfStudy": [category] if category else [],
        "url": url,
        "source": "biorxiv",
    }


def _matches_query(query: str, paper: dict[str, Any]) -> bool:
    normalized = query.strip().lower()
    if normalized in {"", "*", "all"}:
        return True

    haystack_parts = [
        str(paper.get("title", "")),
        str(paper.get("abstract", "")),
        " ".join(str(item) for item in paper.get("fieldsOfStudy", []) if isinstance(item, str)),
    ]
    haystack = " ".join(haystack_parts).lower()
    return normalized in haystack


def _extract_month(publication_date: Any) -> int | None:
    if not isinstance(publication_date, str):
        return None
    parts = publication_date.split("-")
    if len(parts) < 2:
        return None
    try:
        month = int(parts[1])
    except ValueError:
        return None
    return month