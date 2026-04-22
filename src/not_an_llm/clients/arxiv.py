from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime
from typing import Any
import logging
import random
import time
import xml.etree.ElementTree as ET

import requests


logger = logging.getLogger(__name__)

ATOM_NS = "http://www.w3.org/2005/Atom"


@dataclass(slots=True)
class ArxivClient:
    base_url: str = "https://export.arxiv.org/api/query"
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

        papers: list[dict[str, Any]] = []
        start = 0
        date_query = self._build_date_query(published_year=published_year, published_month=published_month)

        while True:
            if limit is not None and len(papers) >= limit:
                break

            batch_size = page_size if limit is None else min(page_size, limit - len(papers))
            if batch_size <= 0:
                break

            payload = self._get_feed(query=query, date_query=date_query, start=start, max_results=batch_size)
            batch = self._parse_entries(
                payload=payload,
                year_min=year_min,
                year_max=year_max,
                published_year=published_year,
                published_month=published_month,
            )
            if not batch:
                break

            papers.extend(batch)
            start += batch_size

            if len(batch) < batch_size:
                break

        return papers

    def _get_feed(self, *, query: str, date_query: str | None, start: int, max_results: int) -> str:
        search_query = self._build_search_query(query=query, date_query=date_query)
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        backoff_seconds = self.initial_backoff_seconds
        for attempt in range(self.max_retries + 1):
            self._respect_rate_limit()
            try:
                response = requests.get(self.base_url, params=params, timeout=self.timeout_seconds)
                self._last_request_at = time.monotonic()
                response.raise_for_status()
                return response.text
            except requests.HTTPError as error:
                status_code = error.response.status_code if error.response is not None else 0
                if status_code not in {429, 500, 502, 503, 504} or attempt >= self.max_retries:
                    raise

                sleep_seconds = max(backoff_seconds, 10.0 if status_code == 429 else 0.0)
                logger.warning(
                    "Retryable HTTP %s from arXiv (attempt %s/%s). sleeping %.2fs",
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
                    "Network error from arXiv (attempt %s/%s). sleeping %.2fs",
                    attempt + 1,
                    self.max_retries + 1,
                    sleep_seconds,
                )

            sleep_seconds += random.uniform(0.0, self.backoff_jitter_seconds)
            time.sleep(sleep_seconds)
            backoff_seconds = min(backoff_seconds * 2.0, self.max_backoff_seconds)

        raise RuntimeError("arXiv request retry loop exhausted unexpectedly")

    def _respect_rate_limit(self) -> None:
        if self._last_request_at <= 0:
            return

        elapsed_seconds = time.monotonic() - self._last_request_at
        sleep_seconds = self.min_request_interval_seconds - elapsed_seconds
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    @staticmethod
    def _parse_entries(
        *,
        payload: str,
        year_min: int,
        year_max: int,
        published_year: int | None = None,
        published_month: int | None = None,
    ) -> list[dict[str, Any]]:
        namespace = {"atom": ATOM_NS}
        root = ET.fromstring(payload)
        entries = root.findall("atom:entry", namespace)

        papers: list[dict[str, Any]] = []
        for entry in entries:
            published_text = _entry_text(entry, "published", namespace)
            if published_text is None:
                continue

            published_at = datetime.fromisoformat(published_text.replace("Z", "+00:00"))
            year = published_at.year
            if year < year_min or year > year_max:
                continue
            if published_year is not None and year != published_year:
                continue
            if published_month is not None and published_at.month != published_month:
                continue

            arxiv_id = _extract_arxiv_id(_entry_text(entry, "id", namespace))
            title = (_entry_text(entry, "title", namespace) or "").strip()
            summary = (_entry_text(entry, "summary", namespace) or "").strip()
            authors = [
                {"name": (author.findtext(f"{{{ATOM_NS}}}name") or "").strip()}
                for author in entry.findall("atom:author", namespace)
            ]
            categories = [
                category.attrib.get("term", "")
                for category in entry.findall("atom:category", namespace)
                if category.attrib.get("term")
            ]

            papers.append(
                {
                    "paperId": arxiv_id,
                    "title": title,
                    "abstract": summary,
                    "year": year,
                    "authors": authors,
                    "venue": "arXiv",
                    "publicationDate": published_at.date().isoformat(),
                    "citationCount": None,
                    "influentialCitationCount": None,
                    "fieldsOfStudy": categories,
                    "publicationTypes": ["preprint"],
                    "journal": None,
                    "isOpenAccess": True,
                    "externalIds": {"ArXiv": arxiv_id},
                    "url": _entry_text(entry, "id", namespace),
                    "tldr": None,
                    "source": "arxiv",
                }
            )

        return papers

    @staticmethod
    def _build_date_query(*, published_year: int | None, published_month: int | None) -> str | None:
        if published_year is None or published_month is None:
            return None

        last_day = calendar.monthrange(published_year, published_month)[1]
        start_stamp = f"{published_year:04d}{published_month:02d}010000"
        end_stamp = f"{published_year:04d}{published_month:02d}{last_day:02d}2359"
        return f"submittedDate:[{start_stamp} TO {end_stamp}]"

    @staticmethod
    def _build_search_query(*, query: str, date_query: str | None) -> str:
        if date_query is None:
            return query
        return f"({query}) AND {date_query}"


def _entry_text(entry: ET.Element, name: str, namespace: dict[str, str]) -> str | None:
    return entry.findtext(f"atom:{name}", default=None, namespaces=namespace)


def _extract_arxiv_id(raw_id: str | None) -> str:
    if raw_id is None:
        return ""
    return raw_id.rstrip("/").split("/")[-1]