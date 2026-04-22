from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
import json
import logging
import os
import random
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SemanticScholarClient:
    base_url: str = "https://api.semanticscholar.org/graph/v1"
    timeout_seconds: int = 30
    min_request_interval_seconds: float = 1.0
    max_retries: int = 5
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    backoff_jitter_seconds: float = 0.25
    _last_request_at: float = 0.0

    def search_papers(
        self,
        *,
        query: str,
        fields: list[str],
        year_min: int,
        year_max: int,
        limit: int,
        page_size: int,
    ) -> list[dict[str, Any]]:
        papers: list[dict[str, Any]] = []
        offset = 0

        while len(papers) < limit:
            batch_size = min(page_size, limit - len(papers))
            params = {
                "query": query,
                "fields": ",".join(fields),
                "limit": batch_size,
                "offset": offset,
                "year": f"{year_min}-{year_max}",
            }
            response = self._get_json(f"{self.base_url}/paper/search", params=params)
            batch = response.get("data", [])
            if not batch:
                break

            papers.extend(batch)
            offset += len(batch)

            if len(batch) < batch_size:
                break

        return papers

    def _get_json(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        full_url = f"{url}?{urlencode(params)}"
        headers = {"User-Agent": "not-an-llm/0.1"}
        api_key = os.getenv("S2_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        backoff_seconds = self.initial_backoff_seconds

        for attempt in range(self.max_retries + 1):
            self._respect_rate_limit()
            request = Request(full_url, headers=headers)

            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    self._last_request_at = time.monotonic()
                    payload = response.read().decode("utf-8")
                    return json.loads(payload)
            except HTTPError as error:
                self._last_request_at = time.monotonic()
                if not self._is_retryable_http_error(error) or attempt >= self.max_retries:
                    raise
                retry_after_seconds = self._retry_after_seconds(error)
                minimum_sleep_seconds = self._minimum_backoff_seconds(error.code)
                sleep_seconds = max(backoff_seconds, retry_after_seconds, minimum_sleep_seconds)
                logger.warning(
                    "Retryable HTTP %s from Semantic Scholar (attempt %s/%s). sleeping %.2fs",
                    error.code,
                    attempt + 1,
                    self.max_retries + 1,
                    sleep_seconds,
                )
            except URLError:
                self._last_request_at = time.monotonic()
                if attempt >= self.max_retries:
                    raise
                sleep_seconds = backoff_seconds
                logger.warning(
                    "Network error from Semantic Scholar (attempt %s/%s). sleeping %.2fs",
                    attempt + 1,
                    self.max_retries + 1,
                    sleep_seconds,
                )

            sleep_seconds += random.uniform(0.0, self.backoff_jitter_seconds)
            time.sleep(sleep_seconds)
            backoff_seconds = min(backoff_seconds * 2.0, self.max_backoff_seconds)

        raise RuntimeError("Semantic Scholar request retry loop exhausted unexpectedly")

    def _respect_rate_limit(self) -> None:
        if self._last_request_at <= 0:
            return

        elapsed_seconds = time.monotonic() - self._last_request_at
        sleep_seconds = self.min_request_interval_seconds - elapsed_seconds
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    @staticmethod
    def _is_retryable_http_error(error: HTTPError) -> bool:
        return error.code in {429, 500, 502, 503, 504}

    @staticmethod
    def _retry_after_seconds(error: HTTPError) -> float:
        retry_after = error.headers.get("Retry-After")
        if retry_after is None:
            return 0.0

        try:
            return max(0.0, float(retry_after))
        except ValueError:
            try:
                retry_after_dt = parsedate_to_datetime(retry_after)
            except (TypeError, ValueError):
                return 0.0

            retry_after_epoch = retry_after_dt.timestamp()
            return max(0.0, retry_after_epoch - time.time())

    @staticmethod
    def _minimum_backoff_seconds(status_code: int) -> float:
        if status_code == 429:
            # If server doesn't provide Retry-After, still wait long enough to cool down.
            return 10.0
        return 0.0
