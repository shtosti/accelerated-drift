from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections import defaultdict
import math
import re
import time

import requests


FULL_TEXT_EXTRACTOR_VERSION = "3"


@dataclass(slots=True)
class FullTextResult:
    body: str
    source: str
    url: str
    content_type: str
    page_count: int
    removed_page_furniture_blocks: int = 0
    removed_table_blocks: int = 0


class ArxivFullTextProvider:
    def __init__(
        self,
        *,
        cache_dir: Path,
        timeout_seconds: int = 120,
        min_request_interval_seconds: float = 3.5,
        max_retries: int = 5,
        initial_backoff_seconds: float = 10.0,
        max_backoff_seconds: float = 60.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._last_request_at = 0.0

    def fetch(self, paper: dict[str, Any]) -> FullTextResult:
        arxiv_id = _arxiv_id(paper)
        if not arxiv_id:
            raise ValueError("Record has no arXiv identifier")

        url = f"https://arxiv.org/pdf/{arxiv_id}"
        cache_path = self.cache_dir / f"{_safe_filename(arxiv_id)}.pdf"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            try:
                content, self._last_request_at = self._download(url)
            except requests.HTTPError as error:
                if error.response is None or error.response.status_code != 404:
                    raise
                unversioned_id = re.sub(r"v\d+$", "", arxiv_id)
                if unversioned_id == arxiv_id:
                    raise
                url = f"https://arxiv.org/pdf/{unversioned_id}"
                content, self._last_request_at = self._download(url)
            cache_path.write_bytes(content)

        (
            body,
            page_count,
            removed_page_furniture_blocks,
            removed_table_blocks,
        ) = _extract_pdf_body(cache_path)
        if not body:
            raise ValueError("PDF extraction produced no body text")
        return FullTextResult(
            body=body,
            source="arxiv_pdf",
            url=url,
            content_type="application/pdf",
            page_count=page_count,
            removed_page_furniture_blocks=removed_page_furniture_blocks,
            removed_table_blocks=removed_table_blocks,
        )

    def _download(self, url: str) -> tuple[bytes, float]:
        return _download_with_retries(
            url,
            timeout_seconds=self.timeout_seconds,
            min_request_interval_seconds=self.min_request_interval_seconds,
            max_retries=self.max_retries,
            initial_backoff_seconds=self.initial_backoff_seconds,
            max_backoff_seconds=self.max_backoff_seconds,
            last_request_at=self._last_request_at,
        )


class MedrxivFullTextProvider:
    def __init__(
        self,
        *,
        cache_dir: Path,
        timeout_seconds: int = 120,
        min_request_interval_seconds: float = 3.5,
        max_retries: int = 5,
        initial_backoff_seconds: float = 10.0,
        max_backoff_seconds: float = 60.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.min_request_interval_seconds = min_request_interval_seconds
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._last_request_at = 0.0

    def fetch(self, paper: dict[str, Any]) -> FullTextResult:
        content_url = str(paper.get("url") or "").strip()
        if not content_url:
            raise ValueError(
                "Record has no versioned medRxiv content URL"
            )

        paper_id = str(paper.get("paperId") or content_url)
        url = _medrxiv_html_url(paper)
        cache_path = self.cache_dir / f"{_safe_filename(paper_id)}.html"
        legacy_pdf_path = self.cache_dir / f"{_safe_filename(paper_id)}.pdf"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists() and legacy_pdf_path.exists():
            (
                body,
                page_count,
                removed_page_furniture_blocks,
                removed_table_blocks,
            ) = _extract_pdf_body(legacy_pdf_path)
            if not body:
                raise ValueError("Cached PDF extraction produced no body text")
            return FullTextResult(
                body=body,
                source="medrxiv_pdf_cleaned",
                url=_medrxiv_pdf_url(paper),
                content_type="application/pdf",
                page_count=page_count,
                removed_page_furniture_blocks=(
                    removed_page_furniture_blocks
                ),
                removed_table_blocks=removed_table_blocks,
            )

        if not cache_path.exists():
            cache_path.write_bytes(self._download_content(url))

        body, removed_table_blocks = _extract_medrxiv_html_body(
            cache_path.read_text(encoding="utf-8")
        )
        if not body:
            raise ValueError("HTML extraction produced no body text")
        return FullTextResult(
            body=body,
            source="medrxiv_html",
            url=url,
            content_type="text/html",
            page_count=0,
            removed_page_furniture_blocks=0,
            removed_table_blocks=removed_table_blocks,
        )

    def _download_content(self, url: str) -> bytes:
        backoff = self.initial_backoff_seconds
        for attempt in range(self.max_retries + 1):
            elapsed = time.monotonic() - self._last_request_at
            delay = self.min_request_interval_seconds - elapsed
            if self._last_request_at and delay > 0:
                time.sleep(delay)

            response = requests.get(
                url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": "not-an-llm/0.1"},
            )
            self._last_request_at = time.monotonic()
            if response.status_code != 429:
                response.raise_for_status()
                return response.content
            if attempt >= self.max_retries:
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After")
            try:
                retry_delay = float(retry_after) if retry_after else 0.0
            except ValueError:
                retry_delay = 0.0
            time.sleep(max(backoff, retry_delay))
            backoff = min(backoff * 2, self.max_backoff_seconds)

        raise RuntimeError("medRxiv PDF retry loop exhausted")


def build_full_text_provider(
    source: str,
    *,
    cache_dir: Path,
    timeout_seconds: int,
    min_request_interval_seconds: float = 3.5,
    max_retries: int = 5,
    initial_backoff_seconds: float = 10.0,
    max_backoff_seconds: float = 60.0,
) -> ArxivFullTextProvider | MedrxivFullTextProvider:
    if source == "arxiv":
        return ArxivFullTextProvider(
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
            min_request_interval_seconds=min_request_interval_seconds,
            max_retries=max_retries,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        )
    if source == "medarxiv":
        return MedrxivFullTextProvider(
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
            min_request_interval_seconds=min_request_interval_seconds,
            max_retries=max_retries,
            initial_backoff_seconds=initial_backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        )
    raise NotImplementedError(
        f"Full-text enrichment is not implemented for {source}"
    )


def _extract_pdf_body(path: Path) -> tuple[str, int, int, int]:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PyMuPDF is required for full-text extraction. Run `uv sync`."
        ) from error

    with fitz.open(path) as document:
        page_blocks = [
            _page_blocks(page, page_number=index + 1)
            for index, page in enumerate(document)
        ]
        repeated_margin_text = _repeated_margin_text(page_blocks)
        cleaned_pages = []
        removed_page_furniture_blocks = 0
        removed_table_blocks = 0
        for blocks in page_blocks:
            kept = []
            for block in blocks:
                if _is_page_furniture(block, repeated_margin_text):
                    removed_page_furniture_blocks += 1
                    continue
                if block["in_table"]:
                    removed_table_blocks += 1
                    continue
                kept.append(block["text"])
            cleaned_pages.append("\n".join(kept))
        page_count = len(document)

    text = _strip_front_matter("\n".join(cleaned_pages))
    text = _strip_references(text)
    return (
        _normalize_pdf_text(text),
        page_count,
        removed_page_furniture_blocks,
        removed_table_blocks,
    )


def _page_blocks(page, *, page_number: int) -> list[dict[str, Any]]:
    table_boxes = []
    try:
        table_boxes = [tuple(table.bbox) for table in page.find_tables().tables]
    except Exception:
        # Table detection is opportunistic; extraction should still succeed.
        table_boxes = []

    height = float(page.rect.height)
    blocks = []
    for raw_block in page.get_text("blocks", sort=True):
        x0, y0, x1, y1, raw_text, *_ = raw_block
        text = str(raw_text).strip()
        if not text:
            continue
        bbox = (float(x0), float(y0), float(x1), float(y1))
        blocks.append(
            {
                "text": text,
                "normalized": _normalize_margin_text(text),
                "bbox": bbox,
                "page_number": page_number,
                "height": height,
                "in_margin": y0 < height * 0.15 or y1 > height * 0.85,
                "in_table": any(
                    _block_overlaps_table(bbox, table_box)
                    for table_box in table_boxes
                ),
            }
        )
    return blocks


def _repeated_margin_text(
    pages: list[list[dict[str, Any]]],
) -> set[str]:
    occurrences: dict[str, set[int]] = defaultdict(set)
    for blocks in pages:
        for block in blocks:
            if block["in_margin"] and block["normalized"]:
                occurrences[block["normalized"]].add(block["page_number"])
    threshold = max(2, math.ceil(len(pages) * 0.2))
    return {
        text
        for text, page_numbers in occurrences.items()
        if len(page_numbers) >= threshold
    }


def _is_page_furniture(
    block: dict[str, Any], repeated_margin_text: set[str]
) -> bool:
    text = " ".join(block["text"].split())
    lowered = text.lower()
    _, y0, _, y1 = block["bbox"]
    height = block["height"]

    if block["in_margin"] and block["normalized"] in repeated_margin_text:
        return True
    if y1 > height * 0.85 and re.fullmatch(r"(?:page\s*)?\d+", lowered):
        return True
    if y0 < height * 0.08 and any(
        phrase in lowered
        for phrase in (
            "medrxiv preprint",
            "not certified by peer review",
            "copyright holder for this preprint",
            "author/funder, who has granted medrxiv",
            "license to display the preprint in perpetuity",
            "it is made available under a",
            "international license",
        )
    ):
        return True
    if lowered.startswith(
        "note: this preprint reports new research that has not been certified"
    ):
        return True
    if re.match(
        r"^(?:figure|fig\.?|table)\s+[a-z]?\d+(?:[.:]|\s+-)",
        lowered,
    ):
        return True
    return False


def _normalize_margin_text(text: str) -> str:
    normalized = " ".join(text.lower().split())
    normalized = re.sub(r"https?://\S+", "<url>", normalized)
    normalized = re.sub(r"\b\d+(?:\.\d+)*\b", "<number>", normalized)
    return normalized


def _block_overlaps_table(
    block: tuple[float, float, float, float],
    table: tuple[float, float, float, float],
) -> bool:
    bx0, by0, bx1, by1 = block
    tx0, ty0, tx1, ty1 = table
    ix0, iy0 = max(bx0, tx0), max(by0, ty0)
    ix1, iy1 = min(bx1, tx1), min(by1, ty1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    intersection = (ix1 - ix0) * (iy1 - iy0)
    block_area = max((bx1 - bx0) * (by1 - by0), 1.0)
    return intersection / block_area >= 0.25


def _strip_front_matter(text: str) -> str:
    # Preprocessing adds the canonical title and abstract, so retain only the body.
    match = re.search(
        r"(?im)^\s*(?:1[\.\s]+)?(?:introduction|background)\s*$",
        text,
    )
    return text[match.start():] if match else text


def _strip_references(text: str) -> str:
    match = re.search(
        r"(?im)^\s*(?:\d+[\.\s]+)?(?:references|bibliography)\s*$",
        text,
    )
    return text[:match.start()] if match else text


def _normalize_pdf_text(text: str) -> str:
    text = re.sub(
        r"arXiv:\d{4}\.\d{4,5}v\d+\s*"
        r"(?:\[[^\]]+\])?\s*"
        r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_medrxiv_html_body(html: str) -> tuple[str, int]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as error:
        raise RuntimeError(
            "BeautifulSoup is required for medRxiv HTML extraction. Run `uv sync`."
        ) from error

    soup = BeautifulSoup(html, "html.parser")
    removable = soup.select(
        "table, figure, .fig, .figure, .table, .table-inline, "
        ".table-expansion, .supplementary-material, .supplementary-data"
    )
    for node in removable:
        node.decompose()

    container = (
        soup.select_one("div.article")
        or soup.select_one("article")
        or soup.select_one("#content-block")
        or soup.select_one("main")
    )
    if container is None:
        raise ValueError("Could not locate medRxiv article content in HTML")

    parts: list[str] = []
    in_body = False
    in_abstract = False
    for node in container.select("h2, h3, h4, p"):
        text = " ".join(node.get_text(" ", strip=True).split())
        if not text:
            continue
        if node.name in {"h2", "h3", "h4"}:
            heading = text.rstrip(":").strip().lower()
            if heading in {"references", "bibliography", "literature cited"}:
                break
            if heading in {"abstract", "summary"}:
                in_abstract = True
                continue
            if in_abstract or not in_body:
                in_abstract = False
                in_body = True
            if in_body:
                parts.append(text)
            continue
        if in_body and not in_abstract:
            parts.append(text)

    return _normalize_pdf_text("\n".join(parts)), len(removable)


def _arxiv_id(paper: dict[str, Any]) -> str:
    external_ids = paper.get("externalIds")
    if isinstance(external_ids, dict) and external_ids.get("ArXiv"):
        return str(external_ids["ArXiv"]).strip()
    return str(paper.get("paperId") or "").strip()


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _download_with_retries(
    url: str,
    *,
    timeout_seconds: int,
    min_request_interval_seconds: float,
    max_retries: int,
    initial_backoff_seconds: float,
    max_backoff_seconds: float,
    last_request_at: float,
) -> tuple[bytes, float]:
    backoff = initial_backoff_seconds
    for attempt in range(max_retries + 1):
        elapsed = time.monotonic() - last_request_at
        delay = min_request_interval_seconds - elapsed
        if last_request_at and delay > 0:
            time.sleep(delay)

        response = requests.get(
            url,
            timeout=timeout_seconds,
            headers={"User-Agent": "not-an-llm/0.1"},
        )
        last_request_at = time.monotonic()
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
            return response.content, last_request_at
        if attempt >= max_retries:
            response.raise_for_status()

        retry_after = response.headers.get("Retry-After")
        try:
            retry_delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            retry_delay = 0.0
        time.sleep(max(backoff, retry_delay))
        backoff = min(backoff * 2, max_backoff_seconds)

    raise RuntimeError("Download retry loop exhausted")


def _medrxiv_pdf_url(paper: dict[str, Any]) -> str:
    paper_id = str(paper.get("paperId") or "").strip()
    publication_date = str(paper.get("publicationDate") or "").strip()
    if paper_id and publication_date:
        doi_suffix = paper_id.rsplit("/", 1)[-1]
        date_path = publication_date.replace("-", "/")
        return (
            "https://www.medrxiv.org/content/medrxiv/early/"
            f"{date_path}/{doi_suffix}.full.pdf"
        )

    content_url = str(paper.get("url") or "").strip().rstrip("/")
    if content_url.endswith(".full.pdf"):
        return content_url
    return f"{content_url}.full.pdf"


def _medrxiv_html_url(paper: dict[str, Any]) -> str:
    return _medrxiv_pdf_url(paper).removesuffix(".pdf")
