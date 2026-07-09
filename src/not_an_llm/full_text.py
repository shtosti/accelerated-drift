from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import requests


@dataclass(slots=True)
class FullTextResult:
    body: str
    source: str
    url: str
    content_type: str
    page_count: int


class ArxivFullTextProvider:
    def __init__(self, *, cache_dir: Path, timeout_seconds: int = 120) -> None:
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds

    def fetch(self, paper: dict[str, Any]) -> FullTextResult:
        arxiv_id = _arxiv_id(paper)
        if not arxiv_id:
            raise ValueError("Record has no arXiv identifier")

        url = f"https://arxiv.org/pdf/{arxiv_id}"
        cache_path = self.cache_dir / f"{_safe_filename(arxiv_id)}.pdf"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            response = requests.get(
                url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": "not-an-llm/0.1"},
            )
            response.raise_for_status()
            cache_path.write_bytes(response.content)

        body, page_count = _extract_pdf_body(cache_path)
        if not body:
            raise ValueError("PDF extraction produced no body text")
        return FullTextResult(
            body=body,
            source="arxiv_pdf",
            url=url,
            content_type="application/pdf",
            page_count=page_count,
        )


class MedrxivFullTextProvider:
    def __init__(self, *, cache_dir: Path, timeout_seconds: int = 120) -> None:
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds

    def fetch(self, paper: dict[str, Any]) -> FullTextResult:
        content_url = str(paper.get("url") or "").strip()
        if not content_url:
            raise ValueError(
                "Record has no versioned medRxiv content URL"
            )

        url = _medrxiv_pdf_url(content_url)
        paper_id = str(paper.get("paperId") or content_url)
        cache_path = self.cache_dir / f"{_safe_filename(paper_id)}.pdf"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            response = requests.get(
                url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": "not-an-llm/0.1"},
            )
            response.raise_for_status()
            cache_path.write_bytes(response.content)

        body, page_count = _extract_pdf_body(cache_path)
        if not body:
            raise ValueError("PDF extraction produced no body text")
        return FullTextResult(
            body=body,
            source="medrxiv_pdf",
            url=url,
            content_type="application/pdf",
            page_count=page_count,
        )


def build_full_text_provider(
    source: str, *, cache_dir: Path, timeout_seconds: int
) -> ArxivFullTextProvider | MedrxivFullTextProvider:
    if source == "arxiv":
        return ArxivFullTextProvider(
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
        )
    if source == "medarxiv":
        return MedrxivFullTextProvider(
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
        )
    raise NotImplementedError(
        f"Full-text enrichment is not implemented for {source}"
    )


def _extract_pdf_body(path: Path) -> tuple[str, int]:
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PyMuPDF is required for full-text extraction. Run `uv sync`."
        ) from error

    with fitz.open(path) as document:
        pages = [page.get_text("text", sort=True) for page in document]
        page_count = len(document)
    text = _strip_front_matter("\n".join(pages))
    text = _strip_references(text)
    return _normalize_pdf_text(text), page_count


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
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _arxiv_id(paper: dict[str, Any]) -> str:
    external_ids = paper.get("externalIds")
    if isinstance(external_ids, dict) and external_ids.get("ArXiv"):
        return str(external_ids["ArXiv"]).strip()
    return str(paper.get("paperId") or "").strip()


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def _medrxiv_pdf_url(content_url: str) -> str:
    url = content_url.rstrip("/")
    if url.endswith(".full.pdf"):
        return url
    return f"{url}.full.pdf"
