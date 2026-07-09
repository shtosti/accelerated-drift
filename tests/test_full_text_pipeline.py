from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import spacy

from not_an_llm.full_text import (
    FullTextResult,
    MedrxivFullTextProvider,
    _medrxiv_pdf_url,
    _strip_references,
    build_full_text_provider,
)
from not_an_llm.pipelines.enrich import _enrich_record
from not_an_llm.preprocessing.text import TextPreprocessor


def _blank_nlp():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


class FullTextPipelineTests(TestCase):
    @patch.object(TextPreprocessor, "_load_nlp", staticmethod(_blank_nlp))
    def test_full_text_mode_combines_title_abstract_and_body(self):
        frame = pd.DataFrame(
            [{
                "title": "A title",
                "abstract": "An abstract.",
                "full_text_body": "Introduction. The body.",
                "year": 2024,
            }]
        )
        result = TextPreprocessor(
            text_mode="full_text"
        ).preprocess_dataframe(frame)
        self.assertEqual(
            result.loc[0, "text_raw"],
            "A title An abstract. Introduction. The body.",
        )

    @patch.object(TextPreprocessor, "_load_nlp", staticmethod(_blank_nlp))
    def test_title_abstract_mode_does_not_include_body(self):
        frame = pd.DataFrame(
            [{
                "title": "A title",
                "abstract": "An abstract.",
                "full_text_body": "The body.",
                "year": 2024,
            }]
        )
        result = TextPreprocessor(
            text_mode="title_abstract"
        ).preprocess_dataframe(frame)
        self.assertEqual(result.loc[0, "text_raw"], "A title An abstract.")

    def test_enrichment_preserves_raw_record_and_adds_metadata(self):
        class Provider:
            def fetch(self, record):
                return FullTextResult(
                    body="Introduction. Body text.",
                    source="arxiv_pdf",
                    url="https://arxiv.org/pdf/1234.5678",
                    content_type="application/pdf",
                    page_count=3,
                )

        raw = {"paperId": "1234.5678", "title": "Original title", "custom": 7}
        enriched = _enrich_record(raw, Provider())
        self.assertEqual(enriched["title"], raw["title"])
        self.assertEqual(enriched["custom"], 7)
        self.assertEqual(enriched["full_text_status"], "available")
        self.assertEqual(enriched["full_text_body"], "Introduction. Body text.")
        self.assertEqual(enriched["full_text_page_count"], 3)

    def test_enrichment_records_failure_without_dropping_paper(self):
        class Provider:
            def fetch(self, record):
                raise RuntimeError("broken PDF")

        raw = {"paperId": "1234.5678", "title": "Original title"}
        enriched = _enrich_record(raw, Provider())
        self.assertEqual(enriched["paperId"], raw["paperId"])
        self.assertEqual(enriched["full_text_status"], "failed")
        self.assertEqual(enriched["full_text_body"], "")
        self.assertIn("broken PDF", enriched["full_text_error"])

    def test_reference_section_is_excluded_from_body(self):
        text = "1 Introduction\nBody text.\nReferences\n[1] Citation"
        self.assertEqual(
            _strip_references(text),
            "1 Introduction\nBody text.\n",
        )

    def test_medrxiv_pdf_url_uses_versioned_content_url(self):
        self.assertEqual(
            _medrxiv_pdf_url(
                "https://www.medrxiv.org/content/10.1101/2024.01.02.123456v2"
            ),
            "https://www.medrxiv.org/content/"
            "10.1101/2024.01.02.123456v2.full.pdf",
        )

    def test_provider_factory_supports_medarxiv(self):
        provider = build_full_text_provider(
            "medarxiv",
            cache_dir=Path("cache"),
            timeout_seconds=42,
        )
        self.assertIsInstance(provider, MedrxivFullTextProvider)
        self.assertEqual(provider.timeout_seconds, 42)
