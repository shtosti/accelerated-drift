from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import spacy

from not_an_llm.full_text import (
    FullTextResult,
    MedrxivFullTextProvider,
    _extract_medrxiv_html_body,
    _normalize_pdf_text,
    _medrxiv_html_url,
    _medrxiv_pdf_url,
    _strip_references,
    build_full_text_provider,
)
from not_an_llm.config import load_config
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
                {
                    "paperId": "10.1101/2024.01.02.123456",
                    "publicationDate": "2024-01-05",
                    "url": (
                        "https://www.medrxiv.org/content/"
                        "10.1101/2024.01.02.123456v2"
                    ),
                }
            ),
            "https://www.medrxiv.org/content/medrxiv/early/"
            "2024/01/05/2024.01.02.123456.full.pdf",
        )

    def test_provider_factory_supports_medarxiv(self):
        provider = build_full_text_provider(
            "medarxiv",
            cache_dir=Path("cache"),
            timeout_seconds=42,
        )
        self.assertIsInstance(provider, MedrxivFullTextProvider)
        self.assertEqual(provider.timeout_seconds, 42)

    def test_medrxiv_html_excludes_abstract_tables_and_references(self):
        html = """
        <main>
          <div class="article">
            <h2>Abstract</h2><p>Do not duplicate this abstract.</p>
            <h2>Introduction</h2><p>Keep this prose.</p>
            <table><tr><td>Remove table cells</td></tr></table>
            <figure><p>Remove figure caption.</p></figure>
            <h2>Methods</h2><p>Keep methods.</p>
            <h2>References</h2><p>Remove citation list.</p>
          </div>
        </main>
        """

        body, removed = _extract_medrxiv_html_body(html)

        self.assertEqual(
            body,
            "Introduction\nKeep this prose.\nMethods\nKeep methods.",
        )
        self.assertEqual(removed, 2)

    def test_medrxiv_html_url_uses_static_full_text_route(self):
        paper = {
            "paperId": "10.1101/2024.01.02.123456",
            "publicationDate": "2024-01-05",
        }
        self.assertEqual(
            _medrxiv_html_url(paper),
            "https://www.medrxiv.org/content/medrxiv/early/"
            "2024/01/05/2024.01.02.123456.full",
        )

    def test_arxiv_stamp_is_removed_from_pdf_text(self):
        text = (
            "Introduction\n"
            "arXiv:2305.00593v1 [cs.LG] 30 Apr 2023\n"
            "Body prose."
        )
        self.assertEqual(
            _normalize_pdf_text(text),
            "Introduction\n \nBody prose.",
        )

    def test_medarxiv_mini_configs_share_enriched_cohort(self):
        abstract = load_config("config_mini_medarxiv_abstract.toml")
        full_text = load_config("config_mini_medarxiv_fulltext.toml")

        self.assertEqual(abstract.collection.source, "medarxiv")
        self.assertEqual(abstract.collection.medarxiv_collection_mode, "monthly")
        self.assertEqual(
            abstract.collection.enriched_output_jsonl,
            full_text.collection.enriched_output_jsonl,
        )
        self.assertTrue(abstract.analysis.paired_full_text_only)
        self.assertTrue(full_text.analysis.paired_full_text_only)
        self.assertEqual(abstract.analysis.text_mode, "title_abstract")
        self.assertEqual(full_text.analysis.text_mode, "full_text")
        self.assertNotEqual(
            abstract.analysis.preprocessed_jsonl,
            full_text.analysis.preprocessed_jsonl,
        )
