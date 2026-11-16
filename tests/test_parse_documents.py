"""Tests for the parsing stage.

The section-boundary cases here are the ones that were silently wrong before:
a heading in the table of contents and a heading used as a cross-reference both
look like the real thing, and picking the wrong one put MD&A text into the
risk-factor chunks.
"""
import pytest

from parse_documents import (
    chunk_text,
    clean_html,
    extract_10k_document,
    extract_sections,
    find_best_match,
    is_cross_reference,
    is_table_of_contents,
    spaced,
)

ITEM_1A_PATTERNS = [
    r'(?i)item\s+1a[\.\-\:\s–—]+' + spaced('risk') + r'\s+' + spaced('factors'),
]


def test_spaced_tolerates_split_words():
    import re
    # Microsoft's filings render the heading as "ITEM 1A. RIS K FACTORS".
    assert re.search(spaced("RISK"), "RIS K")
    assert re.search(spaced("RISK"), "RISK")


def test_table_of_contents_is_detected():
    text = "Item 1A. Risk Factors 16 Item 1B. Unresolved Staff Comments 30"
    assert is_table_of_contents(text, text.index(" 16"))


def test_body_heading_is_not_table_of_contents():
    text = "Item 1A. Risk Factors Our operations are subject to various risks."
    assert not is_table_of_contents(text, text.index(" Our"))


@pytest.mark.parametrize("tail", [
    '” for a discussion of these impacts.',
    ' of this Annual Report on Form 10-K.',
    ' – Risks Related to Regulatory Matters” for a discussion.',
    '; Trends in Our Business in Part II, Item 7.',
    '.” Forward-looking statements provide current expectations.',
])
def test_cross_references_are_detected(tail):
    text = "Item 1A. Risk Factors" + tail
    assert is_cross_reference(text, len("Item 1A. Risk Factors"))


@pytest.mark.parametrize("tail", [
    ' Our operations and financial results are subject to various risks.',
    ' The following risk factors should be considered.',
    ' Please carefully consider the following discussion.',
])
def test_real_headings_are_not_cross_references(tail):
    text = "Item 1A. Risk Factors" + tail
    assert not is_cross_reference(text, len("Item 1A. Risk Factors"))


def test_find_best_match_skips_toc_and_crossrefs():
    text = (
        "Item 1A. Risk Factors 16 Item 1B. Unresolved Staff Comments 30 "          # TOC
        "... see Item 1A. Risk Factors of this Annual Report on Form 10-K. "        # xref
        "ITEM 1A. RISK FACTORS Our operations and financial results are subject "   # real
        "to various risks and uncertainties described below."
    )
    match = find_best_match(text, ITEM_1A_PATTERNS)
    assert match is not None
    assert text[match.end():match.end() + 4] == " Our"


def test_find_best_match_returns_none_when_absent():
    assert find_best_match("nothing relevant here", ITEM_1A_PATTERNS) is None


def test_find_best_match_honours_after_position():
    text = "ITEM 1A. RISK FACTORS Our business is risky and so forth."
    assert find_best_match(text, ITEM_1A_PATTERNS, after_position=50) is None


def test_find_best_match_ignores_invalid_patterns():
    text = "ITEM 1A. RISK FACTORS Our business faces risks."
    match = find_best_match(text, ["(unclosed group"] + ITEM_1A_PATTERNS)
    assert match is not None


def test_clean_html_strips_tags_scripts_and_entities():
    raw = ("<html><body><script>bad()</script><style>p{}</style>"
           "<p>Revenue grew&#160;by&#8201;10&#37;</p></body></html>")
    text = clean_html(raw)
    assert "bad()" not in text
    assert "p{}" not in text
    assert "Revenue grew" in text
    assert "  " not in text


def test_extract_10k_document_picks_the_10k_submission():
    raw = ("<TYPE>EX-21.1<TEXT>exhibit noise</TEXT>"
           "<TYPE>10-K<TEXT>the annual report body</TEXT>")
    assert extract_10k_document(raw).strip() == "the annual report body"


def test_extract_10k_document_falls_back_to_whole_input():
    raw = "no submission markers at all"
    assert extract_10k_document(raw) == raw


def test_chunk_text_overlaps_and_respects_size():
    words = [f"w{i}" for i in range(400)]
    chunks = chunk_text(" ".join(words), chunk_size=150, overlap=50)
    assert all(len(c.split()) <= 150 for c in chunks)
    # consecutive chunks share `overlap` words
    first, second = chunks[0].split(), chunks[1].split()
    assert first[100:150] == second[:50]


def test_chunk_text_drops_trailing_fragments():
    # A tail shorter than 50 characters is not worth embedding.
    assert chunk_text("tiny") == []


def test_extract_sections_requires_substantial_content():
    # A heading with only a few words after it is a cross-reference, not a
    # section, and must not be emitted.
    text = ("padding " * 200) + "ITEM 1A. RISK FACTORS Our risks are few." + (" tail" * 10)
    assert "Item_1A_Risk_Factors" not in extract_sections(text, "X", "Y")
