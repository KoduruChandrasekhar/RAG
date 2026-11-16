"""Tests for the CLI, the metadata filters, and the retrieval evaluator."""
import json

import pytest

import paths
from evaluate_retrieval import is_relevant, score_question
from query_engine import SECTION_CHOICES, FilingScopeEngine, build_parser, format_result

from test_query_engine import make_engine


# ------------------------------------------------------------- where clauses

def test_where_is_none_without_filters():
    assert FilingScopeEngine._build_where() is None
    assert FilingScopeEngine._build_where([], []) is None


def test_where_uses_equality_for_a_single_value():
    assert FilingScopeEngine._build_where(["MSFT"]) == {"ticker": "MSFT"}
    assert FilingScopeEngine._build_where(None, ["Item_7_MDA"]) == {"section": "Item_7_MDA"}


def test_where_uses_in_for_several_values():
    assert FilingScopeEngine._build_where(["MSFT", "AAPL"]) == {
        "ticker": {"$in": ["MSFT", "AAPL"]}}


def test_where_combines_fields_with_and():
    assert FilingScopeEngine._build_where(["MSFT"], ["Item_7_MDA"]) == {
        "$and": [{"ticker": "MSFT"}, {"section": "Item_7_MDA"}]}


def test_where_combines_multi_value_fields():
    assert FilingScopeEngine._build_where(["MSFT", "AAPL"], ["Item_7_MDA"]) == {
        "$and": [{"ticker": {"$in": ["MSFT", "AAPL"]}}, {"section": "Item_7_MDA"}]}


# ------------------------------------------------------- section filtering

def chunk(cid, ticker="AAPL", section="Item_1A_Risk_Factors"):
    return {"chunk_id": cid, "ticker": ticker, "filing_id": "f",
            "section": section, "text": "t"}


class StubBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, tokens):
        return self.scores


def test_sparse_retrieve_filters_by_section():
    chunks = [chunk("r", section="Item_1A_Risk_Factors"),
              chunk("m", section="Item_7_MDA")]
    engine = make_engine(all_chunks=chunks, bm25=StubBM25([0.9, 0.8]))
    out = engine._sparse_retrieve("q", n=5, sections=["Item_7_MDA"])
    assert [c["id"] for c in out] == ["m"]


def test_sparse_retrieve_applies_both_filters():
    chunks = [chunk("a", "AAPL", "Item_7_MDA"),
              chunk("b", "MSFT", "Item_7_MDA"),
              chunk("c", "MSFT", "Item_1A_Risk_Factors")]
    engine = make_engine(all_chunks=chunks, bm25=StubBM25([0.9, 0.8, 0.7]))
    out = engine._sparse_retrieve("q", n=5, tickers=["MSFT"],
                                 sections=["Item_7_MDA"])
    assert [c["id"] for c in out] == ["b"]


# ------------------------------------------------- retrieve() vs query()

def retrieved(cid, ticker="MSFT", section="Item_1A_Risk_Factors"):
    # Shape returned by _dense_retrieve / _sparse_retrieve, which _fuse merges.
    return {"id": cid, "text": "body", "dense_rank": 1, "sparse_rank": 1,
            "metadata": {"ticker": ticker, "filing_id": "f", "section": section}}


def build_retrieval_engine():
    # Stub the two retrievers rather than the fusion, so real RRF code runs.
    engine = make_engine()
    engine._dense_retrieve = lambda q, n=20, tickers=None, sections=None: [retrieved("c1")]
    engine._sparse_retrieve = lambda q, n=20, tickers=None, sections=None: [retrieved("c1")]
    engine._rerank = lambda q, c, top_k=5: c[:top_k]
    return engine


def test_retrieve_makes_no_llm_call():
    engine = build_retrieval_engine()

    def explode(*args, **kwargs):
        raise AssertionError("retrieve() must not generate")

    engine._generate_answer = explode
    result = engine.retrieve("Microsoft risks?")
    assert "answer" not in result
    assert result["sources"][0]["ticker"] == "MSFT"


def test_retrieve_reports_the_scope_it_used():
    engine = build_retrieval_engine()
    result = engine.retrieve("Microsoft risks?", sections=["Item_7_MDA"])
    assert result["tickers"] == ["MSFT"]
    assert result["sections"] == ["Item_7_MDA"]


def test_query_adds_an_answer_to_the_retrieval_result():
    engine = build_retrieval_engine()
    engine._generate_answer = lambda q, chunks: f"answer from {len(chunks)} chunk(s)"
    result = engine.query("Microsoft risks?")
    assert result["answer"] == "answer from 1 chunk(s)"
    assert result["sources"]


def test_query_passes_source_metadata_to_generation():
    # The prompt's citation header reads ticker/filing_id/section off the
    # metadata dict, so query() must hand those through unchanged.
    engine = build_retrieval_engine()
    seen = {}
    engine._generate_answer = lambda q, chunks: seen.update(chunks[0]["metadata"]) or "a"
    engine.query("Microsoft risks?")
    assert seen["ticker"] == "MSFT"
    assert seen["section"] == "Item_1A_Risk_Factors"


def test_retrieve_falls_back_when_a_section_filter_matches_nothing():
    engine = make_engine()
    seen = []

    def dense(q, n=20, tickers=None, sections=None):
        seen.append((tickers, sections))
        return [] if sections else [retrieved("c1")]

    engine._dense_retrieve = dense
    engine._sparse_retrieve = lambda q, n=20, tickers=None, sections=None: []
    engine._rerank = lambda q, c, top_k=5: c
    result = engine.retrieve("Microsoft risks?", sections=["Item_7_MDA"])
    assert len(seen) == 2            # filtered attempt, then whole corpus
    assert result["sources"]
    assert result["trace"]["widened_to_full_corpus"] is True


def test_retrieve_honours_top_k():
    engine = make_engine()
    many = [retrieved(f"c{i}", ticker="AAPL") for i in range(10)]
    engine._dense_retrieve = lambda q, n=20, tickers=None, sections=None: many
    engine._sparse_retrieve = lambda q, n=20, tickers=None, sections=None: []
    engine._rerank = lambda q, c, top_k=5: c[:top_k]
    assert len(engine.retrieve("q", top_k=3)["sources"]) == 3


# ------------------------------------------------------------------- CLI

def test_parser_defaults():
    args = build_parser().parse_args(["What", "are", "the", "risks?"])
    assert " ".join(args.question) == "What are the risks?"
    assert args.ticker is None
    assert args.section is None
    assert args.top_k == 5
    assert not args.retrieval_only
    assert not args.interactive
    assert not args.all_companies


def test_parser_accepts_repeated_tickers():
    args = build_parser().parse_args(["-t", "MSFT", "-t", "AAPL", "q"])
    assert args.ticker == ["MSFT", "AAPL"]


def test_parser_rejects_an_unknown_ticker():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-t", "TSLA", "q"])


def test_parser_rejects_an_unknown_section():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-s", "balance_sheet", "q"])


def test_parser_flags():
    args = build_parser().parse_args(["-r", "-i", "-a", "-k", "9", "-s", "mda", "q"])
    assert args.retrieval_only and args.interactive and args.all_companies
    assert args.top_k == 9
    assert args.section == ["mda"]


def test_section_choices_map_to_real_section_names():
    actual = {"Item_1A_Risk_Factors", "Item_7_MDA"}
    assert set(SECTION_CHOICES.values()) == actual


def test_format_result_without_an_answer_omits_the_answer_block():
    result = {"tickers": ["MSFT"], "sections": [], "sources": [
        {"ticker": "MSFT", "filing_id": "f", "section": "s", "text": "body"}]}
    out = format_result(result)
    assert "ANSWER" not in out
    assert "SOURCES" in out
    assert "[1] MSFT" in out


def test_format_result_with_an_answer_includes_it():
    result = {"tickers": [], "sections": [], "answer": "the answer", "sources": []}
    out = format_result(result)
    assert "the answer" in out
    assert "all companies" in out


def test_format_result_can_include_passage_text():
    result = {"tickers": [], "sections": [], "sources": [
        {"ticker": "AAPL", "filing_id": "f", "section": "s",
         "text": "distinctive passage body"}]}
    assert "distinctive passage body" in format_result(result, show_text=True)
    assert "distinctive passage body" not in format_result(result, show_text=False)


# --------------------------------------------------- retrieval evaluator

def source(ticker, section):
    return {"ticker": ticker, "filing_id": "f", "section": section, "text": "t"}


def test_is_relevant_requires_both_ticker_and_section():
    s = source("MSFT", "Item_7_MDA")
    assert is_relevant(s, ["MSFT"], ["Item_7_MDA"])
    assert not is_relevant(s, ["AAPL"], ["Item_7_MDA"])
    assert not is_relevant(s, ["MSFT"], ["Item_1A_Risk_Factors"])


def test_is_relevant_ignores_an_unspecified_dimension():
    s = source("MSFT", "Item_7_MDA")
    assert is_relevant(s, ["MSFT"], [])
    assert is_relevant(s, [], ["Item_7_MDA"])


def test_score_question_on_a_perfect_result():
    result = {"tickers": ["MSFT"], "sources": [source("MSFT", "Item_7_MDA")] * 3}
    item = {"expected_tickers": ["MSFT"], "expected_sections": ["Item_7_MDA"]}
    score = score_question(result, item)
    assert score["hit"] and score["rr"] == 1.0
    assert score["precision"] == 1.0
    assert score["ticker_ok"]


def test_score_question_on_a_total_miss():
    # This is the pre-fix failure: a Microsoft question answered from Amazon.
    result = {"tickers": [], "sources": [source("AMZN", "Item_1A_Risk_Factors")] * 5}
    item = {"expected_tickers": ["MSFT"], "expected_sections": ["Item_1A_Risk_Factors"]}
    score = score_question(result, item)
    assert not score["hit"]
    assert score["rr"] == 0.0
    assert score["precision"] == 0.0
    assert not score["ticker_ok"]


def test_reciprocal_rank_reflects_the_first_relevant_position():
    result = {"tickers": ["MSFT"], "sources": [
        source("AAPL", "Item_7_MDA"),
        source("AAPL", "Item_7_MDA"),
        source("MSFT", "Item_7_MDA")]}
    item = {"expected_tickers": ["MSFT"], "expected_sections": ["Item_7_MDA"]}
    score = score_question(result, item)
    assert score["rr"] == pytest.approx(1 / 3)
    assert score["precision"] == pytest.approx(1 / 3)


def test_score_question_handles_no_sources():
    score = score_question({"tickers": [], "sources": []},
                           {"expected_tickers": ["MSFT"]})
    assert not score["hit"] and score["precision"] == 0.0


# ------------------------------------------------------------- test data

def test_every_question_is_labelled_and_consistent():
    data = json.load(open(paths.TEST_QUESTIONS, encoding="utf-8"))
    assert data
    valid_sections = {"Item_1A_Risk_Factors", "Item_7_MDA"}
    for item in data:
        assert item["question"].strip()
        assert item["ground_truth"].strip()
        assert item["expected_tickers"], item["question"]
        assert set(item["expected_tickers"]) <= set(paths.TICKERS)
        assert set(item["expected_sections"]) <= valid_sections


def test_test_set_covers_every_company():
    data = json.load(open(paths.TEST_QUESTIONS, encoding="utf-8"))
    covered = {t for item in data for t in item["expected_tickers"]}
    assert covered == set(paths.TICKERS)


def test_detection_agrees_with_the_labels():
    # If the aliases cannot find the company the label names, scoping would
    # silently fall back to the whole corpus for that question.
    data = json.load(open(paths.TEST_QUESTIONS, encoding="utf-8"))
    engine = make_engine()
    for item in data:
        detected = engine._detect_tickers(item["question"])
        assert sorted(detected) == sorted(item["expected_tickers"]), item["question"]
