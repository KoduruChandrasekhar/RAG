"""Tests for the retrieval and generation stages.

FilingScopeEngine.__init__ loads two transformer models and opens ChromaDB, so these
build a bare instance with __new__ and attach only what each method needs.
"""
import pytest

import paths
from query_engine import FilingScopeEngine


def make_engine(**attrs):
    engine = FilingScopeEngine.__new__(FilingScopeEngine)
    for key, value in attrs.items():
        setattr(engine, key, value)
    return engine


def chunk(cid, ticker="AAPL", section="Item_1A_Risk_Factors", text="body text"):
    return {"chunk_id": cid, "ticker": ticker, "filing_id": "f1",
            "section": section, "text": text}


# --------------------------------------------------------------- ticker scope

@pytest.mark.parametrize("question,expected", [
    ("What risks did Apple flag about China?", ["AAPL"]),
    ("Microsoft cybersecurity risks?", ["MSFT"]),
    ("How does Alphabet describe competition?", ["GOOGL"]),
    ("NVIDIA export controls", ["NVDA"]),
    ("What does AWS contribute to revenue?", ["AMZN"]),
    ("What are the main industry risks?", []),
])
def test_detect_tickers(question, expected):
    engine = make_engine()
    assert engine._detect_tickers(question) == expected


def test_detect_tickers_finds_several():
    engine = make_engine()
    found = engine._detect_tickers("Compare Apple and Microsoft supply chains")
    assert set(found) == {"AAPL", "MSFT"}


def test_detect_tickers_is_case_insensitive():
    engine = make_engine()
    assert engine._detect_tickers("NVIDIA and nvidia and Nvidia") == ["NVDA"]


# ------------------------------------------------------------ sparse retrieval

class StubBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, tokens):
        return self.scores


def test_sparse_retrieve_orders_by_score():
    chunks = [chunk("a"), chunk("b"), chunk("c")]
    engine = make_engine(all_chunks=chunks, bm25=StubBM25([0.1, 0.9, 0.5]))
    out = engine._sparse_retrieve("q", n=3)
    assert [c["id"] for c in out] == ["b", "c", "a"]
    assert [c["sparse_rank"] for c in out] == [1, 2, 3]


def test_sparse_retrieve_filters_by_ticker():
    chunks = [chunk("a", "AAPL"), chunk("m", "MSFT"), chunk("n", "NVDA")]
    engine = make_engine(all_chunks=chunks, bm25=StubBM25([0.9, 0.5, 0.8]))
    out = engine._sparse_retrieve("q", n=5, tickers=["MSFT"])
    assert [c["id"] for c in out] == ["m"]


def test_sparse_retrieve_filter_can_span_tickers():
    chunks = [chunk("a", "AAPL"), chunk("m", "MSFT"), chunk("n", "NVDA")]
    engine = make_engine(all_chunks=chunks, bm25=StubBM25([0.9, 0.5, 0.8]))
    out = engine._sparse_retrieve("q", n=5, tickers=["MSFT", "NVDA"])
    assert [c["id"] for c in out] == ["n", "m"]


# ------------------------------------------------------------ dense retrieval

class StubCollection:
    def __init__(self):
        self.last_where = "unset"

    def query(self, query_texts, n_results, where=None):
        self.last_where = where
        return {"documents": [["d1", "d2"]],
                "metadatas": [[{"ticker": "AAPL"}, {"ticker": "AAPL"}]],
                "ids": [["i1", "i2"]]}


def test_dense_retrieve_sends_no_filter_when_unscoped():
    collection = StubCollection()
    engine = make_engine(collection=collection)
    out = engine._dense_retrieve("q", n=2)
    assert collection.last_where is None
    assert [c["dense_rank"] for c in out] == [1, 2]


def test_dense_retrieve_sends_equality_filter_for_one_ticker():
    collection = StubCollection()
    make_engine(collection=collection)._dense_retrieve("q", 2, tickers=["MSFT"])
    assert collection.last_where == {"ticker": "MSFT"}


def test_dense_retrieve_sends_in_filter_for_several_tickers():
    collection = StubCollection()
    make_engine(collection=collection)._dense_retrieve("q", 2, tickers=["MSFT", "AAPL"])
    assert collection.last_where == {"ticker": {"$in": ["MSFT", "AAPL"]}}


# --------------------------------------------------------------- RRF fusion

def test_hybrid_retrieve_rewards_agreement_between_retrievers():
    # "shared" is rank 2 in both lists; each other doc is rank 1 in one list
    # only. Reciprocal-rank fusion should put the agreed-upon doc first.
    dense = [{"id": "dense_only", "text": "x", "metadata": {}, "dense_rank": 1},
             {"id": "shared", "text": "x", "metadata": {}, "dense_rank": 2}]
    sparse = [{"id": "sparse_only", "text": "x", "metadata": {}, "sparse_rank": 1},
              {"id": "shared", "text": "x", "metadata": {}, "sparse_rank": 2}]

    engine = make_engine()
    engine._dense_retrieve = lambda q, n, tickers=None, sections=None: dense
    engine._sparse_retrieve = lambda q, n, tickers=None, sections=None: sparse

    assert [c["id"] for c in engine._hybrid_retrieve("q")][0] == "shared"


def test_hybrid_retrieve_deduplicates():
    dense = [{"id": "same", "text": "x", "metadata": {}, "dense_rank": 1}]
    sparse = [{"id": "same", "text": "x", "metadata": {}, "sparse_rank": 1}]
    engine = make_engine()
    engine._dense_retrieve = lambda q, n, tickers=None, sections=None: dense
    engine._sparse_retrieve = lambda q, n, tickers=None, sections=None: sparse
    assert len(engine._hybrid_retrieve("q")) == 1


# ------------------------------------------------------------------- reranking

class StubReranker:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = pairs
        return self.scores


def test_rerank_sorts_by_score_and_truncates():
    chunks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    engine = make_engine(reranker=StubReranker([0.2, 0.9, 0.5]))
    out = engine._rerank("q", chunks, top_k=2)
    assert [c["text"] for c in out] == ["b", "c"]


def test_rerank_handles_tied_scores():
    # Sorting (score, dict) tuples must never compare the dicts themselves.
    chunks = [{"text": "a"}, {"text": "b"}]
    engine = make_engine(reranker=StubReranker([0.5, 0.5]))
    assert len(engine._rerank("q", chunks, top_k=2)) == 2


def test_rerank_on_empty_input():
    assert make_engine()._rerank("q", []) == []


def test_rerank_pairs_question_with_each_chunk():
    reranker = StubReranker([1.0, 2.0])
    engine = make_engine(reranker=reranker)
    engine._rerank("my question", [{"text": "one"}, {"text": "two"}])
    assert reranker.pairs == [("my question", "one"), ("my question", "two")]


# ------------------------------------------------------------------ generation

class StubError(Exception):
    def __init__(self, code):
        super().__init__(f"error {code}")
        self.code = code


class StubModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class StubResponse:
    def __init__(self, text):
        self.text = text


def make_gen_engine(outcomes, monkeypatch):
    # _generate_with_retry catches genai_errors.APIError; point that at StubError
    # so the stubs can exercise the retry path without the real SDK.
    import query_engine
    monkeypatch.setattr(query_engine.genai_errors, "APIError", StubError)

    class Client:
        pass

    client = Client()
    client.models = StubModels(outcomes)
    engine = make_engine(gemini=client)
    return engine, client.models


def test_generate_answer_includes_citable_source_headers():
    engine = make_engine(gemini=object())
    engine._generate_with_retry = lambda prompt, config: StubResponse(prompt)
    chunks = [{"text": "risk text", "metadata": {
        "ticker": "MSFT", "filing_id": "f1", "section": "Item_1A_Risk_Factors"}}]
    prompt = engine._generate_answer("q", chunks)
    assert "Source [1]" in prompt
    assert "Ticker: MSFT" in prompt
    assert "risk text" in prompt


def test_generate_answer_reports_failure_instead_of_returning_none():
    engine = make_engine(gemini=object())
    engine._generate_with_retry = lambda prompt, config: None
    assert "Generation failed" in engine._generate_answer("q", [])


def test_generate_answer_reports_empty_text():
    engine = make_engine(gemini=object())
    engine._generate_with_retry = lambda prompt, config: StubResponse(None)
    assert "no text" in engine._generate_answer("q", [])


def test_generate_answer_without_a_key_explains_itself():
    # A deployment with no GEMINI_API_KEY must still retrieve; only the written
    # answer is skipped, and it should say so rather than look broken.
    engine = make_engine(gemini=None)
    assert not engine.can_generate
    answer = engine._generate_answer("q", [])
    assert "No GEMINI_API_KEY" in answer
    assert "passages below" in answer


def test_can_generate_is_true_with_a_client():
    assert make_engine(gemini=object()).can_generate


def test_retry_recovers_from_transient_503(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    engine, models = make_gen_engine(
        [StubError(503), StubError(429), StubResponse("done")], monkeypatch)
    assert engine._generate_with_retry("p", None).text == "done"
    assert models.calls == 3


def test_retry_gives_up_and_returns_none(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    engine, models = make_gen_engine([StubError(503)] * 5, monkeypatch)
    assert engine._generate_with_retry("p", None, attempts=5) is None
    assert models.calls == 5


def test_retry_does_not_swallow_real_errors(monkeypatch):
    engine, _ = make_gen_engine([StubError(400)], monkeypatch)
    with pytest.raises(StubError):
        engine._generate_with_retry("p", None)


# ------------------------------------------------------------------- end to end

def hit(cid="c1", ticker="MSFT", section="Item_1A_Risk_Factors"):
    return {"id": cid, "text": "t", "dense_rank": 1, "sparse_rank": 1,
            "metadata": {"ticker": ticker, "filing_id": "f", "section": section}}


def test_query_auto_scopes_and_falls_back_when_filter_is_empty():
    engine = make_engine()
    seen = []

    def dense(question, n=20, tickers=None, sections=None):
        seen.append(tickers)
        return [] if tickers else [hit()]

    engine._dense_retrieve = dense
    engine._sparse_retrieve = lambda q, n=20, tickers=None, sections=None: []
    engine._rerank = lambda q, c, top_k=5: c
    engine._generate_answer = lambda q, c: "answer"

    result = engine.query("What are Microsoft's risks?")
    # Scoped to the detected company first, then re-run unfiltered. The widened
    # pass passes [] rather than None; both mean "no company filter".
    assert seen == [["MSFT"], []]
    assert result["tickers"] == ["MSFT"]
    assert result["sources"][0]["ticker"] == "MSFT"


def test_query_respects_an_explicit_empty_scope():
    engine = make_engine()
    seen = []
    engine._dense_retrieve = lambda q, n=20, tickers=None, sections=None: seen.append(tickers) or []
    engine._sparse_retrieve = lambda q, n=20, tickers=None, sections=None: []
    engine._rerank = lambda q, c, top_k=5: []
    engine._generate_answer = lambda q, c: "answer"
    engine.query("Microsoft risks?", tickers=[])
    assert seen == [[]]                       # no auto-detection, no fallback


def test_trace_records_what_the_pipeline_did():
    engine = make_engine()
    engine._dense_retrieve = lambda q, n=20, tickers=None, sections=None: [hit("a"), hit("b")]
    engine._sparse_retrieve = lambda q, n=20, tickers=None, sections=None: [hit("b")]
    engine._rerank = lambda q, c, top_k=5: c[:top_k]

    trace = engine.retrieve("What are Microsoft's risks?", top_k=2)["trace"]
    assert trace["detected_tickers"] == ["MSFT"]
    assert trace["scope_source"] == "detected"
    assert trace["dense_hits"] == 2
    assert trace["sparse_hits"] == 1
    assert trace["fused_candidates"] == 2      # "b" appears in both lists
    assert trace["kept"] == 2
    assert trace["widened_to_full_corpus"] is False


def test_trace_marks_a_manually_chosen_scope():
    engine = make_engine()
    engine._dense_retrieve = lambda q, n=20, tickers=None, sections=None: [hit()]
    engine._sparse_retrieve = lambda q, n=20, tickers=None, sections=None: []
    engine._rerank = lambda q, c, top_k=5: c

    assert engine.retrieve("risks?", tickers=["AAPL"])["trace"]["scope_source"] == "chosen"
    assert engine.retrieve("risks?", tickers=[])["trace"]["scope_source"] == "all"


# ----------------------------------------------------------------------- paths

def test_paths_are_absolute_and_under_the_project_root():
    import os
    for path in (paths.DATA_DIR, paths.PARSED_DIR, paths.EMBEDDINGS_DIR,
                 paths.FILINGS_DIR, paths.TEST_QUESTIONS):
        assert os.path.isabs(path)
        assert path.startswith(paths.PROJECT_ROOT)


def test_every_ticker_has_aliases():
    assert set(paths.TICKER_ALIASES) == set(paths.TICKERS)
    assert all(paths.TICKER_ALIASES[t] for t in paths.TICKERS)
