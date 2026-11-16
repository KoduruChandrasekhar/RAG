"""Tests for preflight checks, network retries and quota budgeting."""
import json
import os

import httpx
import pytest

import evaluate
import paths
import query_engine
from query_engine import TRANSIENT_NETWORK_ERRORS, FilingScopeEngine, PipelineNotBuilt

from test_query_engine import make_engine


# ------------------------------------------------------------ preflight checks

def test_preflight_names_the_missing_build_step(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PARSED_DIR", str(tmp_path / "nope"))
    with pytest.raises(PipelineNotBuilt) as exc:
        FilingScopeEngine._check_index_exists()
    assert "parse_documents.py" in str(exc.value)


def test_preflight_rejects_an_empty_parsed_directory(tmp_path, monkeypatch):
    empty = tmp_path / "parsed"
    empty.mkdir()
    monkeypatch.setattr(paths, "PARSED_DIR", str(empty))
    with pytest.raises(PipelineNotBuilt):
        FilingScopeEngine._check_index_exists()


def test_preflight_reports_a_missing_vector_store(tmp_path, monkeypatch):
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    (parsed / "a.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(paths, "PARSED_DIR", str(parsed))
    monkeypatch.setattr(paths, "EMBEDDINGS_DIR", str(tmp_path / "missing"))
    with pytest.raises(PipelineNotBuilt) as exc:
        FilingScopeEngine._check_index_exists()
    assert "build_vector_db.py" in str(exc.value)


def test_preflight_passes_on_a_built_project():
    # The real project is built, so this must not raise.
    FilingScopeEngine._check_index_exists()


# -------------------------------------------------------------- network retries

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
    monkeypatch.setattr(query_engine.genai_errors, "APIError", StubError)

    class Client:
        pass

    client = Client()
    client.models = StubModels(outcomes)
    return make_engine(gemini=client), client.models


def test_transient_errors_cover_connect_and_timeout():
    assert issubclass(httpx.ConnectError, TRANSIENT_NETWORK_ERRORS)
    assert issubclass(httpx.ReadTimeout, TRANSIENT_NETWORK_ERRORS)
    assert issubclass(httpx.ConnectTimeout, TRANSIENT_NETWORK_ERRORS)


def test_retry_recovers_from_a_dropped_connection(monkeypatch):
    # This is the failure that killed a real evaluation run: httpx.ConnectError
    # is not a genai APIError, so it escaped the original retry handler.
    monkeypatch.setattr("time.sleep", lambda s: None)
    engine, models = make_gen_engine(
        [httpx.ConnectError("unreachable"), StubResponse("done")], monkeypatch)
    assert engine._generate_with_retry("p", None).text == "done"
    assert models.calls == 2


def test_retry_gives_up_on_persistent_network_failure(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    engine, models = make_gen_engine(
        [httpx.ConnectError("unreachable")] * 4, monkeypatch)
    assert engine._generate_with_retry("p", None, attempts=4) is None
    assert models.calls == 4


def test_retry_mixes_network_and_api_failures(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    engine, models = make_gen_engine(
        [httpx.ReadTimeout("slow"), StubError(503), StubError(429),
         StubResponse("finally")], monkeypatch)
    assert engine._generate_with_retry("p", None).text == "finally"
    assert models.calls == 4


def test_network_failure_surfaces_as_a_message_not_none(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    engine, _ = make_gen_engine([httpx.ConnectError("x")] * 5, monkeypatch)
    answer = engine._generate_answer("q", [])
    assert "Generation failed" in answer


# ------------------------------------------------------------- quota budgeting

def test_estimate_counts_generation_plus_judge_calls():
    # 2 questions x (1 generation + 1 judge call) = 4
    assert evaluate.estimate_requests(2, ["Answer Relevancy"]) == 4


def test_estimate_scales_with_metrics():
    one = evaluate.estimate_requests(3, ["Context Recall"])
    two = evaluate.estimate_requests(3, ["Context Recall", "Context Precision"])
    assert two > one


def test_estimate_weights_faithfulness_higher():
    # Faithfulness decomposes the answer into claims, so it costs more.
    faith = evaluate.estimate_requests(1, ["Faithfulness"])
    recall = evaluate.estimate_requests(1, ["Context Recall"])
    assert faith > recall


def test_full_run_is_flagged_as_over_the_free_tier():
    everything = evaluate.estimate_requests(11, list(evaluate.METRIC_CALL_ESTIMATE))
    assert everything > evaluate.FREE_TIER_DAILY_REQUESTS


def test_a_narrowed_run_fits_the_free_tier():
    narrowed = evaluate.estimate_requests(2, ["Faithfulness"])
    assert narrowed <= evaluate.FREE_TIER_DAILY_REQUESTS


def test_unknown_metric_names_do_not_crash_the_estimate():
    assert evaluate.estimate_requests(1, ["Not A Metric"]) == 2


# ------------------------------------------------------------- evaluate CLI

def test_evaluate_parser_defaults():
    args = evaluate.build_parser().parse_args([])
    assert args.limit is None
    assert args.metrics is None
    assert args.top_k == 5


def test_evaluate_parser_accepts_limit_and_metrics():
    args = evaluate.build_parser().parse_args(
        ["-n", "2", "-m", "Faithfulness", "Context Recall", "-k", "3"])
    assert args.limit == 2
    assert args.metrics == ["Faithfulness", "Context Recall"]
    assert args.top_k == 3


def test_evaluate_parser_rejects_an_unknown_metric():
    with pytest.raises(SystemExit):
        evaluate.build_parser().parse_args(["-m", "Vibes"])


# --------------------------------------------------------- expanded test set

def test_test_set_exercises_both_sections_per_company():
    data = json.load(open(paths.TEST_QUESTIONS, encoding="utf-8"))
    by_section = {}
    for item in data:
        for section in item["expected_sections"]:
            by_section.setdefault(section, set()).update(item["expected_tickers"])
    assert by_section["Item_1A_Risk_Factors"] == set(paths.TICKERS)
    assert by_section["Item_7_MDA"] == set(paths.TICKERS)


def test_test_set_includes_a_multi_company_question():
    # Single-company questions alone let company scoping hide retrieval faults.
    data = json.load(open(paths.TEST_QUESTIONS, encoding="utf-8"))
    assert any(len(item["expected_tickers"]) > 1 for item in data)


def test_questions_are_unique():
    data = json.load(open(paths.TEST_QUESTIONS, encoding="utf-8"))
    questions = [item["question"] for item in data]
    assert len(questions) == len(set(questions))


# ------------------------------------------------------------- display names

def test_every_ticker_has_a_company_name():
    assert set(paths.COMPANY_NAMES) == set(paths.TICKERS)
    assert all(paths.COMPANY_NAMES[t] for t in paths.TICKERS)


def test_company_name_falls_back_to_the_ticker():
    assert paths.company_name("AAPL") == "Apple"
    assert paths.company_name("TSLA") == "TSLA"


def test_section_labels_cover_both_sections():
    assert set(paths.SECTION_LABELS) == {"Item_1A_Risk_Factors", "Item_7_MDA"}
    assert set(paths.SECTION_LABELS_SHORT) == set(paths.SECTION_LABELS)


def test_section_label_has_a_short_form():
    assert paths.section_label("Item_7_MDA") == "Management's Discussion & Analysis"
    assert paths.section_label("Item_7_MDA", short=True) == "MD&A"


def test_section_label_falls_back_to_the_raw_key():
    assert paths.section_label("Item_9_Unknown") == "Item_9_Unknown"


@pytest.mark.parametrize("accession,expected", [
    ("0000320193-24-000123", 2024),
    ("0001018724-25-000004", 2025),
    ("0001193125-26-323660", 2026),
])
def test_filing_year_reads_the_accession_number(accession, expected):
    assert paths.filing_year(accession) == expected


@pytest.mark.parametrize("bad", ["", "nonsense", "123", "0000320193-XX-000123"])
def test_filing_year_returns_none_when_unparseable(bad):
    assert paths.filing_year(bad) is None


def test_every_real_filing_id_yields_a_year():
    # The UI labels sources with this; a None would show as a bare company name.
    import glob
    for path in glob.glob(os.path.join(paths.PARSED_DIR, "*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert paths.filing_year(data["filing_id"]) is not None, data["filing_id"]


# ------------------------------------------------- duplicate suppression

def passage(cid, text, ticker="AAPL"):
    return {"id": cid, "text": text, "dense_rank": 1, "sparse_rank": 1,
            "metadata": {"ticker": ticker, "filing_id": "f",
                         "section": "Item_1A_Risk_Factors"}}


def test_overlap_scores_identical_text_as_one():
    assert FilingScopeEngine._overlap("alpha beta gamma", "alpha beta gamma") == 1.0


def test_overlap_scores_unrelated_text_near_zero():
    assert FilingScopeEngine._overlap("alpha beta", "gamma delta") == 0.0


def test_overlap_handles_empty_text():
    assert FilingScopeEngine._overlap("", "anything") == 0.0


def test_overlap_is_symmetric():
    a, b = "one two three four", "one two three five"
    assert FilingScopeEngine._overlap(a, b) == FilingScopeEngine._overlap(b, a)


def test_diversify_drops_a_restated_passage():
    # Filings repeat boilerplate verbatim across years.
    boiler = "the company may be affected by adverse economic conditions"
    ranked = [passage("a", boiler),
              passage("b", boiler),
              passage("c", "export controls restrict sales of data centre chips")]
    kept = FilingScopeEngine._diversify(ranked, top_k=2)
    assert [c["id"] for c in kept] == ["a", "c"]


def test_diversify_keeps_rank_order():
    ranked = [passage(str(i), f"unique passage number {i}") for i in range(5)]
    kept = FilingScopeEngine._diversify(ranked, top_k=3)
    assert [c["id"] for c in kept] == ["0", "1", "2"]


def test_diversify_backfills_rather_than_returning_too_few():
    # If everything is a duplicate, returning 1 of 3 requested would be worse
    # than returning a repeat.
    same = "identical boilerplate text repeated verbatim"
    ranked = [passage(str(i), same) for i in range(4)]
    assert len(FilingScopeEngine._diversify(ranked, top_k=3)) == 3


def test_diversify_on_empty_input():
    assert FilingScopeEngine._diversify([], top_k=5) == []


# --------------------------------------------------- comparative balance

def test_balance_reserves_slots_for_each_company():
    # The re-ranker put every NVIDIA passage first; Microsoft must still appear.
    ranked = ([passage(f"n{i}", f"nvidia passage {i}", "NVDA") for i in range(8)]
              + [passage(f"m{i}", f"microsoft passage {i}", "MSFT") for i in range(3)])
    kept = FilingScopeEngine._balance_by_company(ranked, 4, ["MSFT", "NVDA"])
    tickers = [c["metadata"]["ticker"] for c in kept]
    assert tickers.count("MSFT") == 2
    assert tickers.count("NVDA") == 2


def test_balance_falls_back_to_plain_diversity_for_one_company():
    ranked = [passage(str(i), f"passage {i}") for i in range(6)]
    kept = FilingScopeEngine._balance_by_company(ranked, 3, ["AAPL"])
    assert [c["id"] for c in kept] == ["0", "1", "2"]


def test_balance_returns_results_in_rank_order():
    ranked = ([passage(f"n{i}", f"nvidia {i}", "NVDA") for i in range(4)]
              + [passage(f"m{i}", f"microsoft {i}", "MSFT") for i in range(4)])
    kept = FilingScopeEngine._balance_by_company(ranked, 4, ["MSFT", "NVDA"])
    order = {id(c): i for i, c in enumerate(ranked)}
    positions = [order[id(c)] for c in kept]
    assert positions == sorted(positions)


def test_balance_uses_spare_slots_when_a_company_is_thin():
    ranked = ([passage(f"n{i}", f"nvidia {i}", "NVDA") for i in range(6)]
              + [passage("m0", "the only microsoft passage", "MSFT")])
    kept = FilingScopeEngine._balance_by_company(ranked, 5, ["MSFT", "NVDA"])
    assert len(kept) == 5
    assert sum(1 for c in kept if c["metadata"]["ticker"] == "MSFT") == 1


def test_balance_still_removes_duplicates():
    same = "identical restated risk language"
    ranked = [passage("n0", same, "NVDA"), passage("n1", same, "NVDA"),
              passage("m0", "distinct microsoft wording", "MSFT"),
              passage("n2", "distinct nvidia wording", "NVDA")]
    kept = FilingScopeEngine._balance_by_company(ranked, 3, ["MSFT", "NVDA"])
    texts = [c["text"] for c in kept]
    assert texts.count(same) <= 1


# ----------------------------------------------- per-company gathering

def test_gather_searches_each_company_separately():
    engine = make_engine()
    seen = []

    def dense(q, n=20, tickers=None, sections=None):
        seen.append(tickers)
        return [passage(f"{tickers[0]}", "text", tickers[0])]

    engine._dense_retrieve = dense
    engine._sparse_retrieve = lambda q, n=20, tickers=None, sections=None: []
    candidates, dense_n, sparse_n = engine._gather(
        "q", ["MSFT", "NVDA"], None, 20)
    assert seen == [["MSFT"], ["NVDA"]]      # one search per company
    assert dense_n == 2


def test_gather_pools_for_a_single_company():
    engine = make_engine()
    seen = []
    engine._dense_retrieve = lambda q, n=20, tickers=None, sections=None: seen.append(tickers) or []
    engine._sparse_retrieve = lambda q, n=20, tickers=None, sections=None: []
    engine._gather("q", ["MSFT"], None, 20)
    assert seen == [["MSFT"]]                # no per-company loop needed


# ----------------------------------------------------------- ablation

def test_ablation_configurations_are_distinct_and_named():
    import ablation
    labels = [label for label, _, _ in ablation.CONFIGURATIONS]
    assert len(labels) == len(set(labels))
    assert any("Dense only" in l for l in labels)
    assert any("scoping OFF" in l for l in labels)


def test_ablation_runners_respect_top_k():
    import ablation
    engine = make_engine()
    chunks = [{"text": f"p{i}", "metadata": {"ticker": "AAPL", "filing_id": "f",
                                             "section": "Item_7_MDA"}}
              for i in range(10)]
    engine._dense_retrieve = lambda q, n=20, tickers=None, sections=None: chunks
    assert len(ablation.dense_only(engine, "q", ["AAPL"], 3)["sources"]) == 3


def test_ablation_scope_override_disables_detection():
    # The "scoping OFF" row must pass [] rather than fall back to detection.
    import ablation
    overrides = {label: override for label, _, override in ablation.CONFIGURATIONS}
    assert overrides["Full, company scoping OFF"] == []
    assert overrides["Full pipeline"] is None


# -------------------------------------------- company detection boundaries

@pytest.mark.parametrize("question,expected", [
    # Substring matching used to fire on these: "l-aws", "mac-hine", "mac-ro".
    ("How do export control laws affect chip sales?", []),
    ("What does NVIDIA say about machine learning?", ["NVDA"]),
    ("Microsoft and macroeconomic conditions", ["MSFT"]),
    ("Does Alphabet discuss macro headwinds?", ["GOOGL"]),
    # Real mentions must still match.
    ("How did AWS revenue change?", ["AMZN"]),
    ("What risks does the Mac line face?", ["AAPL"]),
    ("Does CUDA face competition?", ["NVDA"]),
])
def test_detection_respects_word_boundaries(question, expected):
    engine = make_engine()
    assert sorted(engine._detect_tickers(question)) == sorted(expected)


def test_detection_is_not_fooled_by_punctuation():
    engine = make_engine()
    assert engine._detect_tickers("What about Apple's margins?") == ["AAPL"]
    assert engine._detect_tickers("(NVIDIA) export controls") == ["NVDA"]


# ------------------------------------------------- expanded labelled set

def test_labelled_set_is_large_enough_to_discriminate():
    # Hit Rate and MRR saturated at 1.000 on the original 11 questions, which
    # made them useless for comparing configurations.
    data = json.load(open(paths.TEST_QUESTIONS, encoding="utf-8"))
    assert len(data) >= 40


def test_labelled_set_covers_each_company_several_times():
    import collections
    data = json.load(open(paths.TEST_QUESTIONS, encoding="utf-8"))
    counts = collections.Counter(t for i in data for t in i["expected_tickers"])
    assert set(counts) == set(paths.TICKERS)
    assert min(counts.values()) >= 5


def test_labelled_set_has_several_comparative_questions():
    data = json.load(open(paths.TEST_QUESTIONS, encoding="utf-8"))
    assert sum(1 for i in data if len(i["expected_tickers"]) > 1) >= 3


# ------------------------------------------- UI / engine trace contract

def test_trace_supplies_every_key_the_ui_reads():
    """The analysis panel reads these by name; a rename would break it silently.

    app.py is not imported here (it needs streamlit at module scope), so the
    keys are scraped from its source instead.
    """
    import re as _re
    app_source = open(os.path.join(paths.SRC_DIR, "app.py"), encoding="utf-8").read()
    used = set(_re.findall(r"""trace(?:\.get)?\(?\[?['"]([a-z_]+)['"]""", app_source))
    assert used, "no trace keys found in app.py -- did the panel change?"

    engine = make_engine()
    engine._dense_retrieve = lambda q, n=20, tickers=None, sections=None: [
        {"id": "a", "text": "x", "dense_rank": 1,
         "metadata": {"ticker": "AAPL", "filing_id": "f", "section": "s"}}]
    engine._sparse_retrieve = lambda q, n=20, tickers=None, sections=None: []
    engine._rerank = lambda q, c, top_k=5: c[:top_k]

    emitted = set(engine.retrieve("What risks did Apple flag?", top_k=2)["trace"])
    missing = used - emitted
    assert not missing, f"app.py reads trace keys the engine never sets: {missing}"


def test_answer_relevancy_asks_for_one_candidate():
    """gemini-3.6-flash rejects multi-candidate requests with a 400.

    RAGAS defaults AnswerRelevancy to strictness=3, which asks for three
    generations at once; that made the metric unscoreable against this model
    regardless of quota.
    """
    import evaluate as ev
    import inspect
    source = inspect.getsource(ev.run_evaluation)
    assert "AnswerRelevancy(strictness=1)" in source


def test_metric_estimate_covers_every_selectable_metric():
    import evaluate as ev
    parser_choices = set(ev.METRIC_CALL_ESTIMATE)
    assert parser_choices == {"Faithfulness", "Answer Relevancy",
                              "Context Precision", "Context Recall"}


# ------------------------------------------- passage-level relevance labels

def test_grades_are_ordered_and_named():
    import relevance as rel
    assert rel.IRRELEVANT < rel.PARTIAL < rel.RELEVANT
    assert set(rel.GRADE_NAMES) == {0, 1, 2}


def test_a_human_label_overrides_the_heuristic():
    import relevance as rel
    item = {"question": "q", "expected_tickers": ["AAPL"],
            "expected_sections": ["Item_7_MDA"]}
    source = {"chunk_id": "c1", "ticker": "AAPL", "section": "Item_7_MDA"}
    # heuristic alone would call this fully relevant
    assert rel.grade(source, item, {}) == (rel.RELEVANT, False)
    # a human who read it says otherwise, and wins
    labels = {"q": {"c1": rel.IRRELEVANT}}
    assert rel.grade(source, item, labels) == (rel.IRRELEVANT, True)


def test_unlabelled_passages_fall_back_and_are_flagged():
    import relevance as rel
    item = {"question": "q", "expected_tickers": ["MSFT"],
            "expected_sections": ["Item_1A_Risk_Factors"]}
    source = {"chunk_id": "unseen", "ticker": "MSFT",
              "section": "Item_1A_Risk_Factors"}
    grade, was_labelled = rel.grade(source, item, {"q": {"other": 2}})
    assert grade == rel.RELEVANT
    assert was_labelled is False


def test_ndcg_rewards_putting_the_best_passage_first():
    import relevance as rel
    ideal = [2, 1, 0]
    assert rel.ndcg_at_k([2, 1, 0], ideal, 3) == 1.0
    assert rel.ndcg_at_k([0, 1, 2], ideal, 3) < 1.0


def test_ndcg_separates_rankings_precision_cannot():
    # Both contain the same passages, so precision is identical; only the
    # order differs. This is exactly what the old metric could not see.
    import relevance as rel
    ideal = [2, 2, 1, 1, 0]
    good = rel.ndcg_at_k([2, 2, 1, 1, 0], ideal, 5)
    bad = rel.ndcg_at_k([0, 1, 1, 2, 2], ideal, 5)
    assert good > bad


def test_ndcg_is_zero_when_nothing_is_relevant():
    import relevance as rel
    assert rel.ndcg_at_k([0, 0, 0], [0, 0, 0], 3) == 0.0


def test_score_with_labels_counts_label_coverage():
    import relevance as rel
    item = {"question": "q", "expected_tickers": ["AAPL"],
            "expected_sections": ["Item_7_MDA"]}
    result = {"tickers": ["AAPL"], "sources": [
        {"chunk_id": "a", "ticker": "AAPL", "section": "Item_7_MDA", "text": "x"},
        {"chunk_id": "b", "ticker": "AAPL", "section": "Item_7_MDA", "text": "y"}]}
    score = rel.score_with_labels(result, item, {"q": {"a": 2}}, k=2)
    assert score["label_backed"] == 1
    assert score["judged"] == 2


def test_strict_precision_ignores_partial_grades():
    import relevance as rel
    item = {"question": "q", "expected_tickers": [], "expected_sections": []}
    result = {"tickers": [], "sources": [
        {"chunk_id": "a", "ticker": "AAPL", "section": "s", "text": "x"},
        {"chunk_id": "b", "ticker": "AAPL", "section": "s", "text": "y"}]}
    labels = {"q": {"a": rel.RELEVANT, "b": rel.PARTIAL}}
    score = rel.score_with_labels(result, item, labels, k=2)
    assert score["precision"] == 1.0          # both count as relevant
    assert score["strict_precision"] == 0.5   # only one fully answers


def test_coverage_reports_progress():
    import relevance as rel
    questions = [{"question": "a"}, {"question": "b"}, {"question": "c"}]
    cov = rel.coverage(questions, {"a": {"1": 2, "2": 0}})
    assert cov["labelled_passages"] == 2
    assert cov["questions_started"] == 1
    assert cov["questions_total"] == 3


def test_labels_round_trip(tmp_path):
    import relevance as rel
    path = str(tmp_path / "labels.json")
    assert rel.load_labels(path) == {}          # missing file is not an error
    rel.save_labels({"q": {"c1": 2}}, path)
    assert rel.load_labels(path) == {"q": {"c1": 2}}


def test_retrieval_sources_carry_a_stable_chunk_id():
    # Labels key on chunk_id; without it they could not be stored at all.
    engine = make_engine()
    engine._dense_retrieve = lambda q, n=20, tickers=None, sections=None: [
        {"id": "AAPL_f_Item_7_MDA_3", "text": "x", "dense_rank": 1,
         "metadata": {"ticker": "AAPL", "filing_id": "f", "section": "Item_7_MDA"}}]
    engine._sparse_retrieve = lambda q, n=20, tickers=None, sections=None: []
    engine._rerank = lambda q, c, top_k=5: c[:top_k]
    source = engine.retrieve("Apple margins?", top_k=1)["sources"][0]
    assert source["chunk_id"] == "AAPL_f_Item_7_MDA_3"
