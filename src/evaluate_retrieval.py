"""Retrieval-only evaluation: no LLM, so it costs no API quota.

RAGAS needs a judge model, and the Gemini free tier allows 20 generate_content
requests per day -- far less than a full RAGAS run. This measures the retrieval
half of the pipeline instead, which is where answer quality is actually won or
lost, and it can be run as often as you like.

Each question in test_questions.json carries expected_tickers and
expected_sections. A retrieved chunk counts as relevant when it comes from an
expected ticker and an expected section.

Metrics, all over the top-k chunks:
  Hit Rate      fraction of questions with at least one relevant chunk
  MRR           1 / rank of the first relevant chunk, averaged
  Precision@k   fraction of retrieved chunks that are relevant, averaged
  Ticker Acc.   fraction of questions where company detection was exactly right
"""
import json
import sys

import paths
import relevance
from query_engine import FilingScopeEngine


def is_relevant(source, expected_tickers, expected_sections):
    if expected_tickers and source["ticker"] not in expected_tickers:
        return False
    if expected_sections and source["section"] not in expected_sections:
        return False
    return True


def score_question(result, item, labels=None):
    """Score one question's retrieval.

    With passage-level labels this grades each passage directly and reports
    nDCG. Without them it falls back to the old category test -- expected
    company and expected section -- which cannot tell a good passage from a
    great one.
    """
    if labels:
        return relevance.score_with_labels(result, item, labels,
                                           k=len(result["sources"]) or 5)

    tickers = item.get("expected_tickers", [])
    sections = item.get("expected_sections", [])
    flags = [is_relevant(s, tickers, sections) for s in result["sources"]]

    hit = any(flags)
    rr = 1.0 / (flags.index(True) + 1) if hit else 0.0
    precision = sum(flags) / len(flags) if flags else 0.0
    ticker_ok = sorted(result["tickers"]) == sorted(tickers)

    return {"hit": hit, "rr": rr, "precision": precision,
            "ticker_ok": ticker_ok, "flags": flags,
            "ndcg": None, "label_backed": 0, "judged": len(flags)}


def run_evaluation(top_k=5, questions_path=None):
    print(f"\n{'=' * 60}")
    print(f"Retrieval Evaluation (top_k={top_k}, no LLM calls)")
    print(f"{'=' * 60}\n")

    with open(questions_path or paths.TEST_QUESTIONS, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    labelled = [i for i in test_data if i.get("expected_tickers")
                or i.get("expected_sections")]
    if not labelled:
        print("ERROR: no questions carry expected_tickers / expected_sections.")
        return None

    skipped = len(test_data) - len(labelled)
    print(f"Loaded {len(labelled)} labelled questions"
          f"{f' ({skipped} unlabelled, skipped)' if skipped else ''}.\n")

    engine = FilingScopeEngine()
    scores = []

    for i, item in enumerate(labelled, 1):
        question = item["question"]
        result = engine.retrieve(question, top_k=top_k)
        score = score_question(result, item)
        scores.append(score)

        marks = "".join("#" if f else "." for f in score["flags"])
        print(f"Q{i}: {question[:52]:<52}")
        print(f"    relevant: [{marks}]  hit={'Y' if score['hit'] else 'N'}  "
              f"RR={score['rr']:.2f}  P@{top_k}={score['precision']:.2f}  "
              f"scope={','.join(result['tickers']) or '-'}"
              f"{'' if score['ticker_ok'] else ' (expected ' + ','.join(item.get('expected_tickers', [])) + ')'}")

    n = len(scores)
    summary = {
        "Hit Rate": sum(s["hit"] for s in scores) / n,
        "MRR": sum(s["rr"] for s in scores) / n,
        f"Precision@{top_k}": sum(s["precision"] for s in scores) / n,
        "Ticker Accuracy": sum(s["ticker_ok"] for s in scores) / n,
    }

    print(f"\n{'=' * 60}")
    print(f"Results over {n} questions")
    print(f"{'=' * 60}")
    for name, value in summary.items():
        bar = "#" * int(round(value * 20))
        print(f"  {name:<18}: {value:.3f}  {bar}")
    print(f"{'=' * 60}\n")

    return summary


if __name__ == "__main__":
    top_k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run_evaluation(top_k=top_k)
