"""Ablation study: does each stage of the pipeline actually earn its place?

Every stage added to a RAG pipeline is a claim that it improves retrieval. This
measures those claims by turning stages off one at a time and re-scoring the
same labelled questions.

No LLM is involved, so this costs no API quota and can be re-run freely --
including after a change, to check it helped rather than assuming it did.

    python src/ablation.py
    python src/ablation.py --top-k 3
"""
import argparse
import json

import paths
import relevance
from evaluate_retrieval import score_question
from query_engine import FilingScopeEngine

N_EACH = 20


def _sources(chunks, top_k):
    return [{"chunk_id": c.get("id"),
             "ticker": c["metadata"]["ticker"],
             "filing_id": c["metadata"]["filing_id"],
             "section": c["metadata"]["section"],
             "text": c["text"]} for c in chunks[:top_k]]


def dense_only(engine, question, tickers, top_k):
    chunks = engine._dense_retrieve(question, n=N_EACH, tickers=tickers)
    return {"tickers": tickers, "sources": _sources(chunks, top_k)}


def sparse_only(engine, question, tickers, top_k):
    chunks = engine._sparse_retrieve(question, n=N_EACH, tickers=tickers)
    return {"tickers": tickers, "sources": _sources(chunks, top_k)}


def hybrid_no_rerank(engine, question, tickers, top_k):
    candidates, _, _ = engine._gather(question, tickers, None, N_EACH)
    return {"tickers": tickers, "sources": _sources(candidates, top_k)}


def hybrid_reranked(engine, question, tickers, top_k):
    candidates, _, _ = engine._gather(question, tickers, None, N_EACH)
    ranked = engine._rerank(question, candidates, top_k=max(top_k * 4, top_k))
    return {"tickers": tickers, "sources": _sources(ranked, top_k)}


def full(engine, question, tickers, top_k):
    return engine.retrieve(question, tickers=tickers, top_k=top_k,
                           diversify=True)


# (label, runner, scope_override) -- scope_override of [] disables company
# scoping so its contribution can be measured too.
CONFIGURATIONS = [
    ("Dense only (no BM25, no re-rank)", dense_only, None),
    ("Sparse only (BM25, no re-rank)", sparse_only, None),
    ("Hybrid, no re-ranker", hybrid_no_rerank, None),
    ("Hybrid + re-ranker", hybrid_reranked, None),
    ("Full pipeline", full, None),
    ("Full, company scoping OFF", full, []),
]


def run(top_k=5):
    with open(paths.TEST_QUESTIONS, "r", encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\n{'=' * 74}")
    print(f"Ablation over {len(questions)} questions (top_k={top_k}, no LLM calls)")
    print(f"{'=' * 74}\n")

    engine = FilingScopeEngine()

    labels = relevance.load_labels()
    cov = relevance.coverage(questions, labels)
    if labels:
        print(f"Passage-level labels: {cov['labelled_passages']} judged across "
              f"{cov['questions_started']}/{cov['questions_total']} questions.")
        print("Unlabelled passages fall back to the company+section test.\n")
    else:
        print("No passage labels found, so relevance means 'expected company")
        print("and expected section'. That test cannot separate a good passage")
        print("from a great one. Build labels with build_label_pool.py.\n")

    ndcg_header = f"{'nDCG@' + str(top_k):>8}" if labels else ""
    width = 74 + (8 if labels else 0)
    print(f"\n{'configuration':<36} {'Hit':>6} {'MRR':>6} {f'P@{top_k}':>6}{ndcg_header}")
    print("-" * width)

    results = []
    for label, runner, override in CONFIGURATIONS:
        hit = rr = precision = ndcg = 0.0
        for item in questions:
            tickers = override if override is not None \
                else engine._detect_tickers(item["question"])
            result = runner(engine, item["question"], tickers, top_k)
            score = score_question(result, item, labels)
            hit += score["hit"]
            rr += score["rr"]
            precision += score["precision"]
            ndcg += score.get("ndcg") or 0.0

        n = len(questions)
        row = (label, hit / n, rr / n, precision / n, ndcg / n)
        results.append(row)
        tail = f"{row[4]:>8.3f}" if labels else ""
        print(f"{label:<36} {row[1]:>6.3f} {row[2]:>6.3f} {row[3]:>6.3f}{tail}")

    baseline = next(r for r in results if r[0] == "Dense only (no BM25, no re-rank)")
    best = max(results, key=lambda r: ((r[4] if labels else r[3]), r[2]))
    print("-" * width)
    print(f"Dense-only baseline P@{top_k}: {baseline[3]:.3f}")
    print(f"Best configuration        : {best[0]} ({best[3]:.3f})")
    print(f"Gain from the full stack  : {best[3] - baseline[3]:+.3f}")

    n = len(questions)
    noise = 1 / (n * top_k)
    print()
    print("Read these with two caveats:")
    print()
    print(f"  Sample size. {n} questions means one passage changing place moves")
    print(f"  P@{top_k} by about {noise:.3f}. Differences under ~0.05 are noise.")
    print()
    print("  What counts as relevant. A passage scores only if it comes from an")
    print("  expected company AND an expected section. That favours stages which")
    print("  preserve section membership and penalises the re-ranker, whose job")
    print("  is topical fit -- it can surface a genuinely better passage from the")
    print("  other section and be marked wrong for it. Treat a drop here as a")
    print("  prompt to read the passages, not proof the stage is useless.")
    print()

    return results


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ablation",
        description="Measure what each retrieval stage contributes.")
    parser.add_argument("-k", "--top-k", type=int, default=5, metavar="N",
                        help="passages kept per question (default 5)")
    return parser


if __name__ == "__main__":
    run(top_k=build_parser().parse_args().top_k)
