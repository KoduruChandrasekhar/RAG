import os
import sys
import json
import argparse
import math
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

import paths
from query_engine import FilingScopeEngine
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.run_config import RunConfig


def safe_score(value):
    try:
        if value is None or math.isnan(float(value)):
            return "N/A"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "N/A"


# Roughly how many judge-model calls each metric makes per sample. Used only
# to warn before a run burns through a daily quota.
METRIC_CALL_ESTIMATE = {
    "Faithfulness": 2,
    "Answer Relevancy": 1,
    "Context Precision": 1,
    "Context Recall": 1,
}

FREE_TIER_DAILY_REQUESTS = 20


def estimate_requests(n_questions, metric_names):
    # One answer generation per question, plus judge calls per metric per sample.
    judge = sum(METRIC_CALL_ESTIMATE.get(m, 1) for m in metric_names)
    return n_questions + n_questions * judge


def run_evaluation(label="Stage 4: Hybrid Search + Re-Ranker",
                   limit=None, metric_names=None, top_k=5):
    print(f"\n{'='*55}")
    print(f"RAGAS Evaluation: {label}")
    print(f"{'='*55}\n")

    questions_path = paths.TEST_QUESTIONS
    with open(questions_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    if limit:
        test_data = test_data[:limit]

    selected = metric_names or list(METRIC_CALL_ESTIMATE)
    budget = estimate_requests(len(test_data), selected)

    print(f"Loaded {len(test_data)} test questions.")
    print(f"Metrics: {', '.join(selected)}")
    print(f"Estimated Gemini requests: ~{budget}")
    if budget > FREE_TIER_DAILY_REQUESTS:
        print(f"  WARNING: the free tier allows {FREE_TIER_DAILY_REQUESTS}/day "
              f"for {paths.GEMINI_MODEL}.")
        print("  Narrow it with --limit / --metrics, or use "
              "evaluate_retrieval.py (no LLM, no quota).")
    print()

    engine = FilingScopeEngine()

    # Evaluator LLM (no temperature parameter to eliminate warnings)
    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model=paths.GEMINI_MODEL,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
    )

    # Evaluator Embeddings
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
    )

    # Restricted to 1 worker to stop concurrency bans. The 60s timeout was too
    # short for gemini-3.6-flash's reasoning pass -- every metric except
    # Faithfulness failed with TimeoutError and scored N/A.
    safe_config = RunConfig(timeout=300, max_retries=10, max_workers=1)

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(test_data):
        q = item["question"]
        print(f"Generating answers for Q{i+1}/{len(test_data)}: {q[:60]}...")

        result = engine.query(q, top_k=top_k)

        if not isinstance(result, dict) or "answer" not in result or "sources" not in result:
            print("  ERROR: Invalid query output format. Skipping.")
            continue

        questions.append(q)
        answers.append(result["answer"])
        contexts.append([src["text"] for src in result["sources"]])
        ground_truths.append(item["ground_truth"])

        # Sleep between answer generation to protect the quota
        if i < len(test_data) - 1:
            print("  Sleeping for 15s to respect rate limits...")
            time.sleep(15)

    if not questions:
        print("ERROR: No questions evaluated.")
        return None

    print(f"\nBuilding evaluation dataset from {len(questions)} items...")

    # RAGAS >= 0.2 renamed these columns. The legacy names
    # (question/answer/contexts/ground_truth) are accepted without error but
    # map to all-None fields, so every metric silently scored an empty dataset.
    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths
    })

    # Metrics are run one at a time: a single metric failing (rate limit,
    # timeout) then costs only that metric rather than the whole run.
    # AnswerRelevancy defaults to strictness=3, which asks the model for three
    # candidate generations at once. gemini-3.6-flash rejects that outright:
    #   400 INVALID_ARGUMENT "Multiple candidates is not enabled for this model"
    # so the metric could never score, quota or no quota. strictness=1 asks for
    # one generation, which also costs a third of the requests.
    available = {
        "Faithfulness": Faithfulness,
        "Answer Relevancy": lambda: AnswerRelevancy(strictness=1),
        "Context Precision": ContextPrecision,
        "Context Recall": ContextRecall,
    }
    metrics_to_run = [(name, available[name]()) for name in selected]

    final_scores = {}

    print("\nRunning metric evaluations sequentially to avoid rate-limit deadlocks...")
    for name, metric_obj in metrics_to_run:
        print(f"  -> Scoring {name}...")
        try:
            score_res = evaluate(
                dataset,
                metrics=[metric_obj],
                llm=evaluator_llm,
                embeddings=evaluator_embeddings,
                run_config=safe_config
            )
            # EvaluationResult is not a dict in ragas >= 0.2 (no .keys()).
            # Index it by the metric's own name to get the per-sample scores,
            # then average them.
            per_sample = [v for v in score_res[metric_obj.name] if v is not None]
            mean = sum(per_sample) / len(per_sample) if per_sample else None
            final_scores[name] = safe_score(mean)
        except Exception as e:
            print(f"     Warning: Could not compute {name} ({e})")
            final_scores[name] = "N/A"
        
        # Massive 30-second delay between RAGAS metrics to allow the 20 RPM bucket to refill
        print("  Sleeping 30s before next metric...")
        time.sleep(30)

    print(f"\n{'='*55}")
    print(f"RAGAS Results — {label}")
    print(f"{'='*55}")
    for name, score in final_scores.items():
        print(f"  {name:<20}: {score}")
    print(f"{'='*55}\n")

    return final_scores


def build_parser():
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Score answer quality with RAGAS. Needs Gemini quota; see "
                    "evaluate_retrieval.py for a free alternative.")
    parser.add_argument("-n", "--limit", type=int, metavar="N",
                        help="evaluate only the first N questions")
    parser.add_argument("-m", "--metrics", nargs="+", choices=list(METRIC_CALL_ESTIMATE),
                        metavar="NAME",
                        help="metrics to run (default: all). Choices: "
                             + ", ".join(repr(m) for m in METRIC_CALL_ESTIMATE))
    parser.add_argument("-k", "--top-k", type=int, default=5, metavar="N",
                        help="passages retrieved per question (default 5)")
    parser.add_argument("--label", default="Stage 4: Hybrid Search + Re-Ranker",
                        help="label printed on the results header")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_evaluation(label=args.label, limit=args.limit,
                   metric_names=args.metrics, top_k=args.top_k)