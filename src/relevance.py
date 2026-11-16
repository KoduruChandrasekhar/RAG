"""Passage-level relevance labels, and the metrics they unlock.

The original metric scored a passage relevant when it came from an expected
company and an expected section. That is a category test: many passages satisfy
it equally, so it cannot tell a good passage from a great one. It therefore
penalised exactly the stages that optimise for things it cannot see -- the
re-ranker (topical fit *within* a section) and deduplication (not repeating
yourself).

These labels judge each passage against each question directly, on a graded
scale, which makes nDCG available -- the metric actually designed to say whether
a ranker puts the best material first.

    0  irrelevant     does not help answer the question
    1  partial        related, or useful background
    2  relevant       directly answers part of the question

Labels are keyed by question text and chunk id, so re-running retrieval or
changing the pipeline does not invalidate them.
"""
import json
import math
import os

import paths

LABELS_PATH = os.path.join(paths.SRC_DIR, "relevance_labels.json")

IRRELEVANT, PARTIAL, RELEVANT = 0, 1, 2

GRADE_NAMES = {IRRELEVANT: "irrelevant", PARTIAL: "partial", RELEVANT: "relevant"}


def load_labels(path=None):
    """{question: {chunk_id: grade}}. Missing file means nothing labelled yet."""
    path = path or LABELS_PATH
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_labels(labels, path=None):
    path = path or LABELS_PATH
    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, sort_keys=True)


def heuristic_grade(source, expected_tickers, expected_sections):
    """The old category test, kept as a fallback for unlabelled passages.

    Graded 2/0 rather than 1/0 so it sits on the same scale as a real label,
    while `coverage` reports how much of a score rests on it.
    """
    if expected_tickers and source["ticker"] not in expected_tickers:
        return IRRELEVANT
    if expected_sections and source["section"] not in expected_sections:
        return IRRELEVANT
    return RELEVANT


def grade(source, item, labels):
    """Grade one retrieved passage for one question.

    A human label wins. Without one, fall back to the category heuristic so a
    partially-labelled set still produces a number.
    """
    question_labels = labels.get(item["question"], {})
    chunk_id = source.get("chunk_id") or source.get("id")
    if chunk_id is not None and chunk_id in question_labels:
        return question_labels[chunk_id], True
    return heuristic_grade(source,
                           item.get("expected_tickers", []),
                           item.get("expected_sections", [])), False


def dcg(grades):
    # Standard discounted cumulative gain: (2^g - 1) / log2(rank + 1).
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(grades, ideal_grades, k):
    """nDCG: how close this ranking is to the best possible one.

    `ideal_grades` is every grade available for the question, so the ideal
    ranking is the best k of them sorted highest-first.
    """
    actual = dcg(grades[:k])
    ideal = dcg(sorted(ideal_grades, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def score_with_labels(result, item, labels, k=5):
    """Score one question's results using labels where they exist."""
    graded, from_label = [], 0
    for source in result["sources"][:k]:
        g, labelled = grade(source, item, labels)
        graded.append(g)
        from_label += labelled

    question_labels = labels.get(item["question"], {})
    # The ideal ranking can only be built from grades we actually know about.
    pool = list(question_labels.values()) or graded

    relevant = [g for g in graded if g >= PARTIAL]
    first = next((i for i, g in enumerate(graded) if g >= PARTIAL), None)

    return {
        "ndcg": ndcg_at_k(graded, pool, k),
        "precision": len(relevant) / len(graded) if graded else 0.0,
        "strict_precision": sum(1 for g in graded if g == RELEVANT) / len(graded)
                            if graded else 0.0,
        "hit": first is not None,
        "rr": 1.0 / (first + 1) if first is not None else 0.0,
        "grades": graded,
        "label_backed": from_label,
        "judged": len(graded),
    }


def coverage(questions, labels):
    """How much of the labelling pool is done."""
    labelled = sum(len(labels.get(q["question"], {})) for q in questions)
    questions_touched = sum(1 for q in questions if labels.get(q["question"]))
    return {
        "labelled_passages": labelled,
        "questions_started": questions_touched,
        "questions_total": len(questions),
    }
