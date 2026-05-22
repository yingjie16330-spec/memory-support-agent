"""Evaluation and groundedness utilities for the support agent."""

from __future__ import annotations

import re
import string
from typing import Any

import numpy as np


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "me",
    "more",
    "my",
    "next",
    "no",
    "not",
    "of",
    "on",
    "or",
    "our",
    "please",
    "should",
    "so",
    "still",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "this",
    "to",
    "too",
    "up",
    "use",
    "was",
    "we",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
    "you",
    "your",
}

PUNCT_TRANSLATION = str.maketrans("", "", string.punctuation)
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")


def _normalize(text: str) -> str:
    return text.lower().translate(PUNCT_TRANSLATION)


def _content_words(text: str) -> list[str]:
    tokens = TOKEN_PATTERN.findall(_normalize(text))
    return [token for token in tokens if token not in STOPWORDS]


def _bigrams(tokens: list[str]) -> list[str]:
    return [f"{tokens[index]} {tokens[index + 1]}" for index in range(len(tokens) - 1)]


def _context_text(
    retrieved_docs: list[dict[str, Any]],
    retrieved_memory: list[dict[str, Any]],
) -> str:
    parts: list[str] = []

    for doc in retrieved_docs:
        parts.append(doc.get("title", ""))
        parts.append(doc.get("text", ""))
        parts.append(" ".join(doc.get("tags", [])))

    for memory in retrieved_memory:
        parts.append(memory.get("text", ""))
        parts.append(" ".join(memory.get("tags", [])))

    return "\n".join(parts)


def groundedness_check(
    answer: str,
    retrieved_docs: list[dict[str, Any]],
    retrieved_memory: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Lightweight heuristic groundedness check.

    This does not prove factual correctness. It only checks whether the answer
    is lexically grounded in the retrieved context and is used as a simple
    hallucination-risk signal.
    """

    context = _context_text(retrieved_docs, retrieved_memory)
    if not context.strip():
        return {"groundedness": "unsupported", "groundedness_score": 0.0}

    answer_tokens = _content_words(answer)
    context_tokens = _content_words(context)

    if not answer_tokens:
        return {"groundedness": "unsupported", "groundedness_score": 0.0}

    answer_units = set(answer_tokens + _bigrams(answer_tokens))
    context_units = set(context_tokens + _bigrams(context_tokens))

    if not answer_units:
        return {"groundedness": "unsupported", "groundedness_score": 0.0}

    matched_units = answer_units.intersection(context_units)
    overlap = len(matched_units) / len(answer_units)
    score = round(float(overlap), 4)

    if overlap >= 0.45:
        label = "supported"
    elif overlap >= 0.20:
        label = "partially_supported"
    else:
        label = "unsupported"

    return {"groundedness": label, "groundedness_score": score}


def _retrieval_metrics(
    predicted_ids: list[str],
    gold_ids: list[str],
    top_k: int,
) -> dict[str, float]:
    """
    Hit@k checks whether at least one gold item appears in top-k results.
    Precision@k measures how many retrieved items are relevant.
    Recall@k measures how many gold items were recovered.
    These are standard and interpretable retrieval metrics for IR evaluation.
    """

    if not gold_ids:
        return {"hit": 0.0, "precision": 0.0, "recall": 0.0}

    gold_set = set(gold_ids)
    predicted_set = predicted_ids[:top_k]
    relevant_count = sum(1 for item in predicted_set if item in gold_set)

    hit = 1.0 if relevant_count > 0 else 0.0
    precision = relevant_count / float(top_k)
    recall = relevant_count / float(len(gold_set))

    return {"hit": hit, "precision": precision, "recall": recall}


def evaluate_agent(
    agent: Any,
    test_set: list[dict[str, Any]],
    top_k: int = 3,
) -> dict[str, Any]:
    """Run the full evaluation loop on the synthetic test set."""

    kb_hits: list[float] = []
    kb_precisions: list[float] = []
    kb_recalls: list[float] = []

    memory_hits: list[float] = []
    memory_precisions: list[float] = []
    memory_recalls: list[float] = []

    escalation_correct: list[float] = []
    groundedness_scores: list[float] = []
    groundedness_counts = {
        "supported": 0,
        "partially_supported": 0,
        "unsupported": 0,
    }

    memory_question_count = 0
    per_question: list[dict[str, Any]] = []

    for example in test_set:
        result = agent.run(user_id=example["user_id"], question=example["question"])

        predicted_doc_ids = [item["doc_id"] for item in result["retrieved_docs"]]
        predicted_memory_ids = [item["memory_id"] for item in result["retrieved_memory"]]

        kb_metrics = _retrieval_metrics(predicted_doc_ids, example["gold_docs"], top_k=top_k)
        kb_hits.append(kb_metrics["hit"])
        kb_precisions.append(kb_metrics["precision"])
        kb_recalls.append(kb_metrics["recall"])

        if example["gold_memory"]:
            memory_question_count += 1
            memory_metrics = _retrieval_metrics(
                predicted_memory_ids,
                example["gold_memory"],
                top_k=top_k,
            )
            memory_hits.append(memory_metrics["hit"])
            memory_precisions.append(memory_metrics["precision"])
            memory_recalls.append(memory_metrics["recall"])

        gold_escalate = bool(example["should_escalate"])
        predicted_escalate = bool(result["should_escalate"])
        escalation_correct.append(1.0 if gold_escalate == predicted_escalate else 0.0)

        groundedness_label = result["groundedness"]
        groundedness_counts[groundedness_label] += 1
        groundedness_scores.append(float(result["groundedness_score"]))

        per_question.append(
            {
                "question_id": example["question_id"],
                "predicted_doc_ids": predicted_doc_ids,
                "predicted_memory_ids": predicted_memory_ids,
                "predicted_escalate": predicted_escalate,
                "groundedness": groundedness_label,
            }
        )

    return {
        "num_questions": len(test_set),
        "kb_hit_at_3": round(float(np.mean(kb_hits)), 4) if kb_hits else 0.0,
        "kb_precision_at_3": round(float(np.mean(kb_precisions)), 4) if kb_precisions else 0.0,
        "kb_recall_at_3": round(float(np.mean(kb_recalls)), 4) if kb_recalls else 0.0,
        "memory_questions": memory_question_count,
        "memory_hit_at_3": round(float(np.mean(memory_hits)), 4) if memory_hits else 0.0,
        "memory_precision_at_3": round(float(np.mean(memory_precisions)), 4) if memory_precisions else 0.0,
        "memory_recall_at_3": round(float(np.mean(memory_recalls)), 4) if memory_recalls else 0.0,
        "escalation_accuracy": round(float(np.mean(escalation_correct)), 4) if escalation_correct else 0.0,
        "groundedness_counts": groundedness_counts,
        "average_groundedness_score": round(float(np.mean(groundedness_scores)), 4)
        if groundedness_scores
        else 0.0,
        "per_question": per_question,
    }
