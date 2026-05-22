"""Command line interface for the memory-based support agent project."""

from __future__ import annotations

import argparse
import json
from textwrap import dedent

from src.agent import SupportAgent
from src.data_loader import load_test_set
from src.evaluator import evaluate_agent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Memory-based customer support agent with retrieval evaluation."
    )
    parser.add_argument("--user_id", help="Synthetic user ID for a demo question.")
    parser.add_argument("--question", help="Support question to answer.")
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation on the synthetic test set.",
    )
    parser.add_argument(
        "--list-examples",
        action="store_true",
        help="Print example commands.",
    )
    return parser


def _print_examples() -> None:
    print("Example commands:")
    print("python -m src.cli --list-examples")
    print('python -m src.cli --user_id user_001 --question "I still cannot use my API key. What should I do?"')
    print("python -m src.cli --user_id user_006 --question \"Where can I download my invoice PDF?\"")
    print("python -m src.cli --evaluate")


def _print_run_result(result: dict) -> None:
    print(f"User ID: {result['user_id']}")
    print(f"Question: {result['question']}")
    print("")
    print("Action Trace:")
    for step in result["action_trace"]:
        print(f"- {step}")

    print("")
    print("Retrieved Knowledge Base Documents:")
    for doc in result["retrieved_docs"]:
        print(f"- {doc['doc_id']} | {doc['title']} | raw_score={doc['score']}")

    print("")
    print("Retrieved Memory Records:")
    if result["retrieved_memory"]:
        for memory in result["retrieved_memory"]:
            print(f"- {memory['memory_id']} | raw_score={memory['score']} | {memory['text']}")
    else:
        print("- No memory records found for this user.")

    print("")
    print("Final Answer:")
    print(result["answer"])

    print("")
    print(f"Groundedness: {result['groundedness']}")
    print(f"Groundedness Score: {result['groundedness_score']}")
    print(f"Should Escalate: {result['should_escalate']}")
    print(f"Escalation Reason: {result['escalation_reason']}")


def _print_evaluation_summary(summary: dict) -> None:
    print("Evaluation Summary")
    print("==================")
    print(f"Number of test questions: {summary['num_questions']}")
    print(f"Knowledge Base Hit@3: {summary['kb_hit_at_3']}")
    print(f"Knowledge Base Precision@3: {summary['kb_precision_at_3']}")
    print(f"Knowledge Base Recall@3: {summary['kb_recall_at_3']}")
    print(f"Memory questions with gold labels: {summary['memory_questions']}")
    print(f"Memory Hit@3: {summary['memory_hit_at_3']}")
    print(f"Memory Precision@3: {summary['memory_precision_at_3']}")
    print(f"Memory Recall@3: {summary['memory_recall_at_3']}")
    print(f"Escalation Accuracy: {summary['escalation_accuracy']}")
    print(f"Groundedness Counts: {json.dumps(summary['groundedness_counts'])}")
    print(f"Average Groundedness Score: {summary['average_groundedness_score']}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_examples:
        _print_examples()
        return

    agent = SupportAgent()

    if args.evaluate:
        summary = evaluate_agent(agent, load_test_set())
        _print_evaluation_summary(summary)
        return

    if args.user_id and args.question:
        result = agent.run(user_id=args.user_id, question=args.question)
        _print_run_result(result)
        return

    print(
        dedent(
            """
            Please provide one of the supported command patterns:
            - python -m src.cli --list-examples
            - python -m src.cli --user_id user_001 --question "Your question here"
            - python -m src.cli --evaluate
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
