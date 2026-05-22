from __future__ import annotations

from src.agent import SupportAgent
from src.config import Config
from src.data_loader import load_knowledge_base, load_test_set, load_user_memory
from src.evaluator import evaluate_agent, groundedness_check
from src.retriever import BM25Retriever


def test_data_files_load() -> None:
    kb = load_knowledge_base()
    memory = load_user_memory()
    test_set = load_test_set()

    assert len(kb) >= 15
    assert len(memory) >= 8
    assert len(test_set) == 20


def test_kb_bm25_retrieval_returns_top_k() -> None:
    kb = load_knowledge_base()
    retriever = BM25Retriever(
        kb,
        lambda record: f"{record['title']} {record['text']} {' '.join(record['tags'])}",
    )

    results = retriever.search("API key 401 authentication error", top_k=3)

    assert len(results) == 3
    assert results[0].record["doc_id"].startswith("doc_")


def test_memory_bm25_retrieval_for_specific_user() -> None:
    memory = [item for item in load_user_memory() if item["user_id"] == "user_001"]
    retriever = BM25Retriever(
        memory,
        lambda record: f"{record['text']} {' '.join(record['tags'])}",
    )

    results = retriever.search("I still get 401 with my API key", top_k=3)

    assert len(results) >= 1
    assert results[0].record["user_id"] == "user_001"


def test_evaluation_runs_without_crashing() -> None:
    agent = SupportAgent(Config(openai_api_key=""))
    summary = evaluate_agent(agent, load_test_set())

    assert summary["num_questions"] == 20
    assert "kb_hit_at_3" in summary
    assert "escalation_accuracy" in summary


def test_agent_fallback_mode_without_api_key() -> None:
    agent = SupportAgent(Config(openai_api_key=""))
    result = agent.run("user_001", "I still cannot use my API key. What should I do?")

    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


def test_agent_result_contains_action_trace() -> None:
    agent = SupportAgent(Config(openai_api_key=""))
    result = agent.run("user_006", "Where can I download my invoice PDF?")

    assert "action_trace" in result
    assert result["action_trace"][0] == "TOOL_CALL: search_kb"


def test_groundedness_checker_returns_label_and_score() -> None:
    docs = load_knowledge_base()[:1]
    memories = load_user_memory()[:1]
    answer = "Users can reset their API key in account settings and old keys stop working."

    result = groundedness_check(answer, docs, memories)

    assert "groundedness" in result
    assert "groundedness_score" in result
    assert isinstance(result["groundedness_score"], float)
