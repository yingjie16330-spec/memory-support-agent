"""Support agent implementation using LLM tool calling.

The agent exposes two retrieval tools (knowledge base search and user-memory
search) to the LLM. The LLM decides which tool(s) to call and when, based on
the user's question. This is a genuine tool-calling agent, not a fixed
retrieve-then-generate pipeline.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from openai import OpenAI

from src.config import Config, get_config
from src.data_loader import load_knowledge_base, load_user_memory
from src.evaluator import groundedness_check
from src.retriever import BM25Retriever


MAX_ITERATIONS = 5


class SupportAgent:
    """Tool-calling retrieval agent for a SaaS support demo."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or get_config()
        self.knowledge_base = load_knowledge_base()
        self.user_memory = load_user_memory()
        self.kb_retriever = BM25Retriever(self.knowledge_base, self._kb_text)
        self.memory_by_user = self._group_memory_by_user(self.user_memory)
        self.memory_retrievers = {
            user_id: BM25Retriever(records, self._memory_text)
            for user_id, records in self.memory_by_user.items()
        }
        self.client = self._build_client()

    @staticmethod
    def _kb_text(record: dict[str, Any]) -> str:
        return " ".join(
            [
                record.get("title", ""),
                record.get("text", ""),
                " ".join(record.get("tags", [])),
            ]
        )

    @staticmethod
    def _memory_text(record: dict[str, Any]) -> str:
        return " ".join(
            [
                record.get("text", ""),
                " ".join(record.get("tags", [])),
            ]
        )

    @staticmethod
    def _group_memory_by_user(
        records: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[record["user_id"]].append(record)
        return dict(grouped)

    def _build_client(self) -> OpenAI | None:
        if not self.config.openai_api_key:
            return None
        return OpenAI(
            api_key=self.config.openai_api_key,
            base_url=self.config.openai_base_url,
        )

    def retrieve_knowledge(self, question: str) -> list[dict[str, Any]]:
        results = self.kb_retriever.search(question, top_k=self.config.kb_top_k)
        return [self._format_doc_result(item.record, item.score) for item in results]

    def retrieve_memory(self, user_id: str, question: str) -> list[dict[str, Any]]:
        retriever = self.memory_retrievers.get(user_id)
        if not retriever:
            return []
        results = retriever.search(question, top_k=self.config.memory_top_k)
        return [self._format_memory_result(item.record, item.score) for item in results]

    @staticmethod
    def _tool_schemas() -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "description": (
                        "Search the CloudBox AI product documentation for "
                        "information relevant to the user's question. Use this "
                        "to find facts about features, policies, and "
                        "troubleshooting steps."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for the product docs.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_user_memory",
                    "description": (
                        "Search the current user's past conversation history to "
                        "find what they have already tried or asked before. Use "
                        "this to avoid repeating advice the user already followed."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query for the user's history.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def run(self, user_id: str, question: str) -> dict[str, Any]:
        action_trace: list[str] = []
        retrieved_docs: list[dict[str, Any]] = []
        retrieved_memory: list[dict[str, Any]] = []

        if not self.client:
            return self._demo_mode_result(user_id, question)

        system_prompt = (
            "You are a helpful SaaS customer support assistant for CloudBox AI. "
            "You have two tools: search_knowledge_base (product docs) and "
            "search_user_memory (this user's past history). "
            "Always ground your answer in retrieved information. Decide which "
            "tools to call based on the question. Check the user's memory before "
            "suggesting a step, so you do not repeat something they already tried. "
            "Do not invent features, policies, or guarantees. "
            "If the retrieved context is not enough to answer safely, say that "
            "the case should be escalated to human support."
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User ID: {user_id}\nQuestion: {question}"},
        ]

        final_answer = ""
        for _ in range(MAX_ITERATIONS):
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                tools=self._tool_schemas(),
                temperature=0.2,
            )
            message = response.choices[0].message

            if not message.tool_calls:
                final_answer = (message.content or "").strip()
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                }
            )

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                query = args.get("query", question)

                if name == "search_knowledge_base":
                    docs = self.retrieve_knowledge(query)
                    retrieved_docs = docs
                    action_trace.append(f"LLM called search_knowledge_base(query={query!r})")
                    tool_result = self._docs_to_text(docs)
                elif name == "search_user_memory":
                    mems = self.retrieve_memory(user_id, query)
                    retrieved_memory = mems
                    action_trace.append(f"LLM called search_user_memory(query={query!r})")
                    tool_result = self._memory_to_text(mems)
                else:
                    tool_result = "Unknown tool."
                    action_trace.append(f"LLM called unknown tool {name!r}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result,
                    }
                )
        else:
            action_trace.append("Reached max tool-calling iterations.")
            final_answer = (
                "I could not finish reasoning about this within the allowed steps. "
                "This case should be escalated to human support."
            )

        groundedness = groundedness_check(final_answer, retrieved_docs, retrieved_memory)
        should_escalate, escalation_reason = self.decide_escalation(
            question=question,
            retrieved_docs=retrieved_docs,
            final_answer=final_answer,
            groundedness=groundedness,
        )

        action_trace.append(
            "Decision: escalate to human support"
            if should_escalate
            else "Decision: answer the user directly"
        )

        return {
            "question": question,
            "user_id": user_id,
            "action_trace": action_trace,
            "retrieved_docs": retrieved_docs,
            "retrieved_memory": retrieved_memory,
            "answer": final_answer,
            "groundedness": groundedness["groundedness"],
            "groundedness_score": groundedness["groundedness_score"],
            "should_escalate": should_escalate,
            "escalation_reason": escalation_reason,
        }

    def decide_escalation(
        self,
        question: str,
        retrieved_docs: list[dict[str, Any]],
        final_answer: str,
        groundedness: dict[str, Any],
    ) -> tuple[bool, str]:
        if not retrieved_docs:
            return True, "No relevant knowledge base document was retrieved."

        top_score = float(retrieved_docs[0]["score"])
        if top_score < self.config.kb_score_threshold:
            return True, "Top knowledge base retrieval score is below the confidence threshold."

        if groundedness["groundedness"] == "unsupported":
            return True, "The answer is not grounded in the retrieved context."

        answer_lower = final_answer.lower()
        if "human support" in answer_lower or "escalate" in answer_lower:
            return True, "The agent's own answer recommends human support."

        return False, "The agent found enough grounded context to answer directly."

    def _demo_mode_result(self, user_id: str, question: str) -> dict[str, Any]:
        docs = self.retrieve_knowledge(question)
        mems = self.retrieve_memory(user_id, question)
        note = (
            "[DEMO MODE: No API key set, so no real agent reasoning was performed. "
            "Showing retrieval results only.]"
        )
        groundedness = groundedness_check(note, docs, mems)
        should_escalate, escalation_reason = self.decide_escalation(
            question=question,
            retrieved_docs=docs,
            final_answer=note,
            groundedness=groundedness,
        )
        return {
            "question": question,
            "user_id": user_id,
            "action_trace": [
                "DEMO MODE: ran search_knowledge_base",
                "DEMO MODE: ran search_user_memory",
            ],
            "retrieved_docs": docs,
            "retrieved_memory": mems,
            "answer": note,
            "groundedness": groundedness["groundedness"],
            "groundedness_score": groundedness["groundedness_score"],
            "should_escalate": should_escalate,
            "escalation_reason": escalation_reason,
        }

    @staticmethod
    def _docs_to_text(docs: list[dict[str, Any]]) -> str:
        if not docs:
            return "No relevant product documents found."
        lines = []
        for doc in docs:
            lines.append(
                f"{doc['doc_id']}: {doc['title']} | {doc['text']} "
                f"| tags: {', '.join(doc['tags'])}"
            )
        return "\n".join(lines)

    @staticmethod
    def _memory_to_text(mems: list[dict[str, Any]]) -> str:
        if not mems:
            return "No prior user memory found for this user."
        lines = []
        for mem in mems:
            lines.append(f"{mem['memory_id']}: {mem['text']} | tags: {', '.join(mem['tags'])}")
        return "\n".join(lines)

    @staticmethod
    def _format_doc_result(record: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "doc_id": record["doc_id"],
            "title": record["title"],
            "text": record["text"],
            "tags": record["tags"],
            "score": round(float(score), 4),
        }

    @staticmethod
    def _format_memory_result(record: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "memory_id": record["memory_id"],
            "user_id": record["user_id"],
            "text": record["text"],
            "tags": record["tags"],
            "score": round(float(score), 4),
        }
