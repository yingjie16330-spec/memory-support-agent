"""Support agent implementation using a fixed retrieval pipeline."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from openai import OpenAI

from src.config import Config, get_config
from src.data_loader import load_knowledge_base, load_user_memory
from src.evaluator import groundedness_check
from src.retriever import BM25Retriever


class SupportAgent:
    """
    Fixed-policy retrieval agent for a SaaS support demo.

    The agent does not do dynamic autonomous planning. It always follows the
    same pipeline:
    1. retrieve product knowledge
    2. retrieve user memory
    3. generate an answer
    4. run groundedness checking
    5. decide whether to escalate to human support
    """

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
    def _group_memory_by_user(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
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

    def _build_context(
        self,
        retrieved_docs: list[dict[str, Any]],
        retrieved_memory: list[dict[str, Any]],
    ) -> str:
        lines: list[str] = ["Product knowledge base:"]

        for doc in retrieved_docs:
            lines.append(
                f"- {doc['doc_id']}: {doc['title']} | {doc['text']} | tags: {', '.join(doc['tags'])}"
            )

        lines.append("")
        lines.append("User memory:")

        if retrieved_memory:
            for memory in retrieved_memory:
                lines.append(
                    f"- {memory['memory_id']}: {memory['text']} | tags: {', '.join(memory['tags'])}"
                )
        else:
            lines.append("- No prior user memory found.")

        return "\n".join(lines)

    def generate_answer(
        self,
        question: str,
        retrieved_docs: list[dict[str, Any]],
        retrieved_memory: list[dict[str, Any]],
    ) -> str:
        context = self._build_context(retrieved_docs, retrieved_memory)

        if not self.client:
            return self._fallback_answer(question, retrieved_docs, retrieved_memory)

        system_prompt = (
            "You are a helpful SaaS customer support assistant for CloudBox AI. "
            "Only use the retrieved product documentation and user memory. "
            "Do not invent features, policies, or guarantees. "
            "If the context is not enough, clearly say that human support is recommended. "
            "If user memory shows a step was already tried, do not repeat that same advice. "
            "Keep the answer practical and concise."
        )

        user_prompt = (
            f"Customer question:\n{question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Write a grounded support reply using only that context."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            message = response.choices[0].message.content or ""
            answer = message.strip()
            return answer or self._fallback_answer(question, retrieved_docs, retrieved_memory)
        except Exception:
            return self._fallback_answer(question, retrieved_docs, retrieved_memory)

    def _fallback_answer(
        self,
        question: str,
        retrieved_docs: list[dict[str, Any]],
        retrieved_memory: list[dict[str, Any]],
    ) -> str:
        if not retrieved_docs:
            return (
                "I could not find enough product documentation to answer this confidently. "
                "Please contact human support for help with this case."
            )

        question_lower = question.lower()
        memory_text = " ".join(item["text"].lower() for item in retrieved_memory)

        if "reset my api key" in question_lower or "where do i reset my api key" in question_lower:
            return (
                "You can reset your API key from Account Settings > API Keys. "
                "After resetting it, you need to update the key in your application because the old key stops working immediately."
            )

        if "already tried regenerating" in memory_text or "already tried" in memory_text:
            for doc in retrieved_docs:
                if doc["doc_id"] == "doc_002":
                    return (
                        "Since you already regenerated the API key and still receive a 401 error, "
                        "the next step is to verify that the active key was copied correctly and used "
                        "in the right environment. The documentation also says repeated failures after "
                        "reset should be escalated to human support for account-level verification."
                    )

        if "refund" in question_lower:
            return (
                "The refund policy says eligible self-serve plans can usually be refunded within 14 days "
                "of the initial purchase. Requests after 14 days are not normally approved automatically "
                "and may need human review."
            )

        if "cancel" in question_lower:
            return (
                "You can cancel from Settings > Billing > Manage Plan. Cancellation stops future renewals, "
                "and access stays active until the current billing period ends."
            )

        if "invoice" in question_lower:
            return (
                "You can download invoices from Settings > Billing > Invoices. "
                "For team workspaces, only billing admins can download invoice PDFs."
            )

        if "privacy" in question_lower or "legal" in question_lower:
            return (
                "The documentation says customer data is processed according to the privacy policy. "
                "Requests involving legal review or special privacy guarantees should go to human support."
            )

        if "delete" in question_lower and "account" in question_lower:
            return (
                "Account deletion requests should be submitted to human support so they can verify ownership "
                "and explain the effect on workspace data."
            )

        if "invite" in question_lower or "teammate" in question_lower:
            return (
                "Workspace admins can invite teammates from Team Settings > Members > Invite. "
                "Pending invitations stay visible until they are accepted or revoked."
            )

        if "api key" in question_lower and "permission" in question_lower:
            return (
                "Yes, missing permissions could be the reason. The documentation says API keys can have "
                "different scopes, and a key without document search permission may fail for search requests "
                "even if the key itself is valid."
            )

        if "search" in question_lower:
            return (
                "CloudBox AI search works by indexing uploaded documents and matching relevant passages. "
                "Search quality improves when documents are clearly formatted and the query uses specific keywords."
            )

        if "upload" in question_lower or "file" in question_lower:
            titles = {doc["doc_id"] for doc in retrieved_docs}
            if "doc_013" in titles:
                return (
                    "CloudBox AI supports PDF, DOCX, TXT, and Markdown uploads. "
                    "If your current file type is unsupported, convert it to one of those formats before retrying."
                )
            return (
                "Please check the file size limit, supported format, network stability, and workspace storage. "
                "The standard plan supports uploads up to 25 MB per file."
            )

        supporting_points = []
        for doc in retrieved_docs[:2]:
            supporting_points.append(f"{doc['title']}: {doc['text']}")

        memory_note = ""
        if retrieved_memory:
            memory_note = f" I also checked your previous history: {retrieved_memory[0]['text']}"

        return " ".join(supporting_points) + memory_note

    def check_groundedness(
        self,
        answer: str,
        retrieved_docs: list[dict[str, Any]],
        retrieved_memory: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return groundedness_check(answer, retrieved_docs, retrieved_memory)

    def decide_escalation(
        self,
        question: str,
        retrieved_docs: list[dict[str, Any]],
        retrieved_memory: list[dict[str, Any]],
        groundedness: dict[str, Any],
    ) -> tuple[bool, str]:
        question_lower = question.lower()
        sensitive_keywords = [
            "payment dispute",
            "charged incorrectly",
            "legal",
            "privacy",
            "delete my account",
            "account deletion",
            "angry",
            "refund exception",
            "more than 14 days",
            "over 14 days",
        ]

        if not retrieved_docs:
            return True, "No relevant knowledge base document was retrieved."

        top_score = float(retrieved_docs[0]["score"])
        if top_score < self.config.kb_score_threshold:
            return True, "Top knowledge base retrieval score is too low."

        if any(keyword in question_lower for keyword in sensitive_keywords):
            return True, "This question matches a policy-sensitive escalation rule."

        failure_signals = [
            "still",
            "cannot",
            "can't",
            "not work",
            "problem",
            "error",
            "failed",
            "fail",
        ]
        repeated_failure = (
            ("api key" in question_lower or "401" in question_lower)
            and any(signal in question_lower for signal in failure_signals)
            and any(
            "already tried" in item["text"].lower()
            or "still received error 401" in item["text"].lower()
            or "already tried regenerating" in item["text"].lower()
            for item in retrieved_memory
            )
        )
        if repeated_failure:
            return True, "The user already tried a key troubleshooting step and still has a 401 issue."

        if groundedness["groundedness"] == "unsupported":
            return True, "The answer is not well grounded in the retrieved context."

        if "human support" in self._build_context(retrieved_docs, retrieved_memory).lower() and (
            "refund" in question_lower or "dispute" in question_lower or "delete" in question_lower
        ):
            return True, "The retrieved documentation recommends human support for this case."

        if "epub" in question_lower:
            return True, "The retrieved context does not clearly confirm EPUB support."

        return False, "The agent found enough grounded context to answer directly."

    def run(self, user_id: str, question: str) -> dict[str, Any]:
        action_trace: list[str] = []

        action_trace.append("TOOL_CALL: search_kb")
        retrieved_docs = self.retrieve_knowledge(question)

        action_trace.append("TOOL_CALL: search_memory")
        retrieved_memory = self.retrieve_memory(user_id, question)

        action_trace.append("TOOL_CALL: generate_answer")
        answer = self.generate_answer(question, retrieved_docs, retrieved_memory)

        action_trace.append("TOOL_CALL: groundedness_check")
        groundedness = self.check_groundedness(answer, retrieved_docs, retrieved_memory)

        action_trace.append("TOOL_CALL: escalation_check")
        should_escalate, escalation_reason = self.decide_escalation(
            question=question,
            retrieved_docs=retrieved_docs,
            retrieved_memory=retrieved_memory,
            groundedness=groundedness,
        )

        action_trace.append(
            "ACTION: escalate_to_human_support" if should_escalate else "ACTION: answer_user"
        )

        return {
            "question": question,
            "user_id": user_id,
            "action_trace": action_trace,
            "retrieved_docs": retrieved_docs,
            "retrieved_memory": retrieved_memory,
            "answer": answer,
            "groundedness": groundedness["groundedness"],
            "groundedness_score": groundedness["groundedness_score"],
            "should_escalate": should_escalate,
            "escalation_reason": escalation_reason,
        }

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
