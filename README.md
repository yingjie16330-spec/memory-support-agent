# Memory-based Customer Support Agent with Retrieval Evaluation

## Overview

This project implements a small Python support agent for a synthetic SaaS product called **CloudBox AI**. The agent does not answer from LLM knowledge alone. Instead, it first retrieves information from:

1. a product knowledge base
2. user conversation memory

Then it generates a grounded answer, runs a lightweight groundedness check, and decides whether the case should be escalated to human support.

This project is intentionally simple and suitable for a university Information Retrieval assignment, a GitHub portfolio project, and an entry-level AI application showcase.

## Why this is an AI agent

This system is agent-like because it follows a structured decision pipeline with explicit actions:

- retrieve product knowledge
- retrieve user memory
- generate an answer
- check groundedness
- decide whether to escalate

It also prints an action trace such as `TOOL_CALL: search_kb` and `ACTION: answer_user`.

## Why this is a fixed-policy retrieval agent

This version uses a fixed retrieval pipeline rather than dynamic tool calling. The agent always retrieves from the product knowledge base and user memory before generating an answer. Dynamic tool selection, real function calling, dense retrieval, web search, and multi-step autonomous planning are future work.

## Why this is Information Retrieval

The core system behavior depends on retrieving relevant context from stored text collections before answer generation. The main retrieval method is **BM25**, which is a classic lexical IR baseline. The project evaluates retrieval quality using Hit@3, Precision@3, and Recall@3.

## Main features

- Synthetic CloudBox AI support dataset
- BM25 retrieval for product knowledge base documents
- BM25 retrieval for user-specific memory
- Fixed-policy support agent with action trace
- OpenAI-compatible chat API support
- Default configuration compatible with Berget.AI
- No-API-key fallback mode
- Lightweight groundedness heuristic
- Rule-based escalation logic
- CLI demo
- Retrieval and escalation evaluation
- Basic pytest coverage

## Project structure

```text
memory-support-agent/
├── README.md
├── requirements.txt
├── .env.example
├── report.md
├── demo_script.md
├── data/
│   ├── knowledge_base.jsonl
│   ├── user_memory.jsonl
│   ├── test_set.jsonl
│   └── annotation_guide.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── retriever.py
│   ├── agent.py
│   ├── evaluator.py
│   └── cli.py
└── tests/
    └── test_basic.py
```

## Dataset description

The project uses a small synthetic dataset for a fictional SaaS product called **CloudBox AI**, which supports document upload, AI search, API key management, billing, and team collaboration.

Data files:

- `data/knowledge_base.jsonl`: 18 product support documents
- `data/user_memory.jsonl`: 10 user memory records
- `data/test_set.jsonl`: 20 evaluation questions
- `data/annotation_guide.md`: explanation of gold labels

The dataset is synthetic because real support data often contains private customer information. A synthetic dataset is easier to share, inspect, and evaluate in a university setting.

## BM25 retrieval method

The project uses `rank_bm25` and a simple lowercase tokenizer.

BM25 scores are raw retrieval scores used for ranking. They are not probabilities, and with very small corpora some scores may be low or negative.

BM25 is used because:

- it is a strong and interpretable lexical retrieval baseline
- it accounts for term matching, term frequency, inverse document frequency, and document length normalization
- it is more suitable than a very basic TF-IDF cosine baseline for a small Master-level IR project

This version does **not** use dense embeddings. Dense retrieval is listed as future work.

## Memory retrieval method

User memory retrieval is also based on BM25, but only over memory records that belong to the current `user_id`. This makes the retrieval user-specific and lets the system avoid repeating advice that the user already tried.

## Action trace and `TOOL_CALL`

The agent prints a simple trace:

- `TOOL_CALL: search_kb`
- `TOOL_CALL: search_memory`
- `TOOL_CALL: generate_answer`
- `TOOL_CALL: groundedness_check`
- `TOOL_CALL: escalation_check`
- `ACTION: answer_user` or `ACTION: escalate_to_human_support`

This is meant to demonstrate fixed policy agent behavior in a way that is easy to present.

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Optional: configure API access

Copy `.env.example` to `.env` and fill in your own key:

```bash
cp .env.example .env
```

Default values are compatible with Berget.AI:

```env
OPENAI_API_KEY=your_berget_api_key_here
OPENAI_BASE_URL=https://api.berget.ai/v1
OPENAI_MODEL=gemma-4-31B-it
```

You can also point the same code to another OpenAI-compatible provider by changing the environment variables.

## Run without API key

The project still works without `OPENAI_API_KEY`. In that case, it uses a simple template-based fallback answer generator based on the retrieved documents and memory.

This is useful for:

- demo mode
- local testing
- environments without LLM access

## Run with Berget.AI or another OpenAI-compatible API

If `OPENAI_API_KEY` is set, the agent calls the chat completion API with:

- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

The prompt tells the model to use only the retrieved context and avoid inventing product policies or features.

## Demo command

```bash
python -m src.cli --user_id user_001 --question "I still cannot use my API key. What should I do?"
```

## Evaluation command

```bash
python -m src.cli --evaluate
```

Current result in the validated local run:

```text
Number of test questions: 20
Knowledge Base Hit@3: 0.9
Knowledge Base Precision@3: 0.3667
Knowledge Base Recall@3: 0.8333
Memory Hit@3: 1.0
Memory Precision@3: 0.3333
Memory Recall@3: 1.0
Escalation Accuracy: 1.0
Groundedness Counts: {"supported": 19, "partially_supported": 1, "unsupported": 0}
Average Groundedness Score: 0.7542
```

## List example commands

```bash
python -m src.cli --list-examples
```

## Example output

```text
User ID: user_001
Question: I still cannot use my API key. What should I do?

Action Trace:
- TOOL_CALL: search_kb
- TOOL_CALL: search_memory
- TOOL_CALL: generate_answer
- TOOL_CALL: groundedness_check
- TOOL_CALL: escalation_check
- ACTION: escalate_to_human_support
```

The CLI also prints retrieved document IDs, memory IDs, groundedness score, and escalation reason.

## Evaluation metrics

### Knowledge base retrieval

- `Hit@3`: at least one gold document appears in top 3
- `Precision@3`: number of relevant retrieved documents divided by 3
- `Recall@3`: number of retrieved gold documents divided by number of gold documents

### Memory retrieval

- `Memory Hit@3`
- `Memory Precision@3`
- `Memory Recall@3`

These are reported on questions that have at least one gold memory label.

### Escalation accuracy

This compares the agent's predicted escalation decision against the synthetic `should_escalate` label.

## Groundedness heuristic

This is a lightweight heuristic groundedness check. It does not prove factual correctness, but it checks whether the answer is lexically grounded in the retrieved context. It is used as a simple hallucination-risk signal, not as a perfect hallucination detector.

Implementation summary:

1. combine retrieved documents and memory into one context
2. lowercase text
3. remove punctuation
4. remove common stopwords
5. extract content words and 2-grams
6. compute lexical overlap between answer units and context units
7. label the answer as `supported`, `partially_supported`, or `unsupported`

## Limitations

- The dataset is synthetic and small.
- BM25 is lexical, so it can miss semantic matches.
- The groundedness check is heuristic, not a full hallucination detector.
- Escalation rules are hand-written and simplified.
- The fallback answer generator is basic.
- The system does not do real dynamic tool selection.

## Possible future improvements

- Dense retrieval with embeddings
- Hybrid retrieval with BM25 plus embeddings
- Better reranking
- Real support ticket history instead of synthetic memory
- Better hallucination checking
- More realistic escalation policies
- Web UI or notebook demo
- Real function calling and dynamic tool selection
