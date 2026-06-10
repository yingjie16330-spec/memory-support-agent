# Memory-based Customer Support Agent with Retrieval Evaluation

## Overview

This project implements a small Python support agent for a synthetic SaaS
product called **CloudBox AI**. The agent does not answer from the language
model's internal knowledge alone. Instead, the model is given two retrieval
tools and decides which to call before answering:

1. a product knowledge base
2. user conversation memory

It then generates a grounded answer, runs a lightweight groundedness check, and
decides whether the case should be escalated to human support.

The project is intentionally small and suitable for a university Information
Retrieval assignment and a GitHub portfolio project.

## Why this is an AI agent

The system is a tool-calling agent. The language model is given two tools
(`search_knowledge_base` and `search_user_memory`) and decides on its own which
tool(s) to call, and what query to use, based on the user's question. The agent
loops — calling tools, reading results, and calling again if needed — until it
produces a final answer. Tool selection and query formulation are done by the
model, not hard-coded.

## Why this is Information Retrieval

The agent's behavior depends on retrieving relevant context from stored text
collections before generating an answer. The retrieval method behind both tools
is **BM25**, a classic lexical IR baseline. Retrieval quality is evaluated with
Hit@3, Precision@3, and Recall@3.

## Main features

- Synthetic CloudBox AI support dataset
- BM25 retrieval for product knowledge base documents
- BM25 retrieval for user-specific memory
- LLM tool-calling agent that selects tools and queries
- OpenAI-compatible chat API (used with OpenAI gpt-4o-mini)
- Honest no-API-key demo mode (shows retrieval only)
- Lightweight groundedness heuristic
- Rule-based escalation logic
- CLI demo and evaluation
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

The project uses a small synthetic dataset for a fictional SaaS product called
**CloudBox AI** (document upload, AI search, API key management, billing, team
collaboration).

- `data/knowledge_base.jsonl`: product support documents
- `data/user_memory.jsonl`: user memory records
- `data/test_set.jsonl`: evaluation questions with gold labels
- `data/annotation_guide.md`: explanation of gold labels

The dataset is synthetic because real support data often contains private
customer information. A synthetic dataset is easier to share, inspect, and
evaluate in a university setting.

## How the agent works

The model is given two tools:

- `search_knowledge_base(query)` — searches product documentation
- `search_user_memory(query)` — searches the current user's history

For each question the model decides which tools to call and with what query. The
agent executes the calls, feeds results back to the model, and repeats up to a
small iteration limit until the model returns a final answer. After that, the
system runs a groundedness check and an escalation check on the retrieved
context.

### BM25 retrieval

The project uses `rank_bm25` with a simple lowercase tokenizer. BM25 scores are
raw ranking scores, not probabilities; with a very small corpus some scores can
be low or negative. BM25 is used because it is a strong, interpretable lexical
baseline that accounts for term frequency, inverse document frequency, and
document length normalization. Dense retrieval is listed as future work.

### Memory retrieval

User memory retrieval also uses BM25, but only over records belonging to the
current `user_id`. This makes retrieval user-specific and lets the agent avoid
repeating advice the user already tried.

### Action trace

The agent records the tools the model actually called, for example:

```text
- LLM called search_user_memory(query='API key issue')
- LLM called search_knowledge_base(query='API key troubleshooting')
- Decision: escalate to human support
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API access

Copy `.env.example` to `.env` and fill in your own key:

```bash
cp .env.example .env
```

Example `.env` for OpenAI:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

The code uses the OpenAI-compatible chat API, so it also works with other
OpenAI-compatible providers by changing `OPENAI_BASE_URL` and `OPENAI_MODEL`.

## Run without an API key (demo mode)

The project still runs without `OPENAI_API_KEY`. In that case it does **not**
perform real agent reasoning; it runs both retrievers and prints the retrieval
results with a clear `[DEMO MODE]` label. This is useful for local testing and
inspection without LLM access.

## Demo command

```bash
python -m src.cli --user_id user_001 --question "I still cannot use my API key. What should I do?"
```

## Evaluation command

```bash
python -m src.cli --evaluate
```

Result from a representative local run with OpenAI gpt-4o-mini (numbers vary
slightly between runs because tool queries are LLM-generated):

```text
Number of test questions: 20
Knowledge Base Hit@3: 0.85
Knowledge Base Precision@3: 0.35
Knowledge Base Recall@3: 0.825
Memory Hit@3: 0.7857
Memory Precision@3: 0.2619
Memory Recall@3: 0.7857
Escalation Accuracy: 0.8
Groundedness Counts: {"supported": 6, "partially_supported": 9, "unsupported": 5}
Average Groundedness Score: 0.3612
```

Note: groundedness scores are low because the model paraphrases the source
documents, so lexical overlap with the retrieved text is low even when answers
are factually grounded. A semantic groundedness check would be more accurate;
see Limitations.

## Evaluation metrics

### Knowledge base retrieval

- `Hit@3`: at least one gold document appears in the top 3
- `Precision@3`: relevant retrieved documents divided by 3
- `Recall@3`: retrieved gold documents divided by number of gold documents

### Memory retrieval

- `Memory Hit@3`, `Memory Precision@3`, `Memory Recall@3`, reported on questions
  that have at least one gold memory label

### Escalation accuracy

Compares the agent's predicted escalation decision against the synthetic
`should_escalate` label.

## Groundedness heuristic

A lightweight check of whether the answer is lexically grounded in the retrieved
context. It is a hallucination-risk signal, not a proof of factual correctness:

1. combine retrieved documents and memory into one context
2. lowercase and remove punctuation
3. remove common stopwords
4. extract content words and 2-grams
5. compute lexical overlap between answer units and context units
6. label as `supported`, `partially_supported`, or `unsupported`

## Limitations

- The dataset is synthetic and small.
- BM25 is lexical, so it can miss semantic matches.
- The groundedness heuristic underestimates paraphrased but correct answers.
- Escalation rules are hand-written (correct on 80% of test cases).
- Results vary between runs because tool queries are LLM-generated.
- The system does not do dense retrieval, reranking, or long-horizon planning.

## Possible future improvements

- Dense or hybrid retrieval (BM25 + embeddings)
- Reranking
- Semantic groundedness checking (e.g. LLM-as-judge)
- More realistic memory and escalation policies
- Real support ticket data instead of synthetic memory
