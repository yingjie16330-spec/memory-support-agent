# Demo Video Script

## 1. Project title and goal

Hello, this is my project called **Memory-based Customer Support Agent with
Retrieval Evaluation**.

The goal is to build a small AI support agent for a fictional SaaS product
called CloudBox AI. The main idea is that the agent does not answer only from
model knowledge. It is a tool-calling agent: the language model is given two
retrieval tools and decides which to call before answering. It retrieves from a
product knowledge base and from user conversation memory, generates an answer,
checks groundedness, and decides whether to escalate to human support.

## 2. Show the data files

First, I will show the dataset folder. It contains:

- `knowledge_base.jsonl`
- `user_memory.jsonl`
- `test_set.jsonl`
- `annotation_guide.md`

The knowledge base contains synthetic support documents about API keys, billing,
refunds, team invitations, privacy, uploads, and other product topics.

The user memory file contains short synthetic history records for different
users. For example, one user already tried regenerating an API key, so the agent
should not repeat that same advice.

The test set contains evaluation questions with gold labels for relevant
documents, relevant memory, and whether the case should be escalated. The
annotation guide explains how those labels were assigned.

## 3. Run one example question

Now I will run one demo question:

```bash
python -m src.cli --user_id user_001 --question "I still cannot use my API key. What should I do?"
```

## 4. Show the action trace

The CLI prints an action trace showing the tools the language model actually
chose to call, for example:

- `LLM called search_user_memory(query='API key issue')`
- `LLM called search_knowledge_base(query='API key troubleshooting')`
- `Decision: escalate to human support`

The important point is that the model decided which tools to call and even wrote
its own search queries. The tool selection is not hard-coded.

## 5. Show retrieved knowledge base documents

Next, I look at the retrieved knowledge base documents. For this question I
expect documents about API key troubleshooting, 401 authentication errors, and
API key permissions. These provide the product facts the answer is based on.

## 6. Show retrieved user memory

Then I show the retrieved memory. For this user, the memory says they already
tried regenerating the API key and still got a 401 error. This matters because
the memory changes the answer: the agent should avoid repeating the same basic
step.

## 7. Show the generated answer

Now I show the final answer. Because the model saw the memory, it does not just
repeat "regenerate your key". Instead it suggests verifying the active key and
environment, and recommends human support for account-level verification.

## 8. Show groundedness and escalation

After answer generation, the system prints the groundedness label, groundedness
score, escalation decision, and escalation reason.

I want to be honest about the groundedness score here: it is often low, not
because the answer is wrong, but because the model paraphrases the documents in
its own words, so the simple word-overlap heuristic underestimates how grounded
the answer really is. A semantic check would be more accurate. I list this as a
limitation.

## 9. Run the evaluation command

Now I run:

```bash
python -m src.cli --evaluate
```

This evaluates retrieval quality, escalation accuracy, and groundedness over the
whole synthetic test set. Retrieval is strong — knowledge base Hit@3 is 0.95 and
Recall@3 is about 0.88. Escalation accuracy is 0.80. The groundedness scores are
low for the paraphrasing reason I just explained.

## 10. Explain why this is an Information Retrieval agent

This project is an Information Retrieval system because the answer depends on
retrieving relevant text from stored collections before generation. The
retrieval method behind both tools is BM25, a standard lexical IR baseline,
evaluated with Hit@3, Precision@3, and Recall@3.

## 11. Explain why this is a tool-calling AI agent

This is a tool-calling agent because the language model is given two tools and
decides on its own which to call, and with what query, for each question. It can
call one tool or both, reads the results, and then produces a final decision to
answer or escalate. The tool selection is done by the model, not by fixed code.

## 12. Brief limitations and honesty note

Finally, the limitations:

- the dataset is synthetic and small
- BM25 is lexical only
- the groundedness check is heuristic and underestimates paraphrased answers
- escalation rules are hand-written

I also want to mention that an earlier version of this project used a fixed
pipeline with some hard-coded answers. I found that this did not really match
the idea of an agent and inflated the results, so I rewrote it to use real tool
calling. The numbers I show here come from that honest version.

That is the end of the demo.
