# Demo Video Script

## 1. Project title and goal

Hello, this is my project called **Memory-based Customer Support Agent with Retrieval Evaluation**.

The goal is to build a small AI support agent for a fictional SaaS product called CloudBox AI. The main idea is that the agent does not answer only from model knowledge. It first retrieves from a product knowledge base and from user conversation memory, then it generates an answer, checks groundedness, and decides whether the case should be escalated to human support.

## 2. Show the data files

First, I will show the dataset folder.

Here we have:

- `knowledge_base.jsonl`
- `user_memory.jsonl`
- `test_set.jsonl`
- `annotation_guide.md`

The knowledge base contains synthetic support documents about API keys, billing, refunds, team invitations, privacy, uploads, and other product topics.

The user memory file contains short synthetic history records for different users. For example, one user already tried regenerating an API key, so the agent should not repeat that same advice.

The test set contains evaluation questions with gold labels for relevant documents, relevant memory, and whether the case should be escalated.

The annotation guide explains how those labels were assigned.

## 3. Run one example question

Now I will run one demo question:

```bash
python -m src.cli --user_id user_001 --question "I still cannot use my API key. What should I do?"
```

## 4. Show the action trace

The CLI prints an action trace:

- `TOOL_CALL: search_kb`
- `TOOL_CALL: search_memory`
- `TOOL_CALL: generate_answer`
- `TOOL_CALL: groundedness_check`
- `TOOL_CALL: escalation_check`

This shows that the system behaves like a simple fixed-policy agent.

## 5. Show retrieved knowledge base documents

Next, I look at the retrieved knowledge base documents.

I expect to see documents about:

- API key troubleshooting
- 401 authentication errors
- human support escalation

These retrieved documents provide the product facts that the answer should be based on.

## 6. Show retrieved user memory

Then I show the retrieved memory.

For this user, the memory says that the user already tried regenerating the API key and still got a 401 error.

This is important because memory changes the answer. The agent should avoid repeating the same basic step again.

## 7. Show the generated answer

Now I show the final answer.

The answer should say that the user should verify the active key and environment, and because the user already tried regenerating the key, the case should move toward human support.

## 8. Show groundedness and escalation

After answer generation, the system prints:

- groundedness label
- groundedness score
- escalation decision
- escalation reason

The groundedness check is a lightweight heuristic. It does not prove truth, but it checks whether the answer is lexically supported by the retrieved context.

## 9. Run the evaluation command

Now I run:

```bash
python -m src.cli --evaluate
```

This evaluates retrieval quality, escalation accuracy, and groundedness statistics over the whole synthetic test set.

## 10. Explain why this is an Information Retrieval agent

This project is an Information Retrieval system because the answer depends on retrieving relevant text from stored collections before generation.

The main retrieval method is BM25, which is a standard lexical IR baseline. I evaluate it with Hit@3, Precision@3, and Recall@3.

## 11. Explain why this counts as a fixed-policy AI agent

I describe this system as a fixed-policy AI agent because it always follows the same action pipeline.

It is not a fully autonomous planner and it does not do dynamic tool selection. But it still has agent-like behavior because it takes structured steps, uses retrieved context, and makes a final action decision about answering or escalating.

## 12. Brief limitations

Finally, I mention the limitations:

- the dataset is synthetic
- BM25 is lexical only
- the groundedness check is heuristic
- the system does not include dense retrieval or dynamic tool calling

That is the end of the demo.
