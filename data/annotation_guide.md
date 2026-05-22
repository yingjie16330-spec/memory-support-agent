# Annotation Guide for the Synthetic Test Set

## Purpose of the synthetic test set

This test set is designed for a small Information Retrieval and AI agent course project. It helps evaluate whether the agent retrieves the right product knowledge base documents, retrieves useful user memory, and decides when to escalate to human support.

The labels are synthetic, not collected from real customers. This makes the project easier to inspect and explain, but it also means the data is simpler than a real support environment.

## What `gold_docs` means

`gold_docs` is the list of knowledge base document IDs that should be considered relevant for answering the question.

These labels are used to evaluate product knowledge retrieval with:

- Hit@3
- Precision@3
- Recall@3

In some questions, more than one document is relevant because the answer needs both a main policy document and an escalation or troubleshooting document.

## What `gold_memory` means

`gold_memory` is the list of user memory record IDs that should be considered relevant for the question.

These labels matter when the user has prior interaction history that changes how the answer should be written. For example, if the user already tried regenerating an API key, the agent should avoid repeating that same suggestion.

## What `should_escalate` means

`should_escalate` is a gold label indicating whether the case should be handed to human support.

This label is assigned when:

- the issue matches a policy that clearly requires human handling
- the problem is risky or sensitive, such as legal/privacy concerns
- the user requests account deletion
- the case involves payment disputes or refund exceptions
- the question is likely to be under-supported by the available context

## How labels were assigned

Each question was written against the synthetic CloudBox AI knowledge base and memory records.

The labeling process was:

1. Read the question.
2. Identify which knowledge base documents are directly relevant.
3. Check whether the user has prior memory that changes the response.
4. Decide whether the case should escalate based on project rules.

The goal is not to create the only possible gold label set. The goal is to create a reasonable and inspectable evaluation benchmark for a student project.

## Example: knowledge base only

Question:

`Where do I reset my API key?`

Expected labels:

- `gold_docs`: `doc_001`
- `gold_memory`: none
- `should_escalate`: `false`

Reason:
The answer can be handled directly from the product documentation. User memory is not necessary.

## Example: knowledge base plus memory

Question:

`I still cannot use my API key. What should I do next?`

Expected labels:

- `gold_docs`: `doc_002`, `doc_017`
- `gold_memory`: `mem_001`
- `should_escalate`: `true`

Reason:
The knowledge base explains the 401 issue and when to contact support. The memory shows the user already regenerated the key, so the agent should not repeat the same advice. This is a good example of memory helping reduce repeated or low-value guidance.

## Example: escalate to human support

Question:

`Can you delete my account and all stored data for me?`

Expected labels:

- `gold_docs`: `doc_010`, `doc_017`, optionally `doc_008`
- `gold_memory`: `mem_005`
- `should_escalate`: `true`

Reason:
Account deletion is explicitly defined as a human-support case in the synthetic knowledge base.

## Known limitations of the synthetic labels

- Some questions could plausibly match more than one relevant document.
- Some escalation decisions are simplified compared with real company workflows.
- The dataset is small, so metrics may vary a lot with small implementation changes.
- The labels are easier and cleaner than real support data, which often includes ambiguity, typos, private information, and incomplete context.

This guide is intended to help manual review. If a label looks questionable, it should be checked against the knowledge base and revised transparently.
