# Report Draft: Memory-based Customer Support Agent with Retrieval Evaluation

GitHub repository: [to be added]  
Demo video: [to be added]

## 1. Introduction

This project implements a small customer support agent for a fictional SaaS product called CloudBox AI. The goal is to show that an AI system can improve its answers by retrieving information from external sources before generating a response. In this project, the agent retrieves from two sources: a product knowledge base and a user memory store.

The project is designed as a simple and explainable system for an Information Retrieval course assignment. It is not a fully autonomous agent. Instead, it follows a fixed sequence of steps and exposes that sequence through an action trace.

## 2. System Design

The system is a fixed-policy retrieval agent with the following pipeline:

1. retrieve knowledge base documents
2. retrieve user memory
3. generate an answer
4. run groundedness checking
5. decide whether to escalate to human support

This design makes the system easy to explain and evaluate. It also separates retrieval from generation, which is important for understanding how external context improves responses.

## 3. Information Retrieval Method

The main retrieval method is BM25. I used BM25 because it is a classic and strong lexical baseline in Information Retrieval. It scores documents using term matching, term frequency, inverse document frequency, and document length normalization.

For this project, BM25 was a better choice than using a very simple TF-IDF cosine similarity baseline, because it is more standard and more suitable for an introductory Master-level IR project. Dense retrieval was not included in this version in order to keep the implementation small and interpretable.

## 4. Memory Component

The memory component stores synthetic conversation history for each user. Memory retrieval is also handled with BM25, but only over records that belong to the current user. This makes the memory retrieval user-specific.

The purpose of memory is not long-term personalization in a complex sense. Instead, it is used to avoid repeated advice and to add useful context. For example, if a user already tried regenerating an API key, the agent should not simply repeat the same suggestion again.

## 5. Action Trace and Agent Behavior

The agent prints a simple action trace such as:

- `TOOL_CALL: search_kb`
- `TOOL_CALL: search_memory`
- `TOOL_CALL: generate_answer`
- `TOOL_CALL: groundedness_check`
- `TOOL_CALL: escalation_check`
- `ACTION: answer_user` or `ACTION: escalate_to_human_support`

This makes the system behavior explicit. It is important to note that the agent does not perform dynamic planning or open-ended tool selection. It always follows the same retrieval pipeline. I describe it as a fixed-policy retrieval agent, not a dynamic tool-calling agent.

## 6. Evaluation Method

The project uses a synthetic test set with gold labels for:

- relevant knowledge base documents (`gold_docs`)
- relevant user memory (`gold_memory`)
- escalation decision (`should_escalate`)

The retrieval metrics are Hit@3, Precision@3, and Recall@3. These are appropriate because the agent only retrieves a small top-k set before answer generation. I also report escalation accuracy and groundedness statistics.

The groundedness check is heuristic. It measures lexical overlap between answer units and retrieved context units after lowercasing, punctuation removal, stopword removal, and 2-gram extraction. This is not a proof of truth, but it is a simple hallucination-risk signal.

This is a lightweight heuristic groundedness check. It does not prove factual correctness, but it checks whether the answer is lexically grounded in the retrieved context. It is used as a simple hallucination-risk signal, not as a perfect hallucination detector.

## 7. Results

The system was evaluated by running the CLI evaluation command over the 20-question synthetic test set. In the validated local run, the summary was:

- knowledge base Hit@3 = 0.90
- knowledge base Precision@3 = 0.3667
- knowledge base Recall@3 = 0.8333
- memory Hit@3 = 1.00
- memory Precision@3 = 0.3333
- memory Recall@3 = 1.00
- escalation accuracy = 1.00
- groundedness counts = 19 supported, 1 partially supported, 0 unsupported
- average groundedness score = 0.7542

Because the dataset is small and synthetic, the results are mainly useful for controlled inspection rather than for claiming general performance.

## 8. Reflection on Using AI Coding Tools

AI coding tools were useful for speeding up implementation, generating boilerplate, and helping structure the project. They were especially useful for quickly creating the CLI, test scaffolding, and documentation drafts.

However, I treated the generated code as a draft, not as automatically correct. I still needed to inspect the logic, run tests, and verify that the system behavior matched the assignment description.

## 9. How I Checked and Fixed AI Hallucinations During Development

- I did not blindly trust AI-generated code or text.
- I checked whether generated dataset labels matched the knowledge base.
- I reviewed `gold_docs`, `gold_memory`, and `should_escalate` labels using the annotation guide.
- I ran tests and demo commands to check that the system behaved as described.
- I checked that the agent did not claim to use dynamic tool calling when it only used a fixed pipeline.
- I added a groundedness heuristic to flag unsupported answers.
- I kept API keys out of the repository.

## 10. Limitations

The dataset is synthetic. Real customer support data usually contains private information, including account details and message history. Because of this, I used a small synthetic dataset for controlled evaluation. This is a limitation, but it makes the labels easier to inspect and the evaluation easier to reproduce.

Other limitations are:

- BM25 is purely lexical and may miss semantic similarity.
- The memory store is very small and simplified.
- The escalation logic is rule-based rather than learned.
- The groundedness check is heuristic and may miss subtle hallucinations.
- The system does not include dense retrieval, reranking, or autonomous multi-step planning.
