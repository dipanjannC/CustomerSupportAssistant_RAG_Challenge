# Overview

This folder contains the experiments and purpose of each module.

## Contents

- **`data_validations.py`**:
  - Validates dataset quality (e.g., spelling, ambiguity).
  - Outputs validation results for analysis.
  - This step is for understanding the dataset and creating new features with
  traditional NLP approaches.

- **`evals_with_golden.py`**:
  - Placeholder for evaluation metrics like Precision@k, Recall@k, and MRR.

- **`model.py`**:
  - Defines MistralAI model configuration (e.g., temperature, top_p).
  - It is used for testing phoenix evaluations.

- **`phoenix_eval.py`**:
  - Evaluates the RAG pipeline using Phoenix tools.
  - Supports relevance classification and explanation generation.

- **`reranker.py`**:
  - Placeholder for reranking logic to improve retrieval relevance.

## Important Notes

- Langchain v0.1 is deprecated, they are extending support and development on langgraph.

## Future Work

- Add evaluation metrics in `evals_with_golden.py`.
- Implement reranking logic in `reranker.py` useful for ranking retrieved documents
as LLMs sometimes are dependent on the order of documents it received.
- Create golden dataset for overall validation and evaluation capabilities with Arize Phoenix, we can track , monitor and create datasets with annotations and utilize it for validations.
- Utilize Agentic RAGs or langgraph(passing memory states) for running LLMs on loop till it reaches specific goal.
- Multi Turn Conversation with  langgraph , utilization of pydantic for structured outputs is awesome. Here we can describe for new features to be created by LLMs and how it can act as a a judge.

