# Evaluation Module

The `eval` folder has code to test how well our RAG pipeline works. Currently it combines
LLM and Non LLM based Evaluations.

In future it will check both parts of the system: Retrieval and Generation.

## **LLM Metrics Evaluation** : (llm_metrics.py)

- Uses a language model to evaluate the quality of generated responses.
- Includes metrics such as factual correctness, relevancy, faithfulness, and context precision.
- Additional metrics can be created with AspectCritic from ragas library.

This file has the `LLMMetricsEvaluator` class. It checks the model’s answers for:

- **Factual Correctness**: Is the answer true?  
- **Relevance**: Does the answer fit the question?  
- **Faithfulness**: Does the answer stick to the retrieved info?  
- **Context Precision**: How many retrieved passages actually support the answer?

## **Non-LLM Metrics Evaluation** : (non_llm_metrics.py)

- Uses traditional NLP metrics to evaluate the quality of generated responses.
- Includes metrics such as BLEU, ROUGE-L, METEOR, cosine similarity, and context recall.

This file has the `NonLLMMetricsEvaluator` class. It uses standard NLP scores:

- **BLEU** : Overlap of words between output and reference but here as we still finalizing the golden dataset, we have taken the overlap between input(user query) and output(llm generated response)
- **ROUGE-L** : How much of the reference is covered by the output, here we have taken the same approach like `BLEU`. With experimentations we can finalize to take which combination suits better
for the use case i.e. (input,output), (retrieved,output), (referenced,output)
- **METEOR** : Matches words, stems, and synonyms
- **Cosine Similarity** : How close two text embeddings are
- **Context Recall** : How many correct contexts were found

## Examples

```python
from src.backend.eval.llm_metrics import LLMMetricsEvaluator
from ragas.dataset_schema import SingleTurnSample
import asyncio

# Create a sample
sample = SingleTurnSample(
    user_input="When was the first Super Bowl?",
    response="The first Super Bowl was on January 15, 1967.",
    retrieved_contexts=[
        "The first AFL–NFL World Championship Game was played on January 15, 1967."
    ]
)

# Initialize evaluator
evaluator = LLMMetricsEvaluator()

# Run evaluation
results = asyncio.run(evaluator.report(sample, with_golden=False))
print("LLM Metrics:", results)
```

```python
from src.backend.eval.non_llm_metrics import NonLLMMetricsEvaluator
from ragas.dataset_schema import SingleTurnSample

# Create a sample
sample = SingleTurnSample(
    user_input="When was the first Super Bowl?",
    response="The first Super Bowl was on January 15, 1967.",
    reference="The first Super Bowl was held on January 15, 1967.",
    retrieved_contexts=[
        "The first AFL–NFL World Championship Game was played on January 15, 1967."
    ]
)

# Initialize evaluator
evaluator = NonLLMMetricsEvaluator()

# Run evaluation
results = evaluator.report(sample, with_golden=False)
print("Non-LLM Metrics:", results)
```

To run the combined metrics with RAG pipeline

```bash
python path/to/src/backend/eval/_metrics.py
```

## References

- [RAGAS Metrics Documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [BLEU and ROUGE Documentation](https://www.geeksforgeeks.org/understanding-bleu-and-rouge-score-for-nlp-evaluation/)
- [Metric Design Principles](https://github.com/explodinggradients/ragas/blob/main/docs/concepts/metrics/overview/index.md?utm_source=chatgpt.com)
