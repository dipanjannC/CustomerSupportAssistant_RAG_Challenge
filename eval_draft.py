from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore, RougeScore, ExactMatch

class NonLLMMetricsEvaluator:
    def __init__(self):
        self.metrics = {
            "BLEU": BleuScore(),
            "ROUGE-L": RougeScore(rouge_type="rougeL"),
            "ExactMatch": ExactMatch()
        }

    def evaluate(self, metric_name, user_input, response, reference):
        """
        Evaluate a single metric for a given sample.

        Args:
            metric_name (str): The name of the metric to evaluate.
            user_input (str): The user input text.
            response (str): The response text.
            reference (str): The reference text.

        Returns:
            float: The score for the specified metric.
        """
        if metric_name not in self.metrics:
            raise ValueError(f"Metric '{metric_name}' is not supported.")
        
        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            reference=reference
        )
        metric = self.metrics[metric_name]
        return metric.single_turn_score(sample)

    def run_all_metrics(self, user_input, response, reference):
        """
        Run all available metrics for a given sample.

        Args:
            user_input (str): The user input text.
            response (str): The response text.
            reference (str): The reference text.

        Returns:
            dict: A dictionary of metric names and their scores.
        """
        results = {}
        for metric_name, metric in self.metrics.items():
            sample = SingleTurnSample(
                user_input=user_input,
                response=response,
                reference=reference
            )
            results[metric_name] = metric.single_turn_score(sample)
        return results


if __name__ == "__main__":
    evaluator = NonLLMMetricsEvaluator()

    # Example 1: BLEU
    bleu_score = evaluator.evaluate(
        metric_name="BLEU",
        user_input="translate to French: Hello, world!",
        response="Bonjour, le monde!",
        reference="Bonjour le monde!"
    )
    print("BLEU:", bleu_score)

    # Example 2: ROUGE-L
    rouge_score = evaluator.evaluate(
        metric_name="ROUGE-L",
        user_input="summarize: …",
        response="Key points are A, B, and C.",
        reference="The summary should cover A, B, and C."
    )
    print("ROUGE-L:", rouge_score)

    # Example 3: ExactMatch
    exact_match_score = evaluator.evaluate(
        metric_name="ExactMatch",
        user_input="What is 2 + 2?",
        response="4",
        reference="4"
    )
    print("ExactMatch:", exact_match_score)

    # Example 4: Run all metrics
    all_scores = evaluator.run_all_metrics(
        user_input="translate to French: Hello, world!",
        response="Bonjour, le monde!",
        reference="Bonjour le monde!"
    )
    print("All Metrics:", all_scores)