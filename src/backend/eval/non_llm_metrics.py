

from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore,RougeScore

from ragas.metrics import NonLLMContextRecall


class NonLLMMetricsEvaluator:
    def __init__(self):
        self.BLEU_METRIC = BleuScore()  
        self.ROUGEL_METRIC = RougeScore(rouge_type="rougeL")  
        self.NONLLM_CONTEXT_METRIC = NonLLMContextRecall()
    
    def get_bleu_score(self,sample: SingleTurnSample)->float:   
        """
        Calculate the BLEU score for a given sample.
        """
        # Important Metric as it used to understand recall of LLM generated
        result = self.BLEU_METRIC.single_turn_score(sample)
        return result
    
    def get_rouge_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the ROUGE-L score for a given sample.
        """
        # Important Metric as it used to understand recall of LLM generated
        result = self.ROUGEL_METRIC.single_turn_score(sample)
        return result
    
    def get_nonllm_context_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the Non-LLM Context Recall score for a given sample.
        """
        # Important Metric as it used to understand recall of LLM generated
        result = self.NONLLM_CONTEXT_METRIC.single_turn_score(sample)
        return result
    
    def get_bert_similarity_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the BERT similarity score for a given sample.
        """
        # Important Metric as it used to understand recall of LLM generated
        pass


    def report(self, sample: SingleTurnSample):
        """
        Report the scores for a given sample.
        """
        bleu_score = self.get_bleu_score(sample)
        rouge_score = self.get_rouge_score(sample)
        nonllm_context_score = self.get_nonllm_context_score(sample)

        print(f"BLEU Score: {bleu_score}")
        print(f"ROUGE-L Score: {rouge_score}")
        print(f"Non-LLM Context Recall Score: {nonllm_context_score}")
        return {
            "BLEU": bleu_score,
            "ROUGE-L": rouge_score,
            "Non-LLM Context Recall": nonllm_context_score
        }


if __name__ == "__main__":

    # Wrap your sample
    sample = SingleTurnSample(
        user_input="translate to French: Hello, world!",
        response="Bonjour, le monde!",
        reference="Bonjour le monde!"
    )
    
    evaluator = NonLLMMetricsEvaluator()
    results = evaluator.report(sample)
    print(results)