from llm_metrics import LLMMetricsEvaluator
from non_llm_metrics import NonLLMMetricsEvaluator
from ragas.dataset_schema import SingleTurnSample
import asyncio


from src.backend.vectorstore import Vectorstore
from src.backend.rag import RAG_base
from src.backend.preprocess.parser import sample_parser_for_evaluation
from src.backend.config.logger_config import setup_logging

logger = setup_logging()


class Evaluator:
    def __init__(self):
        self.llm_metrics_evaluator = LLMMetricsEvaluator()
        self.non_llm_metrics_evaluator = NonLLMMetricsEvaluator()

    async def get_eval_report(self, sample : SingleTurnSample):
        """
        Evaluate the retrieval system with a given query.

        Args:
            sample (SingleTurnSample): The sample to evaluate.
        Returns:
            dict: The evaluation results.
        """
        
        results = await self.llm_metrics_evaluator.report(sample)
        results.update(self.non_llm_metrics_evaluator.report(sample))

        return results
    
    async def evaluate_generation(self, sample: SingleTurnSample):
        """
        Evaluate the generation system with a given query.

        Args:
            sample (SingleTurnSample): The sample to evaluate.
        Returns:
            dict: The evaluation results.
        """
        
        #TODO
        pass

    async def evaluate_retrieval(self, sample: SingleTurnSample):
        """
        Evaluate the retrieval system with a given query.

        Args:
            sample (SingleTurnSample): The sample to evaluate.
        Returns:
            dict: The evaluation results.
        """
        
        #TODO
        pass
    


if __name__ == "__main__":

    # Example usage
    vectorstore = Vectorstore(collection_name="customer_support")  # Replace with actual vectorstore initialization
    evaluator = Evaluator()
    
    query = "What is the capital of France?"
    retriever_results = vectorstore.query(query)
    llm_response = RAG_base(query_text=query,retriever=vectorstore)
    eval_sample = sample_parser_for_evaluation(query=query,retriever_results=retriever_results,llm_response=llm_response)

    # Evaluate the sample
    evaluation_results = asyncio.run(evaluator.get_eval_report(eval_sample))
    logger.info(f"Query: {query}")
    logger.info(f"Retriever Results: {retriever_results}")
    logger.info(f"LLM Response: {llm_response}")
    logger.info(f"Evaluation Results:\n{evaluation_results}")
    