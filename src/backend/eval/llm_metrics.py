from ragas.metrics import FactualCorrectness,ResponseRelevancy,ContextPrecision
from ragas.metrics import LLMContextPrecisionWithoutReference,FaithfulnesswithHHEM
from ragas.metrics import AspectCritic
from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import LangchainLLMWrapper
from ragas.embeddings.base import BaseRagasEmbeddings

from src.backend.llm import LLMChatbot
from src.backend.embedder import SentenceTransformerEmbeddings
from src.backend.config.logger_config import setup_logging
import asyncio

logger = setup_logging()

class LLMMetricsEvaluator:
    """
    This class evaluates various metrics for a given sample using a language model (LLM).
    """
    def __init__(self):
        self.llm = LLMChatbot().get_mistral(model_name="ministral-8b-latest")
        self.embedder = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.evaluator_llm = LangchainLLMWrapper(self.llm)
        
        self.FACTUAL_CORRECTNESS_METRIC = FactualCorrectness(llm=self.evaluator_llm)
        self.RESPONSE_RELEVANCY_METRIC = ResponseRelevancy(llm=self.evaluator_llm, embeddings=self.embedder)
        self.CONTEXT_PRECISION_METRIC = ContextPrecision(llm=self.evaluator_llm)
        self.CONTEXT_PRECISION_WITHOUT_REFERENCE_METRIC= LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)
        self.FAITHFULNESS_WITHOUT_REFERENCE_METRIC = FaithfulnesswithHHEM(llm=self.evaluator_llm)

    async def get_factual_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the Factual Correctness score for a given sample.
        """
        result = self.FACTUAL_CORRECTNESS_METRIC.single_turn_score(sample).item()
        return result

    async def get_relevancy_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the Relevancy score for a given sample.
        """
        result = self.RESPONSE_RELEVANCY_METRIC.single_turn_score(sample).item()
        return result
    
    async def get_faithfulness_without_reference_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the Faithfulness score for a given sample.
        """
        result = self.FAITHFULNESS_WITHOUT_REFERENCE_METRIC.single_turn_score(sample)
        return result

    async def get_context_precision_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the Context Precision score for a given sample.
        """
        result =  self.CONTEXT_PRECISION_METRIC.single_turn_score(sample)
        return result
    
    async def context_precision_without_reference_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the Context Precision score for a given sample.
        """
        result =  self.CONTEXT_PRECISION_WITHOUT_REFERENCE_METRIC.single_turn_score(sample)
        return result
    
    async def report(self, sample: SingleTurnSample , with_golden: bool = False)->dict:
        """
        Report the scores for a given sample.
        """

        relevancy_score = await self.get_relevancy_score(sample)
        context_precision_without_precision_score = await self.context_precision_without_reference_score(sample)
        faithfullness_score = await self.get_faithfulness_without_reference_score(sample)

        if with_golden:
            context_precision_score = await self.get_context_precision_score(sample)
            factual_score = await self.get_factual_score(sample) # TODO : after adding golden
            
            return {
                "Factual Correctness": factual_score,
                "Relevancy": relevancy_score,
                "Context Precision": context_precision_score
            }
        
        else:
            return {
                "Relevancy": relevancy_score,
                "Context Precision": context_precision_without_precision_score,
                "Faithfulness": faithfullness_score
            }
        
    
    # TODO
    def get_toxicity_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the Toxicity score for a given sample.
        """
        # Placeholder for toxicity score calculation
        pass
    
    # TODO
    def get_malliciousness_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the Maliciousness score for a given sample.
        """
        scorer =  AspectCritic(
        name="maliciousness",
        definition="Is the submission intended to harm, deceive, or exploit users?",
        llm=evaluator_llm

        )


if __name__ == "__main__":
    # Example usage
    evaluator = LLMMetricsEvaluator()
    llm = LLMChatbot().get_mistral(model_name="ministral-8b-latest")
    evaluator_llm = LangchainLLMWrapper(llm)

    # Sample data
    sample = SingleTurnSample(
            user_input="When was the first super bowl?",
            response="The first superbowl was held on Jan 15, 1967",
            # reference= "The first superbowl was held on Feb 15, 1970",
            retrieved_contexts=[
                "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
            ]
        )

    # Run all metrics
    results = asyncio.run(evaluator.report(sample, with_golden=False))
    print(results)
