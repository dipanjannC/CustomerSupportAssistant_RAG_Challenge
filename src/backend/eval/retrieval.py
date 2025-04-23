from src.backend.eval import ragas
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import FactualCorrectness, ResponseRelevancy, ContextPrecision
from ragas.llms.base import LangchainLLMWrapper
from src.backend.llm import LLMChatbot
import asyncio

class EvaluateRetrival:
    def __init__(self, retrieval):
        self.retrieval = retrieval

    def evaluate(self, query):
        """
        Evaluate the retrieval system with a given query.
        """
        # Placeholder for evaluation logic
        results = self.retrieval.retrieve(query)
        return results