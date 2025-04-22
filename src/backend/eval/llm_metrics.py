from ragas.metrics import FactualCorrectness,ResponseRelevancy,ContextPrecision
from ragas.dataset_schema import SingleTurnSample
from ragas.llms.base import LangchainLLMWrapper
from src.backend.llm import LLMChatbot

import asyncio

sample = SingleTurnSample(
        user_input="When was the first super bowl?",
        response="The first superbowl was held on Jan 15, 1967",
        reference= "The first superbowl was held on Feb 15, 1970",
        retrieved_contexts=[
            "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
        ]
    )

llm = LLMChatbot().get_mistral(model_name="ministral-8b-latest")

# 2) Wrap it for Ragas
evaluator_llm = LangchainLLMWrapper(llm)


async def evaluate_factual_correctness():
    metric = FactualCorrectness(llm=evaluator_llm)
    result = await metric.single_turn_ascore(sample)
    print(f"Factual Correctness score: {result}")
    return result

# async def evaluate_relevancy_correctness():
#     metric = ResponseRelevancy(llm=evaluator_llm,embeddings=None)
#     result = await metric.single_turn_ascore(sample)
#     print(f"Faithfulness score: {result}")
#     return result

async def evaluate_contextual_precision():
    metric = ContextPrecision(llm=evaluator_llm)
    result = await metric.single_turn_ascore(sample)
    print(f"Evaluate Contextual Score: {result}")
    return result

# Run the evaluation
if __name__ == "__main__":
    asyncio.run(evaluate_factual_correctness())
    # asyncio.run(evaluate_relevancy_correctness())
    asyncio.run(evaluate_contextual_precision())