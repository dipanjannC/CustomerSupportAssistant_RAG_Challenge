from ragas.dataset_schema import SingleTurnSample
# from ragas.metrics import BleuScore,RougeScore
from ragas.metrics import NonLLMContextRecall

from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer

from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.meteor_score import meteor_score

from src.backend.embedder import SentenceTransformer
from src.backend.config.logger_config import setup_logging
logger = setup_logging()

class NonLLMMetricsEvaluator:
    def __init__(self):
        # self.BLEU_METRIC = BleuScore() 
        # self.ROUGEL_METRIC =  RougeScore() 
        self.NONLLM_CONTEXT_METRIC = NonLLMContextRecall()
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        

    
    def get_bleu_score(self,sample: SingleTurnSample)->float:   
        """
        Calculate the BLEU score for a given sample.
        """
        # Important Metric as it used to understand recall of LLM generated
        # result = self.BLEU_METRIC.single_turn_score(sample)
        result = corpus_bleu(references=[sample.response], hypothesis=sample.user_input)
        return result
    
    def get_rouge_score(self,sample: SingleTurnSample)->dict:
        """
        Calculate the ROUGE-L score for a given sample.
        """
        # result = self.ROUGEL_METRIC.single_turn_score(sample)
        result = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True).score(
            sample.user_input,
            sample.response
        )
        return result
    
    def get_nonllm_context_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the Non-LLM Context Recall score for a given sample.
        """
        # Important Metric as it used to understand recall of LLM generated
        result = self.NONLLM_CONTEXT_METRIC.single_turn_score(sample)
        return result
    
    def get_meteor_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the METEOR score for a given sample.

        Reference (List): The actual text/ground truth. If there are multiple people 
        generating the ground truth for same datapoint you will have multiple 
        references and all of them are assumed to be correct

        hypothesis (str): The candidate/predicted.

        But here we are considering the retrieved context as the reference and the generated response
        as the hypothesis.


        """
        return meteor_score(references= sample.retrieved_contexts,hypothesis=sample.response)
    
    def get_cosine_similarity_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the cosine similarity score for a given sample.
        """
        embeddings = self.embedder.encode([sample.response, sample.user_input])
        return cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))[0][0].item()
        


    # TODO
    def get_bert_similarity_score(self,sample: SingleTurnSample)->float:
        """
        Calculate the BERT similarity score for a given sample.
        """
        # Important Metric as it used to understand recall of LLM generated
        pass


    def report(self, sample: SingleTurnSample, with_golden: bool = False)->dict:
        """
        Report the scores for a given sample.
        """
        
        rouge_score = self.get_rouge_score(sample)
        cosine_similarity_score = self.get_cosine_similarity_score(sample)
        #TODO : Fix enum error 
        # meteor_score = self.get_meteor_score(sample)
        
        if with_golden:
            bleu_score = self.get_bleu_score(sample)
            nonllm_context_score = self.get_nonllm_context_score(sample)
        
            return {
                "BLEU": bleu_score,
                "ROUGE-L": rouge_score,
                # "Non-LLM Context Recall": nonllm_context_score,
                # "METEOR": meteor_score,
                "Cosine Similarity": cosine_similarity_score
            }
        
        return {
            "ROUGE-L": rouge_score,
            # "METEOR": meteor_score,
            "Cosine Similarity": cosine_similarity_score
        }


if __name__ == "__main__":

    # sample = SingleTurnSample(
    #     user_input="translate to French: Hello, world!",
    #     response="Bonjour, le monde!",
    #     reference="Bonjour le mondeleras asd we uwant!",
    #     reference_contexts=[
    #         "Bonjour le monde!",
    #         "Bonjour le monde! Comment ça va?",
    #         "Bonjour, le monde! let ça va?"
    #     ],
    #     retrieved_contexts=[
    #         "Bonjour le monde!",
    #         "Bonjour, what is that we now?",
    #         "Bonjour, How are you?"
    #     ]

    # )

    # Example usage
    sample = SingleTurnSample(
            user_input="When was the first super bowl?",
            response="The first superbowl was held on Jan 15, 1967",
            reference= "The first superbowl was held on Feb 15, 1970",
            retrieved_contexts=[
                "The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles."
            ],
            # reference_contexts=[
            #     "The first superbowl was held on Feb 15, 1970",
            #     "The first superbowl was held on Jan 15, 1967",
            #     "The first superbowl was held on Jan 15, 1967"
            # ]
        )

    evaluator = NonLLMMetricsEvaluator()
    results = evaluator.report(sample)
    logger.info(f"NON-LLM Metrics: {results}")
   