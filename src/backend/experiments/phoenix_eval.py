# from src.backend.model import MistralAIModel
from src.backend.llm import LLMChatbot
from src.backend.config.logger_config import setup_logging

from src.backend.utilities.code_util import project_root
import os

import pandas as pd

from phoenix.evals import (
    RAG_RELEVANCY_PROMPT_RAILS_MAP,
    RAG_RELEVANCY_PROMPT_TEMPLATE,
    download_benchmark_dataset,
    llm_classify,
)

llm = LLMChatbot().get_mistral(model_name="ministral-8b-latest")

eval_dataset_path = os.path.join(project_root,"data/eval/binary-relevance-classification.json")

df = pd.read_json(eval_dataset_path,lines=False)
df.head()

# #Will ensure the binary value expected from the template is returned
rails = list(RAG_RELEVANCY_PROMPT_RAILS_MAP.values())

relevance_classifications = llm_classify(
    dataframe=df,
    template=RAG_RELEVANCY_PROMPT_TEMPLATE,
    model=llm,
    rails=rails,
    provide_explanation=True, #optional to generate explanations for the value produced by the eval LLM
)
