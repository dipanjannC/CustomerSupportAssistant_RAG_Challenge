# from transformers import AutoTokenizer, AutoModelForCausalLM

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

class MistralAIModel(BaseModel):
    model: str = "mistral-large-latest"
    temperature: float = 0
    top_p: Optional[float] = None
    random_seed: Optional[int] = None
    response_format: Optional[Dict[str, str]] = None
    safe_mode: bool = False
    safe_prompt: bool = False