from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

import os
from dotenv import load_dotenv
load_dotenv()

class LLMChatbot:
    """
    Chatbot LLMs from langchain.

    """
    def __init__(self) -> None:
        pass

    def get_groq(self,model_name: str = None):
        """
        Initializing Groq Chat Model

        Args:
            model_name (str): _description_

        Returns:
            _type_: _description_
        """
        #model_name = "llama-3.1-8b-instant"
        #https://www.promptingguide.ai/models/llama-3
        
        llm = ChatGroq(
            model= model_name,
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        return llm
    
    
    def get_mistral(self, model_name: str = None):
        """
        Initializing Mistral Chat Model

        Args:
            model_name (str): _description_

        Returns:
            _type_: _description_
        """
        #model_name = "ministral-8b-latest"

        # https://www.promptingguide.ai/models/mistral-7b
        
        llm = ChatMistralAI(
            model=model_name,
            temperature=0.3,
            max_retries=2,
            max_tokens=256,
            api_key=os.getenv("MISTRAL_API_KEY"),
        )
        return llm