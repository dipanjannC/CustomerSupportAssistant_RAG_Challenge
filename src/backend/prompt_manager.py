from langchain_core.prompts import ChatPromptTemplate
from src.backend.util.code_util import project_root


from langchain_core.prompts import ChatPromptTemplate

# https://huggingface.co/docs/transformers/main/en/chat_templating


class CustomerAssistantPrompt:
    """

    The function initializes prompts for a Customer Support Assistant
    to help in answering customer queries.

    # Roles and Uses for each prompt:

    The system prompt defines role, tone, edge‑case rules;
    the retriever prompt instructs how to use fetched context;
    the user prompt structures Question→Answer.
    """

    def __init__(self) -> None:
        self.ROLE_PROMPT = """ 
You are an  customer support assistant trained on a large corpus of real-world support tweets. 
You speak directly to the customer in a friendly, empathetic, and professional tone—always in second person. 
You should:
  • Understand and restate the issue clearly before proposing solutions. 
  • If the users query is ambiguous or missing details, ask a clarifying question. 
  • For off topic or out of scope requests, politely apologize and escalate 
    - example  I’m sorry, I don’t have that info—let me connect you to a human agent 
  • Never request sensitive personal data (full credit‑card numbers, SSNs, etc.) only ask for 
  minimal info needed to assist.
"""
        self.RETRIEVER_PROMPT = """
You will be provided with a list of retrieved documents that are relevant to the users question.
When forming your answer, explicitly cite these by prefacing with “Based on our records…” 
and weave the fetched details into your solution.
{context}

"""
        self.USER_PROMPT ="""
[INST] Question: {question} [/INST]

Answer:
"""

    def get_chat_template(self) -> ChatPromptTemplate:
        """
        Constructs a LangChain ChatPromptTemplate that sequences:
        1) system instructions,
        2) retriever instructions,
        3) past chat history,
        4) the user’s current question.

        Returns:
            ChatPromptTemplate: ChatPromptTemplate object for the chat model.
        """
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.ROLE_PROMPT),
                ("assistant", self.RETRIEVER_PROMPT),
                ("placeholder", "{chat_history}"),
                ("user", self.USER_PROMPT),
                # ("placeholder", "{agent_scratchpad}"),
            ]
        )
        return chat_prompt


if __name__ == "__main__":
    pass
