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
  * Understand and restate the issue clearly before proposing solutions.
  * Think step-by-step through the customer's problem before responding.
  * Explicitly explain your reasoning when offering solutions.
  * If the users query is ambiguous or missing details, ask a clarifying question.
  * For off topic or out of scope requests, politely apologize and escalate
    - example "I'm sorry, I don't have that info—let me connect you to a human agent"
  * Never request sensitive personal data (full credit‑card numbers, SSNs, etc.) only ask for
  minimal info needed to assist.

When providing solutions:
  * Break down complex problems into logical steps.
  * Explain why you're recommending specific actions.
  * When possible, give the reasoning behind your suggestions.
  * If there are multiple possible issues, explain why you're prioritizing one diagnosis over others.


"""
        self.RETRIEVER_PROMPT = """
You will be provided with a list of retrieved documents that are relevant to the users question.
When forming your answer:
  * First analyze the retrieved information and identify the most relevant details.
  * Explicitly cite these by prefacing with "Based on our records…" 
  * Explain how the retrieved information connects to the user's specific situation.
  * Show your reasoning for why certain retrieved information is applicable to their case.
  * Weave the fetched details into your solution with clear explanations.


{context}

"""

        self.USER_PROMPT = """
[INST] Question: {question} [/INST]

Answer:
"""

    def get_chat_template(
        self, has_context: bool = False, has_history: bool = False
    ) -> ChatPromptTemplate:
        """
        Constructs a LangChain ChatPromptTemplate that sequences:
        1) system instructions with reasoning guidelines,
        2) retriever instructions if context is available,
        3) past chat history if available,
        4) the user's current question.

        Args:
            has_context (bool): Whether retrieved context is available
            has_history (bool): Whether chat history is available

        Returns:
            ChatPromptTemplate: ChatPromptTemplate object for the chat model.
        """
        messages = [
            ("system", self.ROLE_PROMPT),
        ]

        if has_context:
            messages.append(("system", self.RETRIEVER_PROMPT))

        if has_history:
            messages.append(("human", "{chat_history}"))

        messages.append(("user", self.USER_PROMPT))

        chat_prompt = ChatPromptTemplate.from_messages(messages)
        return chat_prompt

    def get_full_template(self) -> ChatPromptTemplate:
        """
        Returns a ChatPromptTemplate with all components included,
        with appropriate variable placeholders that can be filled
        conditionally at runtime.

        Returns:
            ChatPromptTemplate: Complete template with optional components
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.ROLE_PROMPT),
                # Context is optional
                ("system", self.RETRIEVER_PROMPT),
                # Chat history is optional
                ("human", "{chat_history}"),
                ("user", self.USER_PROMPT),
            ]
        )


if __name__ == "__main__":
    pass
