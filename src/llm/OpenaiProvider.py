from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import List, Optional
import os

load_dotenv()


class OpenAIGenerationProvider:
    def __init__(self):
        self.chat_history = []  # list of (role, content) or Message objects

    def get_generation_provider(self):
        llm = ChatOpenAI(
            model=os.getenv("GENERATION_MODEL_ID"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.7,
        )
        return llm

    def prepare_query(self, query: str, search_results: List[dict], system_prompt: Optional[str] = None):
        """
        Builds the message list: system prompt + context from top search results
        + prior chat history + the new user query.
        """
        context_text = "\n\n".join(
            f"[{i+1}] {result['text']}" for i, result in enumerate(search_results)
        )

        default_system_prompt = (
            "You are a helpful assistant. Use the following context to answer "
            "the user's question. If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context_text}"
        )

        messages = [SystemMessage(content=system_prompt or default_system_prompt)]

        # add prior conversation turns
        messages.extend(self.chat_history)

        # add the new user query
        messages.append(HumanMessage(content=query))

        return messages

    def generate_text(self, query: str, search_results: List[dict], system_prompt: Optional[str] = None):
        llm = self.get_generation_provider()
        messages = self.prepare_query(query, search_results, system_prompt)

        response = llm.invoke(messages)

        # update chat history with this turn
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=response.content))

        return response.content

    def clear_history(self):
        self.chat_history = []