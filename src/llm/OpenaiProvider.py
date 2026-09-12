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
        context_text = "\n\n".join(
            f"[{i+1}] {result['text']}" for i, result in enumerate(search_results)
        )

        default_system_prompt = f"""You are a knowledgeable assistant that answers questions using ONLY the provided context below.

    Guidelines:
    - Base your answer strictly on the context. Do not use outside knowledge or make assumptions beyond what's given.
    - If the context is insufficient to answer the question, say so clearly instead of guessing.
    - Reference specific sources using their number (e.g. "[1]") when citing information.
    - Be concise but complete — prefer clear structure (bullet points, short paragraphs) over long unbroken text.
    - If multiple context passages disagree or are ambiguous, point that out rather than silently picking one.
    - Do not mention "the context" or "the provided text" explicitly in your answer — just answer naturally as if you know this.

    Context:
    {context_text}"""

        messages = [SystemMessage(content=system_prompt or default_system_prompt)]
        messages.extend(self.chat_history)
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