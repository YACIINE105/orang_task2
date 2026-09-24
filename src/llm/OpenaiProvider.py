from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from typing import List, Optional
import os

load_dotenv()


DEFAULT_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions using ONLY the provided context below.

Guidelines:
- Base your answer strictly on the context. Do not use outside knowledge or make assumptions beyond what's given.
- If the context is insufficient to answer the question, say so clearly instead of guessing.
- Reference specific sources using their number (e.g. "[1]") when citing information.
- Be concise but complete — prefer clear structure (bullet points, short paragraphs) over long unbroken text.
- If multiple context passages disagree or are ambiguous, point that out rather than silently picking one.
- Do not mention "the context" or "the provided text" explicitly in your answer — just answer naturally as if you know this."""

class OpenAIGenerationProvider:
    # ------------------------------------------------------------------
    # LLM provider
    # ------------------------------------------------------------------
    def get_generation_provider(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=os.getenv("GENERATION_MODEL_ID"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.5,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def build_system_prompt(self, search_results: List[dict], system_prompt: Optional[str] = None) -> str:
        context_text = "\n\n".join(
            f"[{i + 1}] {result['text']}" for i, result in enumerate(search_results)
        )
        base = system_prompt or DEFAULT_SYSTEM_PROMPT
        return f"{base}\n\nContext:\n{context_text}"

    def prepare_query(
        self,
        query: str,
        search_results: List[dict],
        system_prompt: Optional[str] = None,
    ) -> list:
        return [
            SystemMessage(content=self.build_system_prompt(search_results, system_prompt)),
            HumanMessage(content=query),
        ]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_text(
        self,
        asset_id: str,
        query: str,
        search_results: List[dict],
        system_prompt: Optional[str] = None,
        verbose: bool = False,
    ) -> str:
        llm = self.get_generation_provider()
        messages = self.prepare_query(query, search_results, system_prompt)

        if verbose:
            context_text = "\n\n".join(
                f"[{i + 1}] {r['text']}" for i, r in enumerate(search_results)
            )
            print(
                "\n\n####################################\n\n",
                context_text,
                "\n\n####################################\n\n",
            )

        response = llm.invoke(messages)
        return response.content
    
    
    