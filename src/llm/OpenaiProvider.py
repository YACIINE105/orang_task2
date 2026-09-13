from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
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
    def __init__(self, max_turns: Optional[int] = None):
        """
        max_turns: number of non-system messages (Human/AI) to keep.
                   The system message is always kept separately and rebuilt each turn.
        """
        self.history: List[BaseMessage] = []
        self.max_turns = max_turns

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------
    def add(self, message: BaseMessage):
        self.history.append(message)
        if self.max_turns:
            system_msgs = [m for m in self.history if isinstance(m, SystemMessage)]
            other_msgs = [m for m in self.history if not isinstance(m, SystemMessage)]
            other_msgs = other_msgs[-self.max_turns:]
            self.history = system_msgs + other_msgs

    def clear(self):
        self.history = []

    def get_context(self) -> str:
        """Human-readable dump of the current history (for debugging/logging)."""
        lines = []
        for m in self.history:
            role = m.__class__.__name__.replace("Message", "")
            lines.append(f"{role}: {m.content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM provider
    # ------------------------------------------------------------------
    def get_generation_provider(self) -> ChatOpenAI:
        llm = ChatOpenAI(
            model=os.getenv("GENERATION_MODEL_ID"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            temperature=0.5,
        )
        return llm

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
    ) -> List[BaseMessage]:
        # Remove any existing system message(s) and reinsert a fresh one
        # so the model always sees the latest retrieved context, not stale
        # context from a previous turn.
        self.history = [m for m in self.history if not isinstance(m, SystemMessage)]
        fresh_system = SystemMessage(content=self.build_system_prompt(search_results, system_prompt))
        self.history.insert(0, fresh_system)

        # Add the user's new question
        self.add(HumanMessage(content=query))

        return self.history

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate_text(
        self,
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

        # Record the assistant's reply in history
        self.add(AIMessage(content=response.content))

        return response.content
    
    