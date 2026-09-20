from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
from typing import List, Optional, Dict
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

_ROLE_TO_CLASS = {"system": SystemMessage, "human": HumanMessage, "ai": AIMessage}
_CLASS_TO_ROLE = {SystemMessage: "system", HumanMessage: "human", AIMessage: "ai"}


class OpenAIGenerationProvider:
    def __init__(self, max_turns: Optional[int] = None):
        """
        max_turns: number of non-system messages (Human/AI) to keep per asset.
                   The system message is rebuilt fresh every turn and doesn't count.
        """
        self.sessions: Dict[str, List[BaseMessage]] = {}  
        self.max_turns = max_turns

    # ------------------------------------------------------------------
    # History management (by asset_id)
    # ------------------------------------------------------------------
    def _get_history(self, asset_id: str) -> List[BaseMessage]:
        return self.sessions.setdefault(asset_id, [])

    def _trim(self, asset_id: str):
        if not self.max_turns:
            return
        history = self.sessions[asset_id]
        system_msgs = [m for m in history if isinstance(m, SystemMessage)]
        other_msgs = [m for m in history if not isinstance(m, SystemMessage)]
        self.sessions[asset_id] = system_msgs + other_msgs[-self.max_turns:]

    def load_history(self, asset_id: str, messages: List[BaseMessage]):
        """Restore an asset's history (e.g. loaded from Mongo) at session start."""
        self.sessions[asset_id] = messages

    def load_history_from_dicts(self, asset_id: str, history: List[dict]):
        """Convenience wrapper: restore from [{'role': ..., 'content': ...}, ...]."""
        restored = [_ROLE_TO_CLASS[m["role"]](content=m["content"]) for m in history]
        self.load_history(asset_id, restored)

    def export_history(self, asset_id: str) -> List[dict]:
        """Serialize an asset's history to plain dicts for storage (e.g. in Mongo)."""
        return [
            {"role": _CLASS_TO_ROLE[type(m)], "content": m.content}
            for m in self.sessions.get(asset_id, [])
        ]

    def clear(self, asset_id: str):
        self.sessions.pop(asset_id, None)

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
        asset_id: str,
        query: str,
        search_results: List[dict],
        system_prompt: Optional[str] = None,
    ) -> List[BaseMessage]:
        history = self._get_history(asset_id)


        history = [m for m in history if not isinstance(m, SystemMessage)]
        history.insert(0, SystemMessage(content=self.build_system_prompt(search_results, system_prompt)))
        history.append(HumanMessage(content=query))

        self.sessions[asset_id] = history
        self._trim(asset_id)
        return self.sessions[asset_id]

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
        messages = self.prepare_query(asset_id, query, search_results, system_prompt)

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
        self.sessions[asset_id].append(AIMessage(content=response.content))
        return response.content
    
    
    