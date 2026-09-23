from src.llm.OpenaiProvider import OpenAIGenerationProvider
from src.embedding.OpenAIProvider import OpenAIEmbeddingProvider
from src.Vector_store.v_db import VectorDataBase
from src.graph.state import AgentState
from src.actions.report import write_report
from src.actions.email import send_email
import os


INTENTS = {"question", "report", "email"}


class AgentNodes:
    def __init__(self):
        self.provider = OpenAIGenerationProvider(max_turns=10)
        self.llm = self.provider.get_generation_provider()
        self.embeddings = OpenAIEmbeddingProvider()
        self.qdrant = VectorDataBase(vector_size=int(os.getenv("EMBEDDING_MODEL_DIM")))


    def intake_node(self, state: AgentState) -> dict:
        
        prompt = self.provider.prepare_query_for_agent(state["query"], intention=True)
        intent = self.llm.invoke(prompt).content.strip().lower()
        intent = intent if intent in INTENTS else "question"
        print(f"[intake] intent={intent}")
        
        return {"intent": intent}
        

    def retrieve_docs_node(self, state: AgentState) -> dict:
        ...  # -> {"doc_results": results}
        query_vector = self.embeddings.embed_query(query=state["query"])
        results = self.qdrant.search_embeddings(asset_id=state["asset_id"], query_vector=query_vector)
        print(f"[retrieve_docs] {len(results)} chunks")
        return {"doc_results": results}


    def summarize_node(self, state: AgentState) -> dict:
        
        prompt = self.provider.prepare_query_for_agent(state["query"], summary=True)
        summary = self.provider.generate_text(
            asset_id=state["asset_id"],
            query=prompt,
            search_results=state.get("doc_results", []),
        )
        print("[summarize] done")
        return {"summary": summary}
        

    def act_node(self, state: AgentState) -> dict:
        intent = state.get("intent", "question")
        summary = state.get("summary", "")
        result = {"report_path": None, "email_status": None}

        try:
            if intent == "report":
                result["report_path"] = write_report(state["query"], summary)
            elif intent == "email":
                result["email_status"] = send_email("Summary", summary)
        except Exception as e:
            result["email_status" if intent == "email" else "report_path"] = f"error: {e}"

        print(f"[act] intent={intent} -> {result}")
        
        return result
        

    def answer_node(self, state: AgentState) -> dict:
        answer = state.get("summary", "")

        report_path = state.get("report_path")
        email_status = state.get("email_status")

        if report_path:
            answer += f"\n\nReport saved: {report_path}"
        if email_status:
            answer += f"\n\nEmail: {email_status}"

        print("[answer] done")
        return {"answer": answer}
        
        
        