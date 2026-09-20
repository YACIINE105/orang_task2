from langgraph.graph import StateGraph, START, END
from graph.state import AgentState
from graph.nodes import AgentNodes

def build_graph(nodes=None):
    nodes = nodes or AgentNodes()
    g = StateGraph(AgentState)
    g.add_node("intake", nodes.intake_node)
    g.add_node("retrieve_docs", nodes.retrieve_docs_node)
    g.add_node("summarize", nodes.summarize_node)
    g.add_node("act", nodes.act_node)
    g.add_node("answer", nodes.answer_node)

    g.add_edge(START, "intake")
    g.add_edge("intake", "retrieve_docs")
    g.add_edge("retrieve_docs", "summarize")
    g.add_edge("summarize", "act")
    g.add_edge("act", "answer")
    g.add_edge("answer", END)
    return g.compile()

