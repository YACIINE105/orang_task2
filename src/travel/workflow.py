from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.travel.agents import (
    BudgetAgent,
    DestinationAgent,
    EvaluatorAgent,
    ItineraryAgent,
    RecommendationAgent,
    render_travel_plan,
)
from src.travel.state import TravelState


def save_travel_plan(state: TravelState) -> dict[str, str]:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"travel_plan_{stamp}.md"
    path.write_text(state["travel_plan"], encoding="utf-8")
    return {"report_path": str(path)}


def build_travel_graph():
    destination_agent = DestinationAgent()
    budget_agent = BudgetAgent()
    itinerary_agent = ItineraryAgent()
    recommendation_agent = RecommendationAgent()
    evaluator_agent = EvaluatorAgent()

    graph = StateGraph(TravelState)
    graph.add_node("destination", destination_agent.run)
    graph.add_node("budget", budget_agent.run)
    graph.add_node("itinerary", itinerary_agent.run)
    graph.add_node("recommendation", recommendation_agent.run)
    graph.add_node("evaluation", evaluator_agent.run)
    graph.add_node("compose", lambda state: {"travel_plan": render_travel_plan(state)})
    graph.add_node("automate", save_travel_plan)
    graph.add_edge(START, "destination")
    graph.add_edge("destination", "budget")
    graph.add_edge("budget", "itinerary")
    graph.add_edge("itinerary", "recommendation")
    graph.add_edge("recommendation", "evaluation")
    graph.add_edge("evaluation", "compose")
    graph.add_edge("compose", "automate")
    graph.add_edge("automate", END)
    return graph.compile()


def create_travel_plan(
    destination: str,
    budget: float,
    interests: list[str],
    days: int,
    currency: str = "USD",
) -> dict[str, Any]:
    state: TravelState = {
        "destination": destination,
        "budget": budget,
        "currency": currency.upper(),
        "interests": interests,
        "days": days,
    }
    return build_travel_graph().invoke(state)