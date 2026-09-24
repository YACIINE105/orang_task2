from typing import Any, TypedDict


class TravelState(TypedDict, total=False):
    destination: str
    budget: float
    currency: str
    interests: list[str]
    days: int
    destination_suggestions: list[dict[str, Any]]
    budget_breakdown: dict[str, Any]
    itinerary: list[dict[str, Any]]
    recommendations: list[str]
    evaluation: dict[str, Any]
    travel_plan: str
    report_path: str
