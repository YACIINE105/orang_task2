from typing import Any

from src.travel.state import TravelState


class DestinationAgent:
    _catalog = {
        "culture": ("Old Town", "museum and heritage walk"),
        "history": ("Historic Quarter", "guided landmarks tour"),
        "food": ("Central Market", "local food tasting"),
        "nature": ("Riverside Park", "scenic nature walk"),
        "adventure": ("Hill District", "outdoor viewpoint trek"),
        "beach": ("Coastal Promenade", "beach and sunset visit"),
        "art": ("Arts District", "gallery and studio visit"),
        "shopping": ("Craft Market", "local shopping experience"),
        "relaxation": ("Botanical Gardens", "slow morning and wellness time"),
    }

    def run(self, state: TravelState) -> dict[str, Any]:
        interests = state.get("interests", []) or ["culture"]
        suggestions = []
        for interest in interests:
            place, activity = self._catalog.get(
                interest.lower(), ("City Center", f"{interest} experience")
            )
            suggestions.append({"interest": interest, "place": place, "activity": activity})
        return {"destination_suggestions": suggestions}


class BudgetAgent:
    def run(self, state: TravelState) -> dict[str, Any]:
        days = max(1, state.get("days", 1))
        total_budget = max(0.0, state.get("budget", 0.0))
        daily_budget = total_budget / days if days else 0.0
        breakdown = {
            "accommodation": round(total_budget * 0.35, 2),
            "food": round(total_budget * 0.20, 2),
            "transportation": round(total_budget * 0.15, 2),
            "activities": round(total_budget * 0.20, 2),
            "buffer": round(total_budget * 0.10, 2),
        }
        return {
            "budget_breakdown": {
                "currency": state.get("currency", "USD"),
                "total": round(total_budget, 2),
                "daily_average": round(daily_budget, 2),
                "categories": breakdown,
            }
        }


class ItineraryAgent:
    def run(self, state: TravelState) -> dict[str, Any]:
        suggestions = state.get("destination_suggestions", [])
        days = max(1, state.get("days", 1))
        itinerary = []
        for day in range(1, days + 1):
            suggestion = suggestions[(day - 1) % len(suggestions)] if suggestions else {
                "place": "City Center",
                "activity": "neighborhood walk",
            }
            itinerary.append(
                {
                    "day": day,
                    "morning": f"Explore {suggestion['place']}",
                    "afternoon": suggestion["activity"].capitalize(),
                    "evening": "Dinner near the main evening district",
                }
            )
        return {"itinerary": itinerary}


class RecommendationAgent:
    def run(self, state: TravelState) -> dict[str, Any]:
        suggestions = state.get("destination_suggestions", [])
        budget = state.get("budget_breakdown", {})
        recommendations = [
            f"Prioritize {item['place']} for {item['interest']}."
            for item in suggestions
        ]
        recommendations.append(
            f"Keep the average daily spend near {budget.get('currency', 'USD')} "
            f"{budget.get('daily_average', 0):.2f}."
        )
        recommendations.extend(
            [
                "Reserve accommodation and any timed attractions before departure.",
                "Keep the buffer category available for weather or transport changes.",
            ]
        )
        return {"recommendations": recommendations}


class EvaluatorAgent:
    def run(self, state: TravelState) -> dict[str, Any]:
        total = state.get("budget_breakdown", {}).get("total", 0)
        return {
            "evaluation": {
                "budget_ok": total > 0,
                "days_ok": state.get("days", 0) > 0,
                "interests_covered": len(state.get("destination_suggestions", [])) > 0,
            }
        }


def format_currency(state: TravelState, amount: float) -> str:
    return f"{state.get('currency', 'USD')} {amount:,.2f}"


def render_travel_plan(state: TravelState) -> str:
    budget = state["budget_breakdown"]
    lines = [
        f"# Travel Plan: {state['destination']}",
        "",
        "## Recommended Places & Activities",
    ]
    lines.extend(
        f"- **{item['place']}** ({item['interest']}): {item['activity']}"
        for item in state.get("destination_suggestions", [])
    )
    lines.extend(["", "## Estimated Budget", f"- Total: {format_currency(state, budget['total'])}", f"- Daily average: {format_currency(state, budget['daily_average'])}"])
    lines.extend(f"- {category.title()}: {format_currency(state, amount)}" for category, amount in budget["categories"].items())
    lines.extend(["", "## Day-by-Day Itinerary"])
    for day in state.get("itinerary", []):
        lines.extend([f"### Day {day['day']}", f"- Morning: {day['morning']}", f"- Afternoon: {day['afternoon']}", f"- Evening: {day['evening']}"])
    lines.extend(["", "## Final Recommendations"])
    lines.extend(f"- {recommendation}" for recommendation in state.get("recommendations", []))
    lines.extend(["", "## Plan Evaluation"])
    lines.extend(f"- {key.replace('_', ' ').title()}: {'yes' if value else 'no'}" for key, value in state.get("evaluation", {}).items())
    return "\n".join(lines) + "\n"