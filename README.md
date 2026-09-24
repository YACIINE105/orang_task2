# Multi-Agent Travel Planner

This project builds a travel plan through collaborating specialized agents and automatically saves the final plan as a Markdown report.

## Agents

1. Destination Agent selects places and activities for each interest.
2. Budget Agent estimates accommodation, food, transportation, activities, and a buffer.
3. Itinerary Agent creates a day-by-day schedule from the destination and budget outputs.
4. Recommendation Agent reviews the collected plan and adds practical recommendations.
5. Evaluator Agent checks that the plan has a valid budget, duration, and covered interests.

The agents run in order through a LangGraph state graph. The automation node receives the composed final plan and writes it to `reports/travel_plan_<timestamp>.md`.

## Run

```bash
uv sync
uv run python -m src.main_graph \
  --destination Lisbon \
  --budget 1200 \
  --interests culture,food,nature \
  --days 3 \
  --currency EUR
```

The command prints the complete plan and the automated save result.

## Input

- `--destination`: destination name.
- `--budget`: total budget as a non-negative number.
- `--interests`: comma-separated interests.
- `--days`: positive number of available days.
- `--currency`: currency label, defaulting to `USD`.
