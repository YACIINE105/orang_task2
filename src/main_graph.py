import argparse

from src.travel.workflow import create_travel_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and save a multi-agent travel plan.")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--budget", required=True, type=float)
    parser.add_argument("--interests", required=True, help="Comma-separated interests")
    parser.add_argument("--days", required=True, type=int)
    parser.add_argument("--currency", default="USD")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.days < 1 or args.budget < 0:
        raise SystemExit("days must be positive and budget cannot be negative")
    result = create_travel_plan(
        destination=args.destination,
        budget=args.budget,
        interests=[item.strip() for item in args.interests.split(",") if item.strip()],
        days=args.days,
        currency=args.currency,
    )
    print(result["travel_plan"])
    print(f"Automated Action Result: saved to {result['report_path']}")


if __name__ == "__main__":
    main()
    
    