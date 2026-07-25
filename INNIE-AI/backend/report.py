"""
report.py — Generate performance reports from INNIE AI logs
Usage: python report.py [--event EVENT] [--format table|json|csv]
"""
import argparse
import json
import csv
import sys
from performance import PerformanceMonitor
from config import PERFORMANCE_LOG_PATH


def print_table(data: dict, title: str = ""):
    """Pretty-print a nested dict as an aligned table."""
    if title:
        print(f"\n{'='*60}")
        print(title.center(60))
        print('='*60)

    if not data:
        print("No data.")
        return

    for metric, stats in data.items():
        print(f"\n  {metric}:")
        for stat, val in stats.items():
            if stat == "count":
                print(f"    {stat:>10}: {val}")
            else:
                print(f"    {stat:>10}: {val:.6f}")
    print("="*60)


def export_csv(events: list, path: str):
    """Export raw events to CSV."""
    if not events:
        print("No events to export.")
        return

    # Flatten metrics into columns
    rows = []
    for e in events:
        row = {"timestamp": e.get("timestamp", ""), "event": e.get("event", "")}
        for k, v in e.get("metrics", {}).items():
            row[k] = v
        rows.append(row)

    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys, key=lambda x: (x not in ("timestamp", "event"), x))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description="INNIE AI Performance Reporter")
    parser.add_argument("--event", type=str, default=None, help="Filter by event name")
    parser.add_argument("--format", choices=["table", "json", "csv", "raw"], default="table")
    parser.add_argument("--output", type=str, default=None, help="Output file (for json/csv)")
    args = parser.parse_args()

    perf = PerformanceMonitor(log_path=PERFORMANCE_LOG_PATH)

    if args.format == "raw":
        events = perf.load_history()
        if args.event:
            events = [e for e in events if e.get("event") == args.event]
        for e in events:
            print(json.dumps(e))
        return

    if args.format == "table":
        summary = perf.summarize(event_filter=args.event)
        print_table(summary, title=f"Summary: {args.event or 'ALL EVENTS'}")
        return

    if args.format == "json":
        summary = perf.summarize(event_filter=args.event)
        out = json.dumps(summary, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out)
            print(f"Saved to {args.output}")
        else:
            print(out)
        return

    if args.format == "csv":
        events = perf.load_history()
        if args.event:
            events = [e for e in events if e.get("event") == args.event]
        export_csv(events, args.output or "performance.csv")


if __name__ == "__main__":
    main()
