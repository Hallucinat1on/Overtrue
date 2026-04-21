#!/usr/bin/env python3
"""Recursively collect building categories from summaryfiles."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def collect_categories(root_dir: Path) -> Counter:
    counter = Counter()
    for path in sorted(root_dir.rglob("summary.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Warning: failed to read {path}: {exc}")
            continue

        counts = data.get("building_category")
        if isinstance(counts, dict):
            for category, value in counts.items():
                if isinstance(category, str) and category.strip() and isinstance(value, (int, float)):
                    counter[category.strip()] += int(value)

    return counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursively collect building_categories from summary files and print statistics.")
    parser.add_argument("--input_dir", type=str, default="processed", help="Root directory")
    parser.add_argument("--top", type=int, default=0, help="Print only the top N categories by frequency")
    args = parser.parse_args()

    root_dir = Path(args.input_dir)
    if not root_dir.exists() or not root_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {root_dir}")

    counts = collect_categories(root_dir)
    if not counts:
        print("No building_categories found in any summary.json files.")
        return

    total = sum(counts.values())
    print(f"\nFound {len(counts)} unique building categories across {total} occurrences.\n")
    print(f"{'Category':<40} Count")
    print(f"{'-':<40} {'-'*5}")

    items = counts.most_common(args.top if args.top > 0 else None)
    for category, count in items:
        print(f"{category:<40} {count}")


if __name__ == "__main__":
    main()
