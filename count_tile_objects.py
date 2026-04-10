#!/usr/bin/env python3
"""Count building and road objects in tile JSON files.

Usage:
python count_tile_objects.py --input_dir processed/beijing_haidian
python count_tile_objects.py --input_dir processed/newyork_manhattan
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def count_objects_in_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    buildings = data.get("buildings")
    if not isinstance(buildings, list):
        buildings = []
    road_samples = data.get("road_samples")
    if not isinstance(road_samples, list):
        road_samples = []

    road_point_count = 0
    for road in road_samples:
        pos = road.get("positions")
        if isinstance(pos, list):
            road_point_count += len(pos)

    return {
        "file": str(path),
        "building_count": len(buildings),
        "road_count": len(road_samples),
        "road_point_count": road_point_count,
        "total_count": len(buildings) + len(road_samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Count building and road objects in tile JSON files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing tile JSON files")
    parser.add_argument("--top", type=int, default=8, help="Number of top files to print")
    parser.add_argument("--pattern", type=str, default="tile_*.json", help="Filename pattern to match JSON files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    records = []
    for path in sorted(input_dir.glob(args.pattern)):
        if not path.is_file():
            continue
        try:
            rec = count_objects_in_file(path)
            records.append(rec)
        except Exception as e:
            print(f"Failed to parse {path}: {e}")

    if not records:
        print("No matching JSON files found.")
        return

    top_buildings = sorted(records, key=lambda r: r["building_count"], reverse=True)[: args.top]
    top_roads = sorted(records, key=lambda r: r["road_count"], reverse=True)[: args.top]
    top_road_points = sorted(records, key=lambda r: r["road_point_count"], reverse=True)[: args.top]

    print(f"Top {args.top} files by building count:")
    print("file    building_count    road_count    road_point_count")
    for rec in top_buildings:
        print(f"{rec['file']}, {rec['building_count']}, {rec['road_count']}, {rec['road_point_count']}")

    print()
    print(f"Top {args.top} files by road object count:")
    print("file    road_count    building_count    road_point_count")
    for rec in top_roads:
        print(f"{rec['file']}, {rec['road_count']}, {rec['building_count']}, {rec['road_point_count']}")

    print()
    print(f"Top {args.top} files by road sample point count:")
    print("file    road_point_count    road_count    building_count")
    for rec in top_road_points:
        print(f"{rec['file']}, {rec['road_point_count']}, {rec['road_count']}, {rec['building_count']}")


if __name__ == "__main__":
    main()
