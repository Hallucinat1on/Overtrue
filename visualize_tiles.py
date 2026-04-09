#!/usr/bin/env python3
"""
Visualize a tile JSON file created by split_building_blocks.py and save an overhead view.

example:
python visualize_tiles.py \
  --input processed/newyork_manhattan/tile_x43_y63.json \
  --output processed/newyork_manhattan/tile_x43_y63.png

python visualize_tiles.py \
  --input_dir processed/newyork_manhattan \
  --output_dir processed/newyork_manhattan
"""

import argparse
import json
import math
import random
from hashlib import sha1
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch


def _build_rectangle(center_x, center_y, length, width, rotation_deg):
    half_l = length / 2.0
    half_w = width / 2.0
    coords = [
        (-half_l, -half_w),
        (half_l, -half_w),
        (half_l, half_w),
        (-half_l, half_w),
    ]
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    out = []
    for x, y in coords:
        xr = x * cos_t - y * sin_t + center_x
        yr = x * sin_t + y * cos_t + center_y
        out.append((xr, yr))
    return out


def _type_label(type_field):
    if isinstance(type_field, dict):
        if type_field.get("subtype"):
            return type_field["subtype"]
        if type_field.get("class"):
            return type_field["class"]
        if type_field.get("object_category"):
            return type_field["object_category"]
        if type_field.get("source_type"):
            return type_field["source_type"]
        if type_field.get("highway"):
            return type_field["highway"]
        if type_field.get("building_levels") is not None:
            return f"levels_{type_field['building_levels']}"
        if type_field.get("height") is not None:
            return f"height_{type_field['height']}"
        return "unknown"
    if type_field is None:
        return "unknown"
    return str(type_field)


def _random_color(rng_or_seed) -> tuple:
    if not isinstance(rng_or_seed, random.Random):
        rng = random.Random(rng_or_seed)
    else:
        rng = rng_or_seed
    h = rng.random()
    s = 0.6 + rng.random() * 0.3
    v = 0.7 + rng.random() * 0.2
    color = plt.cm.hsv(h)
    return (color[0], color[1], color[2])


def _build_road_line(points: List[List[int]]) -> List[tuple]:
    return [(p[0], p[1]) for p in points]


def _build_type_colors(labels: List[str], seed: Optional[int] = None) -> Dict[str, tuple]:
    rng = random.Random(seed)
    colors = {}
    for label in sorted(labels):
        colors[label] = _random_color(rng)
    return colors


def visualize_tile(input_path: Path, output_path: Path, dpi: int = 200, type_colors: Optional[Dict[str, tuple]] = None):
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    buildings = data.get("buildings", [])
    road_samples = data.get("road_samples", [])
    if not buildings and not road_samples:
        raise ValueError("Tile JSON has no buildings or road samples")

    xs = []
    ys = []
    for b in buildings:
        xs.extend([b["position"][0] - b["bbox"]["length"] / 2.0, b["position"][0] + b["bbox"]["length"] / 2.0])
        ys.extend([b["position"][1] - b["bbox"]["width"] / 2.0, b["position"][1] + b["bbox"]["width"] / 2.0])
    for r in road_samples:
        for point in r.get("positions", []):
            xs.append(point[0])
            ys.append(point[1])

    minx = min(xs)
    miny = min(ys)
    maxx = max(xs)
    maxy = max(ys)

    if type_colors is None:
        labels = [_type_label(b.get("type")) for b in buildings if b.get("type") is not None]
        labels += [_type_label(r.get("type")) for r in road_samples if r.get("type") is not None]
        type_colors = _build_type_colors(sorted(set(labels)))
    fig, ax = plt.subplots(figsize=(8, 8))
    for b in buildings:
        pos = b.get("position")
        bbox = b.get("bbox", {})
        rot = b.get("rotation_deg", 0)
        if pos is None or bbox is None:
            continue

        length = bbox.get("length")
        width = bbox.get("width")
        if length is None or width is None:
            continue

        label = _type_label(b.get("type"))
        if label not in type_colors:
            type_colors[label] = _random_color(label)

        rect = _build_rectangle(pos[0], pos[1], length, width, rot)
        patch = Polygon(rect, closed=True, edgecolor="black", facecolor=type_colors[label], linewidth=0.5, alpha=0.7)
        ax.add_patch(patch)

    for r in road_samples:
        pts = r.get("positions", [])
        if not pts:
            continue
        xs, ys = zip(*_build_road_line(pts))
        label = _type_label(r.get("type"))
        if label not in type_colors:
            type_colors[label] = _random_color(label)
        ax.plot(xs, ys, color=type_colors[label], linewidth=2.0, alpha=0.8, solid_capstyle="round")

    handles = [Patch(facecolor=col, edgecolor="black", label=lbl, alpha=0.7) for lbl, col in type_colors.items()]
    ax.legend(handles=handles, loc="upper right", fontsize="small", framealpha=0.8)

    ax.set_xlim(minx - 10, maxx + 10)
    ax.set_ylim(miny - 10, maxy + 10)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Tile {Path(input_path).stem} overhead view")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def visualize_folder(input_dir: Path, output_dir: Path, dpi: int = 200) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    json_files = [p for p in json_files if p.name.startswith("tile_")]
    if not json_files:
        raise ValueError(f"No tile JSON files found in {input_dir}")

    labels = set()
    for path in json_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for b in data.get("buildings", []):
            labels.add(_type_label(b.get("type")))
        for r in data.get("road_samples", []):
            labels.add(_type_label(r.get("type")))

    type_colors = _build_type_colors(sorted(labels), seed=random.randrange(2**32))

    for path in json_files:
        out_path = output_dir / f"{path.stem}.png"
        try:
            visualize_tile(path, out_path, dpi=dpi, type_colors=type_colors)
            print(f"Saved {out_path}")
        except ValueError as e:
            print(f"Skip {path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize tile JSON as an overhead view")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Input tile JSON path")
    group.add_argument("--input_dir", type=str, help="Directory of tile JSON files")
    parser.add_argument("--output", type=str, help="Output image path for a single tile")
    parser.add_argument("--output_dir", type=str, help="Output directory for batch visualization")
    parser.add_argument("--dpi", type=int, default=200, help="Output image DPI")
    args = parser.parse_args()

    if args.input:
        if not args.output:
            parser.error("--output is required when --input is used")
        visualize_tile(Path(args.input), Path(args.output), dpi=args.dpi)
    else:
        if not args.output_dir:
            parser.error("--output_dir is required when --input_dir is used")
        visualize_folder(Path(args.input_dir), Path(args.output_dir), dpi=args.dpi)


if __name__ == "__main__":
    main()
