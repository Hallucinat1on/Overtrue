#!/usr/bin/env python3
"""
Visualize a tile JSON file with fixed distinct colors on a black background.

Usage:
  python visualize_tiles_seg.py --input processed/newyork_manhattan/tile_x43_y63.json --output test.png
  python visualize_tiles_seg.py --input_dir processed/tokyo_shibuya --output_dir processed/tokyo_shibuya
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import Polygon as ShapelyPolygon, LineString
from shapely.ops import unary_union


def _load_color_config(config_path: Path) -> Tuple[Dict[str, str], Dict[str, Tuple[float, float, float]]]:
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    building_map = config.get("building")
    color_map = config.get("color")
    if not isinstance(building_map, dict) or not isinstance(color_map, dict):
        raise ValueError("Color config must contain 'building' and 'color' mappings")

    normalized_colors: Dict[str, Tuple[float, float, float]] = {}
    for key, value in color_map.items():
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            raise ValueError(f"Color for '{key}' must be an RGB list of length 3")
        rgb = tuple(float(v) / 255.0 for v in value)
        normalized_colors[key] = rgb

    return building_map, normalized_colors


def _map_category(label: Optional[str], building_map: Dict[str, str]) -> str:
    if not isinstance(label, str) or not label.strip():
        return "unknown"
    label = label.strip()
    mapped = building_map.get(label)
    if isinstance(mapped, str) and mapped.strip():
        return mapped.strip()
    return label


def _get_color_for_category(category: str, color_map: Dict[str, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if category in color_map:
        return color_map[category]
    return color_map.get("unknown", (0.5, 0.5, 0.5))


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


def _building_area(building: dict, use_contours: bool) -> float:
    if use_contours:
        contour = building.get("contour")
        if contour:
            total_area = 0.0
            for poly in contour:
                if not poly or len(poly) < 3:
                    continue
                total_area += ShapelyPolygon(poly).area
            return total_area
    bbox = building.get("bbox") or {}
    length = bbox.get("length") or 0.0
    width = bbox.get("width") or 0.0
    return float(length) * float(width)


def visualize_tile(
    input_path: Path,
    output_path: Path,
    color_config: Path,
    dpi: int = 200,
    use_contours: bool = False,
    min_building_area: float = 0.0,
) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    building_map, color_map = _load_color_config(color_config)

    buildings = data.get("buildings", [])
    road_samples = data.get("road_samples", [])
    if not buildings and not road_samples:
        raise ValueError("Tile JSON has no buildings or road samples")

    xs = []
    ys = []
    for b in buildings:
        contour = b.get("contour")
        if use_contours and contour:
            for poly in contour:
                for x, y in poly:
                    xs.append(x)
                    ys.append(y)
        else:
            pos = b.get("position")
            bbox = b.get("bbox", {})
            if pos is None or bbox is None:
                continue
            xs.extend([pos[0] - bbox.get("length", 0) / 2.0, pos[0] + bbox.get("length", 0) / 2.0])
            ys.extend([pos[1] - bbox.get("width", 0) / 2.0, pos[1] + bbox.get("width", 0) / 2.0])
    for r in road_samples:
        for point in r.get("positions", []):
            xs.append(point[0])
            ys.append(point[1])

    if not xs or not ys:
        raise ValueError("Tile JSON has no coordinate data")

    minx = min(xs)
    miny = min(ys)
    maxx = max(xs)
    maxy = max(ys)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    bldg_polys = []

    for b in buildings:
        area = _building_area(b, use_contours)
        if area < min_building_area:
            continue

        raw_label = _type_label(b.get("type"))
        category = _map_category(raw_label, building_map)
        color = _get_color_for_category(category, color_map)

        contour = b.get("contour")
        if use_contours and contour:
            for poly in contour:
                if not poly:
                    continue
                patch = Polygon(poly, closed=True, edgecolor="black", facecolor=color, linewidth=0.7, alpha=1.0, zorder=2)
                ax.add_patch(patch)
                if len(poly) >= 3:
                    bldg_polys.append(ShapelyPolygon(poly))
            continue

        pos = b.get("position")
        bbox = b.get("bbox", {})
        rot = b.get("rotation_deg", 0)
        if pos is None or bbox is None:
            continue

        length = bbox.get("length")
        width = bbox.get("width")
        if length is None or width is None:
            continue

        rect = _build_rectangle(pos[0], pos[1], length, width, rot)
        patch = Polygon(rect, closed=True, edgecolor="black", facecolor=color, linewidth=0.7, alpha=1.0, zorder=2)
        ax.add_patch(patch)
        if len(rect) >= 3:
            bldg_polys.append(ShapelyPolygon(rect))

    buildings_union = unary_union(bldg_polys) if bldg_polys else ShapelyPolygon()

    # Threshold for dropping tiny road segments (e.g. 15 meters)
    MIN_ROAD_LENGTH = 15.0

    road_color = color_map.get("road", (0.5, 0.5, 0.5))
    for r in road_samples:
        road_type = r.get("type")
        road_class = ""
        if isinstance(road_type, dict):
            road_class = str(road_type.get("class") or "").strip().lower()
        if road_class in {"unknown", "water", "footway", "steps", "cycleway", "subway"}:
            continue

        pts = r.get("positions", [])
        if len(pts) < 2:
            continue

        line = LineString(pts)
        if not buildings_union.is_empty:
            diff = line.difference(buildings_union)
        else:
            diff = line

        if diff.is_empty:
            continue

        geoms = [diff] if diff.geom_type == "LineString" else diff.geoms

        for geom in geoms:
            if geom.length > MIN_ROAD_LENGTH:
                xs_line, ys_line = geom.xy
                # 绘制黑色边框
                ax.plot(
                    xs_line,
                    ys_line,
                    color="black",
                    linewidth=4.0,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    alpha=1.0,
                    zorder=1,
                )
                # 绘制彩色道路中线
                ax.plot(
                    xs_line,
                    ys_line,
                    color=road_color,
                    linewidth=2.5,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    alpha=1.0,
                    zorder=1,
                )

    x_span = maxx - minx
    y_span = maxy - miny
    span = max(x_span, y_span)
    margin = 2.0
    center_x = (minx + maxx) / 2.0
    center_y = (miny + maxy) / 2.0
    half_span = span / 2.0 + margin

    ax.set_xlim(center_x - half_span, center_x + half_span)
    ax.set_ylim(center_y - half_span, center_y + half_span)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def visualize_folder(
    input_dir: Path,
    output_dir: Path,
    color_config: Path,
    dpi: int = 200,
    use_contours: bool = False,
    min_building_area: float = 0.0,
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    json_files = [p for p in json_files if p.name.startswith("tile_")]
    if not json_files:
        raise ValueError(f"No tile JSON files found in {input_dir}")

    for path in json_files:
        out_path = output_dir / f"{path.stem}.png"
        try:
            visualize_tile(path, out_path, color_config, dpi=dpi, use_contours=use_contours, min_building_area=min_building_area)
            print(f"Saved {out_path}")
        except ValueError as e:
            print(f"Skip {path}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize tile JSON with colors loaded from a config JSON file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Input tile JSON path")
    group.add_argument("--input_dir", type=str, help="Directory of tile JSON files")
    parser.add_argument("--output", type=str, help="Output image path for a single tile")
    parser.add_argument("--output_dir", type=str, default="test", help="Output directory for batch visualization")
    parser.add_argument("--config", type=str, default="processed/color.json", help="Map building to parent categories and colors")
    parser.add_argument("--dpi", type=int, default=256, help="Output image DPI")
    parser.add_argument("--use_contours", type=bool, default=True, help="Render building contours instead of bbox rectangles")
    parser.add_argument("--min_building_area", type=float, default=8.0, help="Skip buildings whose area is below this threshold")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists() or not config_path.is_file():
        parser.error(f"Config file not found: {args.config}")

    if args.input:
        if not args.output:
            parser.error("--output is required when --input is used")
        visualize_tile(
            Path(args.input),
            Path(args.output),
            config_path,
            dpi=args.dpi,
            use_contours=args.use_contours,
            min_building_area=args.min_building_area,
        )
    else:
        if not args.output_dir:
            parser.error("--output_dir is required when --input_dir is used")
        visualize_folder(
            Path(args.input_dir),
            Path(args.output_dir),
            dpi=args.dpi,
            use_contours=args.use_contours,
            min_building_area=args.min_building_area,
            color_config=config_path,
        )


if __name__ == "__main__":
    main()
