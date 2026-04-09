#!/usr/bin/env python3
"""
Split building GeoJSON into fixed-size tiles and export each qualified tile to JSON.

Each tile JSON includes:
- Tile metadata (bbox, CRS)
- Building data with tile-relative bbox dimensions and position
- Rotation angle, bbox length/width/height, name, and type

Example:
python split_tiles.py \
    --input raw/newyork_manhattan/all_features.geojson \
    --out_dir processed/newyork_manhattan \
    --tile_size 400 \
    --tile_step 200 \
    --min_buildings 100
"""

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
from shapely.affinity import translate
from shapely.geometry import box

try:
    tqdm = importlib.import_module("tqdm").tqdm
except ImportError:
    tqdm = None


Number = float


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return str(value)
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        return json.loads(json.dumps(value))
    except Exception:
        return str(value)


def _normalize_angle_deg(angle: float) -> float:
    # Normalize to [0, 180) because rectangle orientation is bidirectional.
    angle = angle % 180.0
    if angle < 0:
        angle += 180.0
    return angle


def _oriented_bbox_info(geom) -> Optional[Dict[str, Any]]:
    if geom is None or geom.is_empty:
        return None

    rect = geom.minimum_rotated_rectangle
    if rect is None or rect.is_empty:
        return None

    if rect.geom_type != "Polygon":
        minx, miny, maxx, maxy = geom.bounds
        return {
            "corners": [[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]],
            "position": [(minx + maxx) / 2.0, (miny + maxy) / 2.0],
            "width": maxx - minx,
            "height": maxy - miny,
            "rotation_deg": 0.0,
        }

    coords = list(rect.exterior.coords)
    if len(coords) < 5:
        return None

    corners = coords[:4]

    def edge_len(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    len01 = edge_len(corners[0], corners[1])
    len12 = edge_len(corners[1], corners[2])

    if len01 >= len12:
        p_start, p_end = corners[0], corners[1]
        width, height = len01, len12
    else:
        p_start, p_end = corners[1], corners[2]
        width, height = len12, len01

    angle = math.degrees(math.atan2(p_end[1] - p_start[1], p_end[0] - p_start[0]))
    angle = _normalize_angle_deg(angle)

    centroid = rect.centroid

    return {
        "corners": [[float(x), float(y)] for x, y in corners],
        "position": [float(centroid.x), float(centroid.y)],
        "width": float(width),
        "height": float(height),
        "rotation_deg": float(angle),
    }


def _build_tile_ranges(min_v: float, max_v: float, tile_size: float, tile_step: float) -> Iterable[Tuple[float, float]]:
    start = math.floor(min_v / tile_step) * tile_step
    end = math.ceil(max_v / tile_step) * tile_step

    cur = start
    while cur < end:
        nxt = cur + tile_size
        yield cur, nxt
        cur = cur + tile_step


def _extract_name_and_type(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    names = row_dict.get("names")
    if isinstance(names, dict):
        name = names.get("primary") or names.get("common")
        if isinstance(name, dict):
            name = name.get("value") or name.get("primary")
    else:
        name = None

    raw_height = row_dict.get("height")
    if raw_height is None or (isinstance(raw_height, float) and math.isnan(raw_height)):
        height_value = 3
    else:
        try:
            height_value = int(round(float(raw_height)))
        except Exception:
            height_value = 3

    raw_levels = row_dict.get("building:levels")
    if raw_levels is not None:
        try:
            building_levels = int(round(float(raw_levels)))
        except Exception:
            building_levels = None
    else:
        building_levels = None

    type_info = {
        "object_category": row_dict.get("object_category"),
        "source_type": row_dict.get("source_type"),
        "subtype": row_dict.get("subtype"),
        "class": row_dict.get("class"),
        "building_levels": building_levels,
        "height": height_value,
    }
    type_info = {k: _json_safe(v) for k, v in type_info.items() if v is not None}

    return {
        "name": _json_safe(name),
        "type": type_info,
    }


def _iter_with_progress(iterable, total: Optional[int] = None, desc: Optional[str] = None):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="tile", leave=False)
    if total is not None and desc:
        print(f"{desc}: 0/{total}", end="", flush=True)
        count = 0
        for item in iterable:
            yield item
            count += 1
            print(f"\r{desc}: {count}/{total}", end="", flush=True)
        print()
        return
    return iterable


def _aabb_from_building(building: Dict[str, Any]) -> Tuple[float, float, float, float]:
    cx, cy = building["position"]
    length = building["bbox"]["length"]
    width = building["bbox"]["width"]
    half_l = length / 2.0
    half_w = width / 2.0
    return cx - half_l, cy - half_w, cx + half_l, cy + half_w


def _intersection_area(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _filter_overlapping_buildings(buildings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n = len(buildings)
    if n < 2:
        return [{k: v for k, v in b.items() if not str(k).startswith("__")} for b in buildings]

    keep = [True] * n
    geoms = [None] * n
    areas = [0.0] * n
    for i, b in enumerate(buildings):
        geom = b.get("__overlap_poly")
        if geom is None or geom.is_empty:
            # Fallback to axis-aligned rectangle if no rotated geometry is available.
            geom = box(*_aabb_from_building(b))
        geoms[i] = geom
        areas[i] = float(geom.area)

    for i in range(n):
        if not keep[i]:
            continue
        if areas[i] <= 0.0:
            keep[i] = False
            continue
        for j in range(i + 1, n):
            if not keep[j]:
                continue
            if areas[j] <= 0.0:
                keep[j] = False
                continue
            overlap = geoms[i].intersection(geoms[j]).area
            if overlap <= 0.0:
                continue
            if overlap > 0.2 * areas[i] or overlap > 0.2 * areas[j]:
                if areas[i] >= areas[j]:
                    keep[i] = False
                    break
                keep[j] = False

    return [{k: v for k, v in b.items() if not str(k).startswith("__")} for idx, b in enumerate(buildings) if keep[idx]]


def split_to_tiles(
    input_path: Path,
    out_dir: Path,
    tile_size: float,
    tile_step: float,
    min_buildings: int,
) -> Dict[str, Any]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if tile_step <= 0:
        raise ValueError("tile_step must be positive")
    if tile_step > tile_size:
        raise ValueError("tile_step cannot be larger than tile_size")
    if min_buildings < 1:
        raise ValueError("min_buildings must be >= 1")

    gdf = gpd.read_file(input_path)
    if gdf.empty:
        raise ValueError("Input dataset is empty")

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.is_valid].copy()
    if gdf.empty:
        raise ValueError("No valid geometries in input dataset")

    if "id" not in gdf.columns:
        gdf["id"] = [f"feature_{i}" for i in range(len(gdf))]

    crs_text = gdf.crs.to_string() if gdf.crs is not None else None

    minx, miny, maxx, maxy = gdf.total_bounds
    x_ranges = list(_build_tile_ranges(minx, maxx, tile_size, tile_step))
    y_ranges = list(_build_tile_ranges(miny, maxy, tile_size, tile_step))

    out_dir.mkdir(parents=True, exist_ok=True)

    sindex = gdf.sindex
    qualified = []
    total_tiles = len(x_ranges) * len(y_ranges)

    progress_iter = _iter_with_progress(
        ((ix, iy, x0, x1, y0, y1) for ix, (x0, x1) in enumerate(x_ranges) for iy, (y0, y1) in enumerate(y_ranges)),
        total=total_tiles,
        desc="Processing tiles",
    )

    for ix, iy, x0, x1, y0, y1 in progress_iter:
        tile_poly = box(x0, y0, x1, y1)

        try:
            candidate_idx = list(sindex.query(tile_poly, predicate="intersects"))
        except TypeError:
            candidate_idx = list(sindex.query(tile_poly))

        if not candidate_idx:
            continue

        tile_gdf = gdf.iloc[candidate_idx]
        tile_gdf = tile_gdf[tile_gdf.geometry.intersects(tile_poly)]
        if tile_gdf.empty:
            continue

        building_count = len(tile_gdf)
        if building_count < min_buildings:
            continue

        buildings = []
        for _, row in tile_gdf.iterrows():
            geom = row.geometry
            row_dict = row.to_dict()
            oriented_bbox = _oriented_bbox_info(geom)
            name_type = _extract_name_and_type(row_dict)
            height_value = name_type["type"].get("height")
            overlap_poly = geom.minimum_rotated_rectangle

            if oriented_bbox is not None:
                length = int(round(max(oriented_bbox["width"], oriented_bbox["height"])))
                width = int(round(min(oriented_bbox["width"], oriented_bbox["height"])))
            else:
                bounds = geom.bounds
                length = int(round(bounds[2] - bounds[0]))
                width = int(round(bounds[3] - bounds[1]))

            if height_value is None:
                height_value = 3

            minx, miny, _, _ = geom.bounds
            bbox_x = int(round(minx - x0))
            bbox_y = int(round(miny - y0))
            bbox_dims = {
                "length": length,
                "width": width,
                "height": int(round(height_value)),
            }
            position = [
                int(round(bbox_x + length / 2.0)),
                int(round(bbox_y + width / 2.0)),
            ]

            buildings.append(
                {
                    "rotation_deg": int(round(oriented_bbox["rotation_deg"])) if oriented_bbox is not None else 0,
                    "bbox": bbox_dims,
                    "position": position,
                    "name": name_type.get("name"),
                    "type": name_type.get("type"),
                    "__overlap_poly": overlap_poly,
                }
            )

        buildings = _filter_overlapping_buildings(buildings)
        building_count = len(buildings)

        tile_id = f"tile_x{ix}_y{iy}"
        tile_data = {
            "tile_id": tile_id,
            "crs": crs_text,
            "tile_bbox": [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))],
            "region_size": {
                "width": int(round(x1 - x0)),
                "height": int(round(y1 - y0)),
            },
            "building_count": int(building_count),
            "buildings": buildings,
        }

        tile_file = out_dir / f"{tile_id}.json"
        with tile_file.open("w", encoding="utf-8") as f:
            json.dump(tile_data, f, ensure_ascii=False)

        qualified.append(
            {
                "tile_id": tile_id,
                "tile_file": str(tile_file),
                "tile_bbox": tile_data["tile_bbox"],
                "building_count": building_count,
            }
        )

    summary = {
        "input": str(input_path),
        "output_dir": str(out_dir),
        "crs": crs_text,
        "tile_size": float(tile_size),
        "tile_step": float(tile_step),
        "min_buildings": int(min_buildings),
        "total_tiles_scanned": total_tiles,
        "qualified_tile_count": len(qualified),
        "qualified_tiles": qualified,
    }

    summary_file = out_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("Split building GeoJSON into 800m-like tiles and export one JSON per qualified tile.")
    )
    parser.add_argument("--input", type=str, required=True, help="Input all_features.geojson path")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for tile JSON files")
    parser.add_argument("--tile_size", type=float, default=800.0, help="Tile size in current CRS units (meters for UTM)")
    parser.add_argument(
        "--tile_step",
        type=float,
        default=None,
        help="Step/stride for tile origin in current CRS units (overlap if smaller than tile_size)",
    )
    parser.add_argument("--min_buildings", type=int, default=30, help="Minimum building count to keep a tile")
    args = parser.parse_args()

    tile_step = args.tile_step if args.tile_step is not None else args.tile_size

    summary = split_to_tiles(
        input_path=Path(args.input),
        out_dir=Path(args.out_dir),
        tile_size=args.tile_size,
        tile_step=tile_step,
        min_buildings=args.min_buildings,
    )

    print(f"Qualified tiles: {summary['qualified_tile_count']}")
    print(f"Summary: {Path(args.out_dir) / 'summary.json'}")


if __name__ == "__main__":
    main()
