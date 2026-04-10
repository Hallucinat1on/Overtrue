#!/usr/bin/env python3
"""
Fetch building data from Overture Maps Foundation and export a single GeoJSON
named all_features.geojson.

This script is designed to be compatible with downstream processing that expects
fields like object_category and building:levels.

Usage:
python fetch_buildings.py \
  --bbox "39.95,116.25;40.05,116.38" \
  --out_dir raw/beijing_haidian \
  --target_crs EPSG:32651 \
  --include_parts \
  --include_roads

python fetch_buildings.py \
  --bbox "40.70,-74.02;40.88,-73.92" \
  --out_dir raw/newyork_manhattan \
  --target_crs EPSG:32618 \
  --include_parts \
  --include_roads

python fetch_buildings.py \
  --bbox "35.65,139.68;35.72,139.77" \
  --out_dir raw/tokyo_shibuya \
  --target_crs EPSG:32654 \
  --include_parts \
  --include_roads

python fetch_buildings.py \
  --bbox "22.27,114.15;22.33,114.21" \
  --out_dir raw/hongkong_kowloon \
  --target_crs EPSG:32650 \
  --include_parts \
  --include_roads

python fetch_buildings.py \
  --bbox "51.48,-0.15;51.53,-0.05" \
  --out_dir raw/london_central \
  --target_crs EPSG:32630 \
  --include_parts \
  --include_roads
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Optional, Tuple

try:
    import geopandas as gpd
    import pandas as pd
except Exception as e:
    print("Missing dependency: please install geopandas.\n", e, file=sys.stderr)
    raise


def _normalize_crs(target_crs: Optional[str]) -> Optional[str]:
    if target_crs is None:
        return None
    text = str(target_crs).strip()
    if not text:
        return None
    if text.lower().startswith("epsg:"):
        return text.upper()
    if text.isdigit():
        return f"EPSG:{text}"
    return text


def _parse_bbox(text: str) -> Tuple[float, float, float, float]:
    s = str(text).strip()
    if not s:
        raise ValueError("--bbox is required")

    # Accept either:
    # 1) lat1,lon1;lat2,lon2
    # 2) lat1,lon1,lat2,lon2
    if ";" in s:
        parts = s.split(";")
        if len(parts) != 2:
            raise ValueError("Invalid --bbox format")
        a = [p.strip() for p in parts[0].split(",")]
        b = [p.strip() for p in parts[1].split(",")]
        if len(a) != 2 or len(b) != 2:
            raise ValueError("Invalid --bbox format")
        lat1, lon1 = float(a[0]), float(a[1])
        lat2, lon2 = float(b[0]), float(b[1])
    else:
        parts = [p.strip() for p in s.split(",")]
        if len(parts) != 4:
            raise ValueError("Invalid --bbox format")
        lat1, lon1, lat2, lon2 = map(float, parts)

    if not (-90.0 <= lat1 <= 90.0 and -90.0 <= lat2 <= 90.0):
        raise ValueError("Latitude out of range")
    if not (-180.0 <= lon1 <= 180.0 and -180.0 <= lon2 <= 180.0):
        raise ValueError("Longitude out of range")

    south, north = min(lat1, lat2), max(lat1, lat2)
    west, east = min(lon1, lon2), max(lon1, lon2)
    return west, south, east, north


def _run_overture_download(
    out_file: Path,
    bbox_wsen: Tuple[float, float, float, float],
    feature_type: str,
    release: Optional[str],
    use_stac: bool,
) -> None:
    _run_overture_download_with_fallback(
        out_file=out_file,
        bbox_wsen=bbox_wsen,
        feature_type=feature_type,
        release=release,
        use_stac=use_stac,
    )


def _build_overture_cmd(
    out_file: Path,
    bbox_wsen: Tuple[float, float, float, float],
    feature_type: str,
    release: Optional[str],
    use_stac: bool,
):
    west, south, east, north = bbox_wsen
    cmd = [
        "overturemaps",
        "download",
        f"--bbox={west},{south},{east},{north}",
        "-f",
        "geojson",
        "--type",
        feature_type,
        "-o",
        str(out_file),
    ]

    if release:
        cmd.extend(["--release", str(release)])
    if not use_stac:
        cmd.append("--no-stac")

    return cmd


def _extract_available_releases_from_help() -> list:
    try:
        proc = subprocess.run(
            ["overturemaps", "download", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    # Example: -r, --release [2025-08-20.0|2025-08-20.1|2025-09-24.0]
    m = re.search(r"--release\s*\[([^\]]+)\]", text)
    if not m:
        return []

    raw_items = [x.strip() for x in m.group(1).split("|") if x.strip()]
    # Deduplicate while preserving order.
    seen = set()
    items = []
    for it in raw_items:
        if it in seen:
            continue
        seen.add(it)
        items.append(it)
    return items


def _run_overture_download_with_fallback(
    out_file: Path,
    bbox_wsen: Tuple[float, float, float, float],
    feature_type: str,
    release: Optional[str],
    use_stac: bool,
) -> None:
    available_releases = _extract_available_releases_from_help()

    attempts = []
    seen = set()

    def add_attempt(rel: Optional[str], stac: bool) -> None:
        key = (rel, stac)
        if key in seen:
            return
        seen.add(key)
        attempts.append(key)

    # Preferred attempt: user intent first.
    add_attempt(release, use_stac)

    # If STAC is enabled, add no-stac fallback for same release.
    if use_stac:
        add_attempt(release, False)

    # If release is not pinned, iterate all known releases.
    if release is None and available_releases:
        # Prefer newer looking release strings first.
        for rel in sorted(available_releases, reverse=True):
            add_attempt(rel, use_stac)
            if use_stac:
                add_attempt(rel, False)

    errors = []
    for rel, stac in attempts:
        cmd = _build_overture_cmd(
            out_file=out_file,
            bbox_wsen=bbox_wsen,
            feature_type=feature_type,
            release=rel,
            use_stac=stac,
        )
        print(
            "Try overture download:",
            f"type={feature_type}, release={rel or 'default'}, stac={stac}",
        )

        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except FileNotFoundError as e:
            raise RuntimeError("Cannot find overturemaps CLI. Install with: pip install overturemaps") from e

        if proc.returncode == 0:
            # Surface stdout for easier debugging when user runs the script.
            if proc.stdout:
                print(proc.stdout.strip())
            return

        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        err_text = " | ".join(tail) if tail else f"exit_code={proc.returncode}"
        errors.append(f"release={rel or 'default'}, stac={stac}, err={err_text}")

    detail = "\n  - " + "\n  - ".join(errors[-6:]) if errors else ""
    raise RuntimeError(f"overturemaps download failed for type={feature_type} after retries.{detail}")


def _coerce_levels(row) -> Optional[int]:
    # Overture has num_floors and optionally level. Map to building:levels.
    num_floors = row.get("num_floors")
    if num_floors is not None:
        try:
            return max(1, int(round(float(num_floors))))
        except Exception:
            pass

    level = row.get("level")
    if level is not None:
        try:
            return max(1, int(round(float(level))))
        except Exception:
            return None

    return None


def _sanitize_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, dict, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


def _ensure_required_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if "object_category" not in gdf.columns:
        gdf["object_category"] = "building"
    else:
        missing_mask = gdf["object_category"].isna()
        if missing_mask.any():
            gdf.loc[missing_mask, "object_category"] = "building"

    # Only building features should get building levels.
    if "building:levels" not in gdf.columns:
        gdf["building:levels"] = pd.NA
    building_mask = gdf["object_category"] == "building"
    if building_mask.any():
        levels_missing = gdf.loc[building_mask, "building:levels"].isna()
        if levels_missing.any():
            gdf.loc[building_mask & levels_missing, "building:levels"] = gdf.loc[building_mask & levels_missing].apply(
                _coerce_levels, axis=1
            )

    if "levels_f" not in gdf.columns:
        gdf["levels_f"] = gdf["building:levels"]

    # Mark source fields for traceability.
    if "source_dataset" not in gdf.columns:
        gdf["source_dataset"] = "Overture Maps Foundation"

    if "source_type" not in gdf.columns:
        if "type" in gdf.columns:
            gdf["source_type"] = gdf["type"]
        else:
            gdf["source_type"] = gdf["object_category"]

    return gdf


def _read_geojson_if_exists(path: Path) -> gpd.GeoDataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    gdf = gpd.read_file(path)
    if gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf


def fetch(
    out_dir: Path,
    bbox: str,
    target_crs: Optional[str],
    include_parts: bool,
    include_roads: bool,
    release: Optional[str],
    use_stac: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bbox_wsen = _parse_bbox(bbox)
    target_crs = _normalize_crs(target_crs)

    with tempfile.TemporaryDirectory(prefix="overture_buildings_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        building_geojson = tmp_dir_path / "building.geojson"
        print("Downloading Overture type=building ...")
        _run_overture_download(building_geojson, bbox_wsen, "building", release, use_stac)
        building_gdf = _read_geojson_if_exists(building_geojson)
        building_gdf["object_category"] = "building"

        frames = [building_gdf]

        if include_parts:
            part_geojson = tmp_dir_path / "building_part.geojson"
            print("Downloading Overture type=building_part ...")
            try:
                _run_overture_download(part_geojson, bbox_wsen, "building_part", release, use_stac)
                part_gdf = _read_geojson_if_exists(part_geojson)
                if not part_gdf.empty:
                    part_gdf["object_category"] = "building"
                    frames.append(part_gdf)
            except RuntimeError as e:
                # Keep building-only export when building_part is unavailable.
                print(f"Skip building_part due to error: {e}")

        if include_roads:
            for feature_type in ("segment", "connector"):
                road_geojson = tmp_dir_path / f"road_{feature_type}.geojson"
                print(f"Downloading Overture type={feature_type} ...")
                try:
                    _run_overture_download(road_geojson, bbox_wsen, feature_type, release, use_stac)
                    road_gdf = _read_geojson_if_exists(road_geojson)
                    if not road_gdf.empty:
                        road_gdf["object_category"] = "road"
                        frames.append(road_gdf)
                except RuntimeError as e:
                    print(f"Skip {feature_type} due to error: {e}")

        all_features = gpd.GeoDataFrame(
            pd.concat(frames, ignore_index=True),
            geometry="geometry",
            crs=frames[0].crs if frames and frames[0].crs is not None else "EPSG:4326",
        )

    if all_features.empty:
        print("No Overture building features found in area.")
        return

    # Keep only valid geometries.
    all_features = all_features[all_features.geometry.notna()].copy()
    all_features = all_features[all_features.geometry.is_valid].copy()
    all_features = all_features[~all_features.geometry.is_empty].copy()

    all_features = _ensure_required_columns(all_features)

    if target_crs is not None:
        try:
            all_features = all_features.to_crs(target_crs)
            print(f"Reprojected all features -> {target_crs}")
        except Exception as e:
            print(f"Reproject all_features failed: {e}")

    # Fiona/GeoPandas cannot serialize list/dict columns to GeoJSON directly.
    for col in all_features.columns:
        if col == all_features.geometry.name:
            continue
        if all_features[col].dtype == object:
            all_features[col] = all_features[col].apply(_sanitize_value)

    out_file = out_dir / "all_features.geojson"
    all_features.to_file(out_file, driver="GeoJSON")
    print(f"Saved Overture buildings -> {out_file}")
    print(f"Feature count: {len(all_features)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Overture buildings for a bbox and save as all_features.geojson")
    parser.add_argument(
        "--bbox",
        type=str,
        required=True,
        help=("Bounding box by two corners in lat/lon. " "Format: 'lat1,lon1;lat2,lon2' or 'lat1,lon1,lat2,lon2'."),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for all_features.geojson",
    )
    parser.add_argument(
        "--target_crs",
        type=str,
        default="EPSG:32651",
        help="Optional target CRS to reproject output",
    )
    parser.add_argument(
        "--include_parts",
        action="store_true",
        help="Also download building_part for denser building geometry coverage",
    )
    parser.add_argument(
        "--include_roads",
        action="store_true",
        help="Also download road features and include them in all_features.geojson",
    )
    parser.add_argument(
        "--release",
        type=str,
        default=None,
        help="Optional Overture release tag, e.g. 2026-03-25.0",
    )
    parser.add_argument(
        "--no_stac",
        action="store_true",
        help="Disable STAC acceleration and query latest release directly",
    )
    args = parser.parse_args()

    fetch(
        out_dir=Path(args.out_dir),
        bbox=args.bbox,
        target_crs=args.target_crs,
        include_parts=args.include_parts,
        include_roads=args.include_roads,
        release=args.release,
        use_stac=(not args.no_stac),
    )


if __name__ == "__main__":
    main()
