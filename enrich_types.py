#!/usr/bin/env python3
"""
Backfill building type information from OSM using Overture source record IDs.

Input is a GeoJSON exported by fetch_building.py (for example all_features.geojson).
The script tries to parse OSM object IDs from the sources field and fetches OSM tags
through the OSM API, then writes enriched fields to a new GeoJSON.

Example:
python enrich_type.py \
  --input raw/newyork_manhattan/all_features.geojson \
  --output raw/newyork_manhattan/all_features_with_enriched_type.geojson \
  --cache raw/newyork_manhattan/osm_type_cache.json \
  --sleep 0.05
"""

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd
import importlib

try:
    tqdm = importlib.import_module("tqdm").tqdm
except ImportError:
    tqdm = None


OSM_ID_PATTERN = re.compile(r"([nwr])(\d+)(?:@\d+)?$")
GENERIC_BUILDING_VALUES = {"yes", "building"}


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def _save_json_file(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _iter_with_progress(iterable, total: Optional[int] = None, desc: Optional[str] = None):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, unit="row", leave=False)
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


def _parse_sources(value: Any) -> list:
    if _is_missing(value):
        return []

    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return decoded
            if isinstance(decoded, dict):
                return [decoded]
            return []
        except Exception:
            return []

    return []


def _extract_osm_obj_id_from_sources(value: Any) -> Optional[str]:
    for src in _parse_sources(value):
        if not isinstance(src, dict):
            continue
        record_id = src.get("record_id")
        if not isinstance(record_id, str):
            continue
        m = OSM_ID_PATTERN.match(record_id.strip())
        if m:
            return f"{m.group(1)}{m.group(2)}"
    return None


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if pd.isna(v) or math.isinf(v):
        return None
    return v


def _extract_name_text(row: pd.Series) -> str:
    names = row.get("names")
    if isinstance(names, dict):
        primary = names.get("primary")
        common = names.get("common")
        for val in (primary, common):
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, dict):
                text = val.get("value") or val.get("primary")
                if isinstance(text, str) and text.strip():
                    return text.strip()
    if isinstance(names, str):
        s = names.strip()
        if s:
            try:
                decoded = json.loads(s)
                if isinstance(decoded, dict):
                    text = decoded.get("primary") or decoded.get("common")
                    if isinstance(text, str) and text.strip():
                        return text.strip()
            except Exception:
                pass
    direct_name = row.get("name")
    if isinstance(direct_name, str) and direct_name.strip():
        return direct_name.strip()
    return ""


def _extract_existing_type(row: pd.Series) -> Optional[str]:
    subtype = row.get("subtype")
    if isinstance(subtype, str) and subtype.strip():
        return f"building={subtype.strip().lower()}"
    clazz = row.get("class")
    if isinstance(clazz, str) and clazz.strip():
        return f"building={clazz.strip().lower()}"
    return None


def _osm_api_path(osm_obj_id: str) -> Optional[str]:
    if len(osm_obj_id) < 2:
        return None
    obj_type = osm_obj_id[0]
    obj_num = osm_obj_id[1:]
    if not obj_num.isdigit():
        return None

    type_map = {"n": "node", "w": "way", "r": "relation"}
    mapped = type_map.get(obj_type)
    if mapped is None:
        return None
    return f"https://api.openstreetmap.org/api/0.6/{mapped}/{obj_num}.json"


def _fetch_osm_tags(osm_obj_id: str, timeout: float = 10.0, retries: int = 2) -> Optional[Dict[str, str]]:
    url = _osm_api_path(osm_obj_id)
    if url is None:
        return None

    last_error = None
    for attempt in range(retries + 1):
        try:
            with urlopen(url, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            elements = payload.get("elements") if isinstance(payload, dict) else None
            if not elements or not isinstance(elements, list):
                return None
            tags = elements[0].get("tags") if isinstance(elements[0], dict) else None
            if isinstance(tags, dict):
                out = {}
                for k, v in tags.items():
                    out[str(k)] = str(v)
                return out
            return None
        except HTTPError as e:
            last_error = e
            if e.code in (404, 410):
                return None
            if e.code == 429 and attempt < retries:
                time.sleep(1.0 + attempt)
                continue
            if attempt < retries:
                time.sleep(0.5 + attempt)
                continue
            break
        except (URLError, TimeoutError, json.JSONDecodeError) as e:
            last_error = e
            if attempt < retries:
                time.sleep(0.5 + attempt)
                continue
            break

    if last_error is not None:
        return None
    return None


def _infer_type_from_osm_tags(tags: Dict[str, str]) -> Tuple[Optional[str], str, str]:
    # Return (osm_type, confidence, matched_rule).
    if not tags:
        return None, "low", "no-tags"

    if tags.get("amenity"):
        return f"amenity={tags['amenity']}", "high", "amenity"
    if tags.get("shop"):
        return f"shop={tags['shop']}", "high", "shop"
    if tags.get("office"):
        return f"office={tags['office']}", "high", "office"
    if tags.get("tourism"):
        return f"tourism={tags['tourism']}", "high", "tourism"

    building = tags.get("building")
    if isinstance(building, str) and building.strip():
        b = building.strip().lower()
        if b not in GENERIC_BUILDING_VALUES:
            return f"building={b}", "medium", "building-specific"
        return None, "low", "building-generic"

    return None, "low", "fallback"


def _shape_metrics(geom) -> Tuple[Optional[float], Optional[float]]:
    if geom is None or geom.is_empty:
        return None, None
    area = _to_float(getattr(geom, "area", None))
    try:
        minx, miny, maxx, maxy = geom.bounds
        dx = max(maxx - minx, 0.0)
        dy = max(maxy - miny, 0.0)
        short = min(dx, dy)
        long_ = max(dx, dy)
        aspect = (long_ / short) if short > 0 else None
    except Exception:
        aspect = None
    return area, aspect


def _infer_type_from_local_features(row: pd.Series) -> Tuple[str, str, str]:
    geom = row.geometry
    area, aspect = _shape_metrics(geom)
    name_text = _extract_name_text(row).lower()

    height = _to_float(row.get("height"))
    levels = _to_float(row.get("building:levels")) or _to_float(row.get("levels_f")) or _to_float(row.get("num_floors"))

    if any(k in name_text for k in ("hospital", "clinic", "medical center")):
        return "building=hospital", "high", "name-medical"
    if any(k in name_text for k in ("school", "academy", "kindergarten")):
        return "building=school", "high", "name-school"
    if any(k in name_text for k in ("university", "college", "campus")):
        return "building=university", "high", "name-university"
    if any(k in name_text for k in ("hotel", "motel", "inn")):
        return "building=hotel", "high", "name-hotel"
    if any(k in name_text for k in ("church", "cathedral", "chapel")):
        return "building=church", "high", "name-religion"
    if any(k in name_text for k in ("mall", "plaza", "shopping", "market")):
        return "building=retail", "medium", "name-retail"
    if any(k in name_text for k in ("warehouse", "depot", "logistics", "storage")):
        return "building=warehouse", "high", "name-warehouse"
    if any(k in name_text for k in ("factory", "plant", "works")):
        return "building=industrial", "high", "name-industrial"
    if any(k in name_text for k in ("office", "headquarters", "hq", "tower")):
        return "building=office", "medium", "name-office"

    if (levels is not None and levels >= 25) or (height is not None and height >= 80):
        if any(k in name_text for k in ("residence", "apartment", "residential")):
            return "building=apartments", "medium", "height-high-res"
        return "building=commercial", "medium", "height-high"

    if area is not None and area >= 3500 and ((levels is not None and levels <= 4) or (height is not None and height <= 20)):
        return "building=warehouse", "medium", "area-large-low"

    if area is not None and area <= 220 and (levels is None or levels <= 3):
        if aspect is not None and aspect <= 2.8:
            return "building=house", "medium", "small-compact"

    if (levels is not None and levels >= 8) or (height is not None and height >= 30):
        return "building=apartments", "low", "midrise"

    return "building=residential", "low", "fallback-residential"


def enrich_from_osm(
    input_path: Path,
    output_path: Path,
    cache_path: Optional[Path],
    sleep_sec: float,
    max_requests: int,
    overwrite_existing: bool,
) -> None:
    gdf = gpd.read_file(input_path)
    if gdf.empty:
        raise ValueError("Input is empty")

    if "sources" not in gdf.columns:
        raise ValueError("Input does not contain 'sources' column")

    cache = _load_json_file(cache_path) if cache_path else {}

    osm_obj_ids = gdf["sources"].apply(_extract_osm_obj_id_from_sources)

    fetched = 0
    hit = 0

    osm_type_raw = []
    inferred_type = []
    type_confidence = []
    type_source = []
    matched_rule = []
    osm_tags_json = []

    iterator = _iter_with_progress(gdf.iterrows(), total=len(gdf), desc="Enriching")
    for pos, (idx, row) in enumerate(iterator):
        _ = idx

        existing_type = _extract_existing_type(row)
        if (not overwrite_existing) and existing_type is not None:
            osm_type_raw.append(existing_type)
            inferred_type.append(existing_type)
            type_confidence.append("high")
            type_source.append("existing")
            matched_rule.append("existing-type")
            osm_tags_json.append(None)
            continue

        obj_id = osm_obj_ids.iloc[pos]
        if not isinstance(obj_id, str):
            guessed_type, guessed_conf, guessed_rule = _infer_type_from_local_features(row)
            osm_type_raw.append(None)
            inferred_type.append(guessed_type)
            type_confidence.append(guessed_conf)
            type_source.append("heuristic")
            matched_rule.append(f"heuristic:{guessed_rule}")
            osm_tags_json.append(None)
            continue

        tags = cache.get(obj_id)
        if tags is None:
            if fetched >= max_requests:
                tags = None
            else:
                tags = _fetch_osm_tags(obj_id)
                fetched += 1
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
                cache[obj_id] = tags
        else:
            hit += 1

        if not isinstance(tags, dict):
            guessed_type, guessed_conf, guessed_rule = _infer_type_from_local_features(row)
            osm_type_raw.append(None)
            inferred_type.append(guessed_type)
            type_confidence.append(guessed_conf)
            type_source.append("heuristic")
            matched_rule.append(f"heuristic:{guessed_rule}")
            osm_tags_json.append(None)
            continue

        osm_type, confidence, rule = _infer_type_from_osm_tags(tags)
        if osm_type is not None:
            print(f"OSM type found for {obj_id}: {osm_type} (rule={rule})")
            osm_type_raw.append(osm_type)
            inferred_type.append(osm_type)
            type_confidence.append(confidence)
            type_source.append("osm")
            matched_rule.append(rule)
        else:
            guessed_type, guessed_conf, guessed_rule = _infer_type_from_local_features(row)
            osm_type_raw.append(None)
            inferred_type.append(guessed_type)
            type_confidence.append(guessed_conf)
            type_source.append("heuristic")
            matched_rule.append(f"heuristic:{guessed_rule}|osm:{rule}")
        osm_tags_json.append(json.dumps(tags, ensure_ascii=False))

    gdf["osm_type_raw"] = osm_type_raw
    gdf["inferred_type"] = inferred_type
    gdf["type_confidence"] = type_confidence
    gdf["type_source"] = type_source
    gdf["type_rule"] = matched_rule
    gdf["osm_tags_json"] = osm_tags_json

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path, driver="GeoJSON")

    if cache_path:
        _save_json_file(cache_path, cache)

    print(f"Rows: {len(gdf)}")
    print(f"Fetched from OSM API: {fetched}")
    print(f"Cache hits: {hit}")
    print(f"Output: {output_path}")
    if cache_path:
        print(f"Cache: {cache_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill building type info from OSM based on source record IDs")
    parser.add_argument("--input", type=str, required=True, help="Input GeoJSON path")
    parser.add_argument("--output", type=str, required=True, help="Output GeoJSON path")
    parser.add_argument("--cache", type=str, default=None, help="Optional cache JSON path for OSM tag lookups")
    parser.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between OSM API requests")
    parser.add_argument("--max_requests", type=int, default=5000, help="Maximum number of OSM API requests in this run")
    parser.add_argument(
        "--overwrite_existing",
        action="store_true",
        help="Also overwrite rows that already have subtype/class",
    )
    args = parser.parse_args()

    enrich_from_osm(
        input_path=Path(args.input),
        output_path=Path(args.output),
        cache_path=Path(args.cache) if args.cache else None,
        sleep_sec=args.sleep,
        max_requests=args.max_requests,
        overwrite_existing=args.overwrite_existing,
    )


if __name__ == "__main__":
    main()
