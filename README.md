# Overture Tile Processing

## Overview / 概述

This repository contains scripts for downloading building/road data, splitting it into tiles, inferring building types, and visualizing tile JSON files.

本仓库包含用于下载建筑/道路数据、将数据切分为瓦片、推断建筑类型以及可视化瓦片 JSON 文件的脚本。

## Scripts / 脚本

- `fetch_buildings.py`
  - Download Overture building data and optionally building parts or roads.
  - Fetches `building`, optionally `building_part` and road-related feature types.
  - Saves output as `all_features.geojson`.

- `split_tiles.py`
  - Split a GeoJSON of features into fixed-size tiles.
  - Outputs per-tile JSON files containing building objects and sampled road objects.

- `visualize_tiles.py`
  - Render tile JSON files as overhead images.
  - Supports buildings and road sample lines in each tile.

- `count_tile_objects.py`
  - Count building and road objects in tile JSON files.
  - Print top N files by building / road count.

## Dependencies / 依赖

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install geopandas pandas shapely matplotlib tqdm
```

## Example Usage / 示例用法

### Fetch data / 下载数据

```bash
python fetch_buildings.py \
  --bbox "40.70,-74.02;40.88,-73.92" \
  --out_dir raw/newyork_manhattan \
  --target_crs EPSG:32618 \
  --include_parts \
  --include_roads
```

### Split tiles / 切分瓦片

```bash
python split_tiles.py \
  --input raw/newyork_manhattan/all_features.geojson \
  --out_dir processed/newyork_manhattan \
  --tile_size 400 \
  --tile_step 200 \
  --min_buildings 200
```

### Visualize tiles / 可视化瓦片

```bash
python visualize_tiles.py \
  --input_dir processed/newyork_manhattan \
  --output_dir processed/newyork_manhattan
```

### Count objects in tiles / 统计瓦片对象

```bash
python count_tile_objects.py --input_dir processed/newyork_manhattan --top 10
```

## Notes / 说明

- The `split_tiles.py` workflow currently infers building types locally and emits sampled road geometries as road objects.
- `visualize_tiles.py` can render both building rectangles and road polylines.
- `count_tile_objects.py` prints the top files by building count and by road count.

- `split_tiles.py` 当前仅基于本地规则推断建筑类型，并将道路几何采样结果作为道路对象保存。
- `visualize_tiles.py` 能同时渲染建筑矩形和道路折线。
- `count_tile_objects.py` 分别输出建筑数量和道路数量排名前 N 的文件。
