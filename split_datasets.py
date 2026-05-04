"""
Collect PNG Paths And Split
"""

import argparse
import random
import json
from pathlib import Path


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


def _map_category(label, building_map):
    if not isinstance(label, str) or not label.strip():
        return "unknown"
    label = label.strip()
    mapped = building_map.get(label)
    if isinstance(mapped, str) and mapped.strip():
        return mapped.strip()
    return label


def extract_condition_vector(png_path: Path, building_map: dict, category_order: list):
    json_path = png_path.with_suffix(".json")
    if not json_path.exists():
        return [0.0] * len(category_order) + [0.0, 0.0]

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    buildings = data.get("buildings", [])
    road_samples = data.get("road_samples", [])

    cat_counts = {c: 0 for c in category_order}
    for b in buildings:
        label = _type_label(b.get("type"))
        mapped = _map_category(label, building_map)
        if mapped in cat_counts:
            cat_counts[mapped] += 1

    props = [cat_counts[c] for c in category_order]

    num_roads = len(road_samples)
    total_pts = sum(len(r.get("positions", [])) for r in road_samples)
    avg_road_pts = total_pts / num_roads if num_roads > 0 else 0.0

    return props + [float(num_roads), float(avg_road_pts)]


def collect_png_paths(root_dir: Path):
    if not root_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {root_dir}")

    return sorted(str(path.resolve()) for path in root_dir.rglob("*.png") if path.is_file())


def main():
    parser = argparse.ArgumentParser(description="递归收集 PNG 图片，提取 JSON 条件向量，并保存到 TXT 文件")
    parser.add_argument(
        "--input_dir", default="/opt/data/private/yihengxu/MajutsuCity/layout_gen/datasets/processed", help="要扫描的输入目录"
    )
    parser.add_argument("--config", default="processed-400/color.json", help="类别映射配置文件")
    parser.add_argument("--ratio", type=float, default=0.9, help="训练集比例，范围 0.0-1.0，默认 0.9")
    parser.add_argument(
        "--train_output",
        default="/opt/data/private/yihengxu/MajutsuCity/layout_gen/datasets/train.txt",
        help="训练集输出 TXT 文件路径，默认 train.txt",
    )
    parser.add_argument(
        "--val_output",
        default="/opt/data/private/yihengxu/MajutsuCity/layout_gen/datasets/val.txt",
        help="验证集输出 TXT 文件路径，默认 val.txt",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认不固定")
    args = parser.parse_args()

    if not 0.0 <= args.ratio <= 1.0:
        raise ValueError("ratio 必须在 0.0 到 1.0 之间")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    building_map = config.get("building", {})
    color_keys = config.get("color", {})
    # 按照 color.json 中的顺序提取建筑类别（排除了 road）
    category_order = [k for k in color_keys.keys() if k != "road"]

    root_dir = Path(args.input_dir)
    png_paths = collect_png_paths(root_dir)

    dataset_lines = []
    for p in png_paths:
        vector = extract_condition_vector(Path(p), building_map, category_order)
        # 将向量各维度序列化，与路径以空格相连（组成单行）
        vec_str = " ".join(f"{v:.4f}" for v in vector)
        dataset_lines.append(f"{p} {vec_str}")

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(dataset_lines)

    split_index = int(len(dataset_lines) * args.ratio)
    train_lines = dataset_lines[:split_index]
    val_lines = dataset_lines[split_index:]

    train_file = Path(args.train_output)
    val_file = Path(args.val_output)
    train_file.parent.mkdir(parents=True, exist_ok=True)
    val_file.parent.mkdir(parents=True, exist_ok=True)

    train_file.write_text("\n".join(train_lines) + ("\n" if train_lines else ""), encoding="utf-8")
    val_file.write_text("\n".join(val_lines) + ("\n" if val_lines else ""), encoding="utf-8")

    print(f"已处理 {len(png_paths)} 个 PNG 文件")
    print(f"特征向量维度: {len(category_order) + 2} (- 包含 {len(category_order)} 个建筑类别占比, 1个道路条数, 1个道路平均点数)")
    print(f"提取顺序: {', '.join(category_order)}, road_count, avg_road_pts")
    print(f"训练集 {len(train_lines)} 个，已保存至: {train_file}")
    print(f"验证集 {len(val_lines)} 个，已保存至: {val_file}")


if __name__ == "__main__":
    main()
