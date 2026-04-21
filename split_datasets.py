"""
Collect PNG Paths And Split
"""

import argparse
import random
from pathlib import Path


def collect_png_paths(root_dir: Path):
    if not root_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {root_dir}")
    if not root_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是目录: {root_dir}")

    return sorted(str(path.resolve()) for path in root_dir.rglob("*.png") if path.is_file())


def main():
    parser = argparse.ArgumentParser(description="递归收集目录下所有 PNG 图片的完整路径，并保存到 TXT 文件")
    parser.add_argument("--input_dir", default="processed", help="要扫描的输入目录")
    parser.add_argument("--ratio", type=float, default=0.9, help="训练集比例，范围 0.0-1.0，默认 0.8")
    parser.add_argument("--train_output", default="train.txt", help="训练集输出 TXT 文件路径，默认 train.txt")
    parser.add_argument("--val_output", default="val.txt", help="验证集输出 TXT 文件路径，默认 val.txt")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，默认不固定")
    args = parser.parse_args()

    if not 0.0 <= args.ratio <= 1.0:
        raise ValueError("ratio 必须在 0.0 到 1.0 之间")

    root_dir = Path(args.input_dir)
    png_paths = collect_png_paths(root_dir)

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(png_paths)

    split_index = int(len(png_paths) * args.ratio)
    train_paths = png_paths[:split_index]
    val_paths = png_paths[split_index:]

    train_file = Path(args.train_output)
    val_file = Path(args.val_output)
    train_file.parent.mkdir(parents=True, exist_ok=True)
    val_file.parent.mkdir(parents=True, exist_ok=True)

    train_file.write_text("\n".join(train_paths) + ("\n" if train_paths else ""), encoding="utf-8")
    val_file.write_text("\n".join(val_paths) + ("\n" if val_paths else ""), encoding="utf-8")

    print(f"已找到 {len(png_paths)} 个 PNG 文件")
    print(f"训练集 {len(train_paths)} 个，已保存至: {train_file}")
    print(f"验证集 {len(val_paths)} 个，已保存至: {val_file}")


if __name__ == "__main__":
    main()
