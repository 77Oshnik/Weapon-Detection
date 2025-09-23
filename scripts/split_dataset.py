#!/usr/bin/env python3
"""
Split a YOLO dataset that has all images/labels in a single folder into train/val/test splits.
This is useful for datasets like Pistols that come in 'export/' format instead of pre-split.

Usage:
  python scripts/split_dataset.py --source dataset/Pistols.v1-resize-416x416.yolov8/export --output dataset/Pistols.v1-resize-416x416.yolov8 --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_image_label_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    """Get matching image-label pairs based on filename stems."""
    pairs = []
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(images_dir.glob(f'*{ext}'))
        image_files.extend(images_dir.glob(f'*{ext.upper()}'))
    
    for img_path in image_files:
        # Look for corresponding label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            print(f"[WARN] No label found for image: {img_path}")
    
    return pairs


def split_pairs(pairs: List[Tuple[Path, Path]], train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[List, List, List]:
    """Split image-label pairs into train/val/test."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
    
    # Shuffle pairs
    pairs_copy = pairs.copy()
    random.shuffle(pairs_copy)
    
    n = len(pairs_copy)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_pairs = pairs_copy[:n_train]
    val_pairs = pairs_copy[n_train:n_train + n_val]
    test_pairs = pairs_copy[n_train + n_val:]
    
    return train_pairs, val_pairs, test_pairs


def copy_pairs_to_split(pairs: List[Tuple[Path, Path]], output_dir: Path, split_name: str):
    """Copy image-label pairs to the appropriate split directory."""
    split_images_dir = output_dir / split_name / "images"
    split_labels_dir = output_dir / split_name / "labels"
    
    ensure_dir(split_images_dir)
    ensure_dir(split_labels_dir)
    
    for img_path, label_path in pairs:
        # Copy image
        dest_img = split_images_dir / img_path.name
        shutil.copy2(img_path, dest_img)
        
        # Copy label
        dest_label = split_labels_dir / label_path.name
        shutil.copy2(label_path, dest_label)
    
    print(f"[INFO] {split_name}: copied {len(pairs)} image-label pairs")


def update_data_yaml(output_dir: Path, class_names: List[str]):
    """Create or update data.yaml with correct paths and class info."""
    data_yaml_path = output_dir / "data.yaml"
    
    # Use relative paths from the data.yaml location
    content = f"""train: train/images
val: valid/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""
    
    data_yaml_path.write_text(content, encoding='utf-8')
    print(f"[INFO] Updated {data_yaml_path}")


def main():
    parser = argparse.ArgumentParser(description="Split YOLO dataset into train/val/test")
    parser.add_argument("--source", type=Path, required=True,
                        help="Source directory containing images/ and labels/ folders")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory where train/val/test splits will be created")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-names", type=str, nargs='+', default=None,
                        help="Class names for data.yaml (e.g., --class-names pistol)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without actually copying files")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Validate input
    source_images_dir = args.source / "images"
    source_labels_dir = args.source / "labels"
    
    if not source_images_dir.exists():
        print(f"[ERROR] Images directory not found: {source_images_dir}")
        return 1
    
    if not source_labels_dir.exists():
        print(f"[ERROR] Labels directory not found: {source_labels_dir}")
        return 1
    
    # Get image-label pairs
    print(f"[INFO] Scanning {args.source} for image-label pairs...")
    pairs = get_image_label_pairs(source_images_dir, source_labels_dir)
    print(f"[INFO] Found {len(pairs)} valid image-label pairs")
    
    if len(pairs) == 0:
        print("[ERROR] No valid image-label pairs found!")
        return 1
    
    # Split the pairs
    train_pairs, val_pairs, test_pairs = split_pairs(pairs, args.train_ratio, args.val_ratio, args.test_ratio)
    
    print(f"[INFO] Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    
    if args.dry_run:
        print("[DRY RUN] Would copy files to:")
        print(f"  Train: {args.output / 'train'}")
        print(f"  Val: {args.output / 'valid'}")
        print(f"  Test: {args.output / 'test'}")
        return 0
    
    # Copy files to splits
    copy_pairs_to_split(train_pairs, args.output, "train")
    copy_pairs_to_split(val_pairs, args.output, "valid")
    copy_pairs_to_split(test_pairs, args.output, "test")
    
    # Update data.yaml
    if args.class_names:
        update_data_yaml(args.output, args.class_names)
    else:
        print("[INFO] No class names provided, skipping data.yaml update")
    
    print(f"[SUCCESS] Dataset split completed in {args.output}")
    return 0


if __name__ == "__main__":
    exit(main())