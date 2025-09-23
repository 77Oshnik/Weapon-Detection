#!/usr/bin/env python3
"""
Merge multiple YOLO datasets into a unified multi-class weapon detection dataset.
Handles class remapping, file copying, and creates a unified data.yaml.

Usage:
  python scripts/merge_datasets.py --output dataset/unified_weapon_dataset --config scripts/merge_config.yaml
"""

import argparse
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_dataset_config(config_path: Path) -> Dict:
    """Load dataset merge configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def remap_label_file(label_path: Path, class_mapping: Dict[int, int]) -> List[str]:
    """Remap class IDs in a YOLO label file and return new lines."""
    if not label_path.exists():
        return []
    
    new_lines = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:  # class x y w h [confidence]
                old_class_id = int(parts[0])
                if old_class_id in class_mapping:
                    new_class_id = class_mapping[old_class_id]
                    parts[0] = str(new_class_id)
                    new_lines.append(' '.join(parts))
                else:
                    print(f"[WARN] Unknown class ID {old_class_id} in {label_path}")
    
    return new_lines


def process_dataset(dataset_config: Dict, output_dir: Path, split_name: str):
    """Process one dataset split (train/val/test)."""
    dataset_name = dataset_config['name']
    dataset_path = Path(dataset_config['path'])
    class_mapping = dataset_config['class_mapping']
    
    # Source paths
    if split_name == 'val':
        # Handle both 'val' and 'valid' naming
        src_images_dir = dataset_path / 'valid' / 'images'
        src_labels_dir = dataset_path / 'valid' / 'labels'
        if not src_images_dir.exists():
            src_images_dir = dataset_path / 'val' / 'images' 
            src_labels_dir = dataset_path / 'val' / 'labels'
    else:
        src_images_dir = dataset_path / split_name / 'images'
        src_labels_dir = dataset_path / split_name / 'labels'
    
    # Destination paths
    dst_images_dir = output_dir / split_name / 'images'
    dst_labels_dir = output_dir / split_name / 'labels'
    
    ensure_dir(dst_images_dir)
    ensure_dir(dst_labels_dir)
    
    if not src_images_dir.exists():
        print(f"[WARN] {dataset_name} {split_name} images not found: {src_images_dir}")
        return 0, 0
    
    copied_images = 0
    copied_labels = 0
    
    # Process all images in the source directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    for img_path in src_images_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue
            
        # Copy image with dataset prefix to avoid name conflicts
        dst_img_name = f"{dataset_name}_{img_path.name}"
        dst_img_path = dst_images_dir / dst_img_name
        shutil.copy2(img_path, dst_img_path)
        copied_images += 1
        
        # Process corresponding label file
        src_label_path = src_labels_dir / f"{img_path.stem}.txt"
        dst_label_path = dst_labels_dir / f"{dst_img_name.rsplit('.', 1)[0]}.txt"
        
        if src_label_path.exists():
            # Remap class IDs and write new label file
            new_lines = remap_label_file(src_label_path, class_mapping)
            with open(dst_label_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
                if new_lines:  # Add final newline if file has content
                    f.write('\n')
            copied_labels += 1
        else:
            # Create empty label file (for no_weapon images)
            dst_label_path.write_text('', encoding='utf-8')
            copied_labels += 1
    
    print(f"[INFO] {dataset_name} {split_name}: {copied_images} images, {copied_labels} labels")
    return copied_images, copied_labels


def create_unified_data_yaml(output_dir: Path, unified_classes: List[str]):
    """Create data.yaml for the unified dataset."""
    data_yaml_content = f"""train: train/images
val: val/images
test: test/images

nc: {len(unified_classes)}
names: {unified_classes}

# Unified weapon detection dataset
# Classes: {', '.join(unified_classes)}
"""
    
    data_yaml_path = output_dir / 'data.yaml'
    data_yaml_path.write_text(data_yaml_content, encoding='utf-8')
    print(f"[INFO] Created unified data.yaml with {len(unified_classes)} classes")


def main():
    parser = argparse.ArgumentParser(description="Merge YOLO datasets into unified multi-class dataset")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output directory for unified dataset")
    parser.add_argument("--config", type=Path, 
                        default=Path("scripts/merge_config.yaml"),
                        help="Configuration file for dataset merging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without copying files")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_dataset_config(args.config)
    
    print(f"[INFO] Merging {len(config['datasets'])} datasets into {args.output}")
    print(f"[INFO] Unified classes: {config['unified_classes']}")
    
    if args.dry_run:
        print("[DRY RUN] Would merge the following datasets:")
        for dataset in config['datasets']:
            print(f"  - {dataset['name']}: {dataset['path']}")
        return 0
    
    # Create output directory structure
    for split in ['train', 'val', 'test']:
        ensure_dir(args.output / split / 'images')
        ensure_dir(args.output / split / 'labels')
    
    # Process each dataset
    total_stats = defaultdict(lambda: defaultdict(int))
    
    for dataset_config in config['datasets']:
        print(f"\n[INFO] Processing dataset: {dataset_config['name']}")
        
        for split in ['train', 'val', 'test']:
            images, labels = process_dataset(dataset_config, args.output, split)
            total_stats[split]['images'] += images
            total_stats[split]['labels'] += labels
    
    # Create unified data.yaml
    create_unified_data_yaml(args.output, config['unified_classes'])
    
    # Print summary
    print(f"\n[SUMMARY] Unified dataset created in {args.output}")
    for split in ['train', 'val', 'test']:
        stats = total_stats[split]
        print(f"  {split}: {stats['images']} images, {stats['labels']} labels")
    
    total_images = sum(stats['images'] for stats in total_stats.values())
    total_labels = sum(stats['labels'] for stats in total_stats.values())
    print(f"  Total: {total_images} images, {total_labels} labels")
    
    return 0


if __name__ == "__main__":
    exit(main())