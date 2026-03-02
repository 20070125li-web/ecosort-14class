#!/usr/bin/env python3
"""
EcoSort V6 Data Preprocessing Script
Maps raw dataset to EcoSort V6 15-class taxonomy and splits into train/validation sets
"""

import os
import yaml
import json
import shutil
from pathlib import Path
from collections import defaultdict
import random
import argparse


def load_mapping(mapping_file="configs/dataset_mapping.yaml"):
    """Load V6 15-class taxonomy mapping configuration"""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['daily_life_mapping_v6_15class']


def scan_raw_data(raw_dir="data/raw"):
    """Scan raw dataset directory and collect image paths by class"""
    raw_dir = Path(raw_dir)
    class_images = defaultdict(list)

    for class_dir in raw_dir.iterdir():
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        if class_name in ['Liquid', 'trash']:  # Skip deprecated classes
            continue

        # Collect all image files with common extensions
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            for img_path in class_dir.glob(ext):
                class_images[class_name].append(img_path)

    return class_images


def map_classes(class_images, mapping_config):
    """Map raw classes to EcoSort V6 15-class taxonomy"""
    mapped_images = defaultdict(list)

    drop_classes = set(mapping_config.get('drop_raw_classes', []))
    raw_to_target = mapping_config['raw_to_target']

    for raw_class, images in class_images.items():
        if raw_class in drop_classes:
            print(f"  ⚠️  Skipping dropped class: {raw_class}")
            continue

        if raw_class not in raw_to_target:
            print(f"  ⚠️  Unmapped class: {raw_class}")
            continue

        target_class = raw_to_target[raw_class]
        mapped_images[target_class].extend(images)

    return mapped_images


def split_dataset(mapped_images, val_split=0.2, seed=42):
    """Split dataset into training and validation sets with fixed random seed"""
    random.seed(seed)

    train_data = []
    val_data = []

    for class_name, images in mapped_images.items():
        # Shuffle images for random split
        random.shuffle(images)

        # Calculate split index
        split_idx = int(len(images) * (1 - val_split))
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Create training data entries
        for img_path in train_imgs:
            train_data.append({
                'image_path': str(img_path),
                'label': class_name
            })

        # Create validation data entries
        for img_path in val_imgs:
            val_data.append({
                'image_path': str(img_path),
                'label': class_name
            })

    return train_data, val_data


def save_processed_data(train_data, val_data, output_dir, class_names):
    """Save processed dataset metadata (JSON format for dynamic loading)
    
    Creates directory structure and saves metadata JSON file containing:
    - Class names and count
    - Train/validation split information
    - Image paths and labels for each split
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directory structure for train/validation splits
    for split, data in [('train', train_data), ('val', val_data)]:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)

        for class_name in class_names:
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)

    # Create metadata dictionary with complete dataset information
    metadata = {
        'class_names': class_names,
        'num_classes': len(class_names),
        'num_train': len(train_data),
        'num_val': len(val_data),
        'train': train_data,
        'val': val_data
    }

    # Save metadata to JSON file
    with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata


def print_statistics(mapped_images, train_data, val_data):
    """Print comprehensive dataset statistics"""
    print("\n" + "="*60)
    print("📊 EcoSort V6 15-Class Dataset Statistics")
    print("="*60)

    print("\nOriginal class distribution:")
    for class_name in sorted(mapped_images.keys()):
        count = len(mapped_images[class_name])
        print(f"  {class_name:25s}: {count:4d} images")

    print(f"\n{'='*60}")
    print("Split results:")
    print(f"  Training set: {len(train_data)} images")
    print(f"  Validation set: {len(val_data)} images")
    print(f"  Total: {len(train_data) + len(val_data)} images")
    print(f"  Number of classes: {len(mapped_images)}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="EcoSort V6 Data Preprocessing")
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                       help='Directory containing raw dataset')
    parser.add_argument('--output_dir', type=str,
                       default='data/proc/ecosort_v6_15class',
                       help='Output directory for processed dataset')
    parser.add_argument('--mapping_file', type=str,
                       default='configs/dataset_mapping.yaml',
                       help='Path to V6 class mapping configuration file')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Fraction of data to use for validation (0.0-1.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible train/val split')

    args = parser.parse_args()

    print("="*60)
    print("🔄 EcoSort V6 Data Preprocessing Pipeline")
    print("="*60)
    print(f"📂 Raw data directory: {args.raw_dir}")
    print(f"📂 Output directory: {args.output_dir}")
    print(f"📋 Mapping config: {args.mapping_file}")
    print(f"🔀 Validation split ratio: {args.val_split}")
    print(f"🎲 Random seed: {args.seed}")
    print("="*60)

    # Step 1: Load V6 class mapping configuration
    print("\n📋 Loading V6 15-class mapping configuration...")
    mapping_config = load_mapping(args.mapping_file)
    print(f"  ✅ Target classes loaded: {len(mapping_config['target_classes'])} classes")

    # Step 2: Scan raw dataset directory
    print("\n🔍 Scanning raw dataset directory...")
    class_images = scan_raw_data(args.raw_dir)
    total_raw = sum(len(imgs) for imgs in class_images.values())
    print(f"  ✅ Found {len(class_images)} classes with {total_raw} total images")

    # Step 3: Apply V6 class mapping
    print("\n🔄 Applying V6 15-class taxonomy mapping...")
    mapped_images = map_classes(class_images, mapping_config)
    total_mapped = sum(len(imgs) for imgs in mapped_images.values())
    print(f"  ✅ Mapped to {len(mapped_images)} target classes with {total_mapped} images")

    # Step 4: Split into training/validation sets
    print("\n🔀 Splitting dataset into train/validation sets...")
    train_data, val_data = split_dataset(mapped_images, args.val_split, args.seed)
    print(f"  ✅ Training set: {len(train_data)} images")
    print(f"  ✅ Validation set: {len(val_data)} images")

    # Step 5: Save processed dataset metadata
    print(f"\n💾 Saving processed data to {args.output_dir}...")
    # Extract sorted class names
    class_names = sorted(set([d['label'] for d in train_data]))
    metadata = save_processed_data(train_data, val_data, args.output_dir, class_names)
    print(f"  ✅ Metadata saved to: {args.output_dir}/metadata.json")

    # Step 6: Print comprehensive dataset statistics
    print_statistics(mapped_images, train_data, val_data)

    print("\n✅ Data preprocessing completed successfully!")
    print(f"📂 Processed dataset location: {args.output_dir}")
    print(f"📄 Metadata file: {args.output_dir}/metadata.json")


if __name__ == "__main__":
    main()