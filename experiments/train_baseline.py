"""
EcoSort Training Entry Script
Usage:
    python experiments/train_baseline.py --config configs/baseline_resnet50.yaml
"""

import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import create_dataloaders
from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model
from src.train.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict):
    """Create model architecture based on configuration"""
    model_type = config['model']['type']

    if model_type == 'resnet':
        return create_resnet_model(
            backbone=config['model']['backbone'],
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout=config['model'].get('dropout', 0.3),
            use_attention=config['model'].get('use_attention', False)
        )
    elif model_type == 'efficientnet':
        return create_efficientnet_model(
            backbone=config['model']['backbone'],
            num_classes=config['model']['num_classes'],
            pretrained=config['model']['pretrained'],
            dropout=config['model'].get('dropout', 0.3)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description='EcoSort Training Pipeline')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML configuration file')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override dataset root directory (overrides config)')
    parser.add_argument('--exp-name', type=str, default=None,
                        help='Custom experiment name (overrides config value)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Base directory for saving model checkpoints')

    args = parser.parse_args()

    # Load base configuration
    config = load_config(args.config)

    # Override config values with command-line arguments
    if args.data_root:
        config['data']['root_dir'] = args.data_root
    if args.exp_name:
        config['experiment_name'] = args.exp_name
    else:
        # Use config filename as default experiment name
        config['experiment_name'] = Path(args.config).stem

    print(f"\n{'='*60}")
    print(f"EcoSort Training: {config['experiment_name']}")
    print(f"{'='*60}\n")

    # Print configuration for verification
    print("Training Configuration:")
    print(yaml.dump(config, default_flow_style=False))

    # Create data loaders
    print("\nInitializing data loaders...")
    try:
        config_class_names = config.get('data', {}).get('class_names')
        # Detect strong augmentation (random erasing enabled)
        strong_aug = config.get('augmentation', {}).get('random_erasing_prob', 0) > 0
        
        train_loader, val_loader = create_dataloaders(
            data_root=config['data']['root_dir'],
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            img_size=config['data']['img_size'],
            val_split=config['data']['val_split'],
            class_names=config_class_names,
            strong_aug=strong_aug
        )

        # Dynamically sync dataset metadata with config
        inferred_class_names = train_loader.dataset.class_names
        config['model']['num_classes'] = len(inferred_class_names)
        config['data']['class_names'] = inferred_class_names

        # Get class distribution for potential weighting
        class_counts = train_loader.dataset.get_class_distribution()
        ordered_counts = [class_counts[name] for name in inferred_class_names]
        config['data']['class_counts'] = ordered_counts

        print(f"Successfully detected {len(inferred_class_names)} classes")
        print(f"Class names: {inferred_class_names}")
        print(f"Class distribution: {dict(zip(inferred_class_names, ordered_counts))}")
        
    except Exception as e:
        print(f"Error initializing data loaders: {str(e)}")
        print("\nPlease ensure your dataset is organized in the following structure:")
        print("data/raw/")
        print("  ├── recyclable/")
        print("  ├── hazardous/")
        print("  ├── kitchen/")
        print("  └── other/")
        return

    # Initialize model architecture
    print("\nBuilding model architecture...")
    model = create_model(config)

    # Initialize training manager
    print("\nInitializing training manager...")
    # Compile full trainer configuration
    trainer_config = dict(config['training'])
    trainer_config['data'] = config.get('data', {})
    trainer_config['loss'] = config.get('loss', {})

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=config['experiment_name'],
        use_wandb=not args.no_wandb
    )

    # Start training process
    print("\nStarting model training...\n")
    trainer.train()

    # Training completion
    checkpoint_path = Path(args.checkpoint_dir) / config['experiment_name']
    print("\nTraining completed successfully!")
    print(f"Model checkpoints saved to: {checkpoint_path.absolute()}")


if __name__ == '__main__':
    main()
