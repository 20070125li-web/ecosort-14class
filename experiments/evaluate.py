"""
EcoSort Model Evaluation Script
Comprehensive model evaluation and visualization
"""

import argparse
import yaml
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_score, recall_score
)
from tqdm import tqdm

from src.data.dataset import TrashDataset, get_data_transforms
from src.models.resnet_classifier import create_resnet_model
from src.models.efficientnet_classifier import create_efficientnet_model
from src.train.trainer import load_checkpoint


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> dict:
    """Evaluate model performance on test dataset

    Returns:
        metrics: Dictionary containing comprehensive evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate aggregate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # Per-class metrics
    class_names = TrashDataset.CLASS_NAMES
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True
    )

    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision_macro': precision,
        'recall_macro': recall,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

    return metrics


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: list,
    save_path: str = None
):
    """Plot confusion matrix visualization"""
    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")

    plt.show()


def plot_per_class_accuracy(
    report: dict,
    save_path: str = None
):
    """Plot per-class accuracy bar chart"""
    # Exclude aggregate metrics (macro avg, weighted avg, accuracy)
    class_names = [name for name in report.keys() if name not in ['macro avg', 'weighted avg', 'accuracy']]
    accuracies = [report[name]['precision'] for name in class_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, accuracies, color='steelblue')
    plt.ylim(0, 1.0)
    plt.title('Per-Class Accuracy', fontsize=16)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.xticks(rotation=45)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved per-class accuracy plot to {save_path}")

    plt.show()


def save_evaluation_report(
    metrics: dict,
    save_path: str
):
    """Save evaluation metrics to JSON file"""
    report = {
        'accuracy': float(metrics['accuracy']),
        'f1_macro': float(metrics['f1_macro']),
        'precision_macro': float(metrics['precision_macro']),
        'recall_macro': float(metrics['recall_macro']),
        'per_class_metrics': metrics['classification_report']
    }

    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Saved evaluation report to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='EcoSort Model Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint file (.pth)')
    parser.add_argument('--data-root', type=str, default='data/raw',
                        help='Root directory of dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--img-size', type=int, default=256,
                        help='Input image size (pixels)')
    parser.add_argument('--output-dir', type=str, default='checkpoints/evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--model-type', type=str, default='resnet',
                        choices=['resnet', 'efficientnet'],
                        help='Backbone model architecture type')

    args = parser.parse_args()

    # Create output directory if not exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("EcoSort Model Evaluation")
    print(f"{'='*60}\n")

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print("\nLoading test dataset...")
    test_dataset = TrashDataset(
        root_dir=args.data_root,
        transform=get_data_transforms('val', args.img_size),
        split='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Test dataset size: {len(test_dataset)} samples")

    # Get class information
    class_names = test_dataset.class_names
    num_classes = len(class_names)

    # Initialize model
    print("\nLoading model checkpoint...")
    if args.model_type == 'resnet':
        model = create_resnet_model(
            backbone='resnet50',
            num_classes=num_classes,
            pretrained=False
        )
    else:  # efficientnet
        model = create_efficientnet_model(
            backbone='efficientnet-b3',
            num_classes=num_classes,
            pretrained=False
        )

    # Load trained weights
    checkpoint = load_checkpoint(args.checkpoint, model)
    model = model.to(device)

    # Run evaluation
    print("\nRunning model evaluation...")
    metrics = evaluate_model(model, test_loader, device)

    # Print summary results
    print(f"\n{'='*60}")
    print("Evaluation Results Summary")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1-Score:  {metrics['f1_macro']:.4f}")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall:    {metrics['recall_macro']:.4f}")
    print(f"{'='*60}\n")

    # Print detailed per-class report
    print("Per-Class Classification Report:")
    print(classification_report(
        metrics['labels'], metrics['predictions'],
        target_names=class_names
    ))

    # Save evaluation results
    save_evaluation_report(metrics, output_dir / 'evaluation_report.json')

    # Generate visualizations
    plot_confusion_matrix(
        metrics['labels'],
        metrics['predictions'],
        class_names,
        save_path=output_dir / 'confusion_matrix.png'
    )

    plot_per_class_accuracy(
        metrics['classification_report'],
        save_path=output_dir / 'per_class_accuracy.png'
    )

    print(f"\nEvaluation completed successfully! All results saved to {output_dir}")


if __name__ == '__main__':
    main()
