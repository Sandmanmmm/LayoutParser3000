"""
Visualization Utilities
Visualize predictions, attention maps, and training progress.
"""

from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from pathlib import Path


def visualize_predictions(
    image_path: Path,
    predictions: List[Dict],
    save_path: Optional[Path] = None,
    show: bool = True
) -> None:
    """
    Visualize bounding boxes and predicted labels on image.
    
    Args:
        image_path: Path to document image
        predictions: List of predictions with text, bbox, and label
        save_path: Path to save visualization
        show: Whether to display the image
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Color map for different entity types
    color_map = {
        'INVOICE_NUMBER': (255, 0, 0),
        'DATE': (0, 255, 0),
        'VENDOR_NAME': (0, 0, 255),
        'TOTAL_AMOUNT': (255, 165, 0),
        'LINE_ITEM': (128, 0, 128),
        'O': (128, 128, 128)
    }
    
    # Draw each prediction
    for pred in predictions:
        bbox = pred['bbox']
        label = pred['label']
        
        # Get entity type (B-/I- prefix)
        entity_type = label.split('-')[1] if '-' in label else label
        color = color_map.get(entity_type, (200, 200, 200))
        
        # Draw rectangle
        draw.rectangle(bbox, outline=color, width=2)
        
        # Draw label
        text = f"{entity_type}: {pred['text']}"
        text_bbox = draw.textbbox((bbox[0], bbox[1] - 15), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((bbox[0], bbox[1] - 15), text, fill=(255, 255, 255), font=font)
    
    # Save or show
    if save_path:
        image.save(save_path)
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.figure(figsize=(15, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def plot_training_curves(
    metrics_history: List[Dict],
    save_path: Optional[Path] = None
) -> None:
    """
    Plot training and validation curves.
    
    Args:
        metrics_history: List of metrics dictionaries
        save_path: Path to save plot
    """
    epochs = [m['epoch'] for m in metrics_history]
    
    # Extract metrics
    train_loss = [m.get('train_loss', 0) for m in metrics_history]
    val_loss = [m.get('eval_loss', 0) for m in metrics_history]
    val_f1 = [m.get('eval_f1', 0) for m in metrics_history]
    val_acc = [m.get('accuracy', 0) for m in metrics_history]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(epochs, train_loss, label='Train Loss', marker='o')
    axes[0, 0].plot(epochs, val_loss, label='Val Loss', marker='s')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[0, 1].plot(epochs, val_f1, label='Validation F1', color='green', marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Validation F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(epochs, val_acc, label='Validation Accuracy', color='orange', marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Multiple metrics comparison
    axes[1, 1].plot(epochs, val_f1, label='F1', marker='o')
    axes[1, 1].plot(epochs, val_acc, label='Accuracy', marker='s')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Metrics Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    save_path: Optional[Path] = None,
    normalize: bool = True
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix array
        class_names: List of class names
        save_path: Path to save plot
        normalize: Whether to normalize values
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def visualize_attention(
    image_path: Path,
    attention_weights: np.ndarray,
    tokens: List[str],
    layer: int = -1,
    head: int = 0,
    save_path: Optional[Path] = None
) -> None:
    """
    Visualize attention weights on document image.
    
    Args:
        image_path: Path to document image
        attention_weights: Attention weights (num_layers, num_heads, seq_len, seq_len)
        tokens: List of tokens
        layer: Which layer to visualize (-1 for last)
        head: Which attention head to visualize
        save_path: Path to save visualization
    """
    # Get attention for specific layer and head
    attn = attention_weights[layer, head]
    
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Show image
    ax1.imshow(image)
    ax1.set_title('Document Image')
    ax1.axis('off')
    
    # Show attention heatmap
    im = ax2.imshow(attn, cmap='hot', interpolation='nearest')
    ax2.set_title(f'Attention Weights (Layer {layer}, Head {head})')
    ax2.set_xlabel('Key Position')
    ax2.set_ylabel('Query Position')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2)
    
    # Add token labels if not too many
    if len(tokens) <= 50:
        ax2.set_xticks(range(len(tokens)))
        ax2.set_yticks(range(len(tokens)))
        ax2.set_xticklabels(tokens, rotation=90, fontsize=8)
        ax2.set_yticklabels(tokens, fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    
    plt.show()


def create_entity_distribution_plot(
    predictions: List[Dict],
    save_path: Optional[Path] = None
) -> None:
    """
    Create bar plot of entity distribution.
    
    Args:
        predictions: List of predictions with labels
        save_path: Path to save plot
    """
    from collections import Counter
    
    # Count entities
    entity_counts = Counter()
    for pred in predictions:
        label = pred['label']
        if label != 'O':
            entity_type = label.split('-')[1] if '-' in label else label
            entity_counts[entity_type] += 1
    
    # Create plot
    entities = list(entity_counts.keys())
    counts = list(entity_counts.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(entities, counts, color='steelblue')
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.title('Entity Distribution in Predictions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved entity distribution to {save_path}")
    
    plt.show()
