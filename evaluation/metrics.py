"""
Evaluation Metrics Module
Comprehensive metrics for token classification and table detection.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    classification_report
)
from seqeval.metrics import (
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
    classification_report as seqeval_report
)
import logging

logger = logging.getLogger(__name__)


def compute_token_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_list: List[str],
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Compute token classification metrics.
    
    Args:
        predictions: Predicted label IDs (batch_size, seq_len)
        labels: True label IDs (batch_size, seq_len)
        label_list: List of label names
        ignore_index: Label index to ignore (padding)
        
    Returns:
        Dictionary of metrics
    """
    # Flatten arrays
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Remove ignored indices
    mask = labels != ignore_index
    predictions = predictions[mask]
    labels = labels[mask]
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        average='macro',
        zero_division=0
    )
    
    # Weighted metrics (account for class imbalance)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='weighted',
        zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
    }
    
    return metrics


def compute_seqeval_metrics(
    predictions: List[List[int]],
    labels: List[List[int]],
    label_list: List[str],
    ignore_index: int = -100
) -> Dict[str, float]:
    """
    Compute sequence-level metrics using seqeval (better for NER).
    
    Args:
        predictions: List of predicted sequences
        labels: List of true label sequences
        label_list: List of label names
        ignore_index: Label index to ignore
        
    Returns:
        Dictionary of metrics
    """
    # Convert IDs to label names
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_filtered = []
        
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != ignore_index:
                true_seq.append(label_list[label_id])
                pred_seq_filtered.append(label_list[pred_id])
        
        if true_seq:  # Only add non-empty sequences
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_filtered)
    
    # Compute seqeval metrics
    precision = seqeval_precision(true_labels, pred_labels)
    recall = seqeval_recall(true_labels, pred_labels)
    f1 = seqeval_f1(true_labels, pred_labels)
    
    metrics = {
        'seqeval_precision': precision,
        'seqeval_recall': recall,
        'seqeval_f1': f1
    }
    
    # Get detailed report
    report = seqeval_report(true_labels, pred_labels, output_dict=True)
    
    # Add per-entity metrics for key entities
    key_entities = ['INVOICE_NUMBER', 'DATE', 'TOTAL_AMOUNT', 'VENDOR_NAME']
    for entity in key_entities:
        if entity in report:
            metrics[f'{entity.lower()}_f1'] = report[entity]['f1-score']
    
    return metrics


def compute_table_detection_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 3
) -> Dict[str, float]:
    """
    Compute metrics for table structure detection.
    
    Args:
        predictions: Predicted table labels
        labels: True table labels
        num_classes: Number of table classes
        
    Returns:
        Dictionary of metrics
    """
    # Flatten
    predictions = predictions.flatten()
    labels = labels.flatten()
    
    # Overall accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average=None,
        labels=range(num_classes),
        zero_division=0
    )
    
    # Class names
    class_names = ['non-table', 'table-row', 'table-header']
    
    metrics = {'table_accuracy': accuracy}
    
    for i, class_name in enumerate(class_names):
        if i < len(precision):
            metrics[f'table_{class_name}_precision'] = precision[i]
            metrics[f'table_{class_name}_recall'] = recall[i]
            metrics[f'table_{class_name}_f1'] = f1[i]
    
    # Macro average for tables
    metrics['table_macro_f1'] = np.mean(f1)
    
    return metrics


def compute_iou(
    pred_boxes: List[List[float]],
    true_boxes: List[List[float]]
) -> float:
    """
    Compute IoU (Intersection over Union) for bounding boxes.
    
    Args:
        pred_boxes: Predicted boxes [[x1, y1, x2, y2], ...]
        true_boxes: True boxes [[x1, y1, x2, y2], ...]
        
    Returns:
        Mean IoU score
    """
    ious = []
    
    for pred_box, true_box in zip(pred_boxes, true_boxes):
        # Calculate intersection
        x1 = max(pred_box[0], true_box[0])
        y1 = max(pred_box[1], true_box[1])
        x2 = min(pred_box[2], true_box[2])
        y2 = min(pred_box[3], true_box[3])
        
        if x2 < x1 or y2 < y1:
            iou = 0.0
        else:
            intersection = (x2 - x1) * (y2 - y1)
            
            # Calculate union
            pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            true_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])
            union = pred_area + true_area - intersection
            
            iou = intersection / union if union > 0 else 0.0
        
        ious.append(iou)
    
    return np.mean(ious) if ious else 0.0


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    label_list: Optional[List[str]] = None,
    compute_seqeval: bool = True
) -> Dict[str, float]:
    """
    Compute all relevant metrics.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        label_list: List of label names
        compute_seqeval: Whether to compute seqeval metrics
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Token-level metrics
    if label_list:
        token_metrics = compute_token_classification_metrics(
            predictions,
            labels,
            label_list
        )
        metrics.update(token_metrics)
        
        # Sequence-level metrics
        if compute_seqeval and len(predictions.shape) == 2:
            try:
                seqeval_metrics = compute_seqeval_metrics(
                    predictions.tolist(),
                    labels.tolist(),
                    label_list
                )
                metrics.update(seqeval_metrics)
            except Exception as e:
                logger.warning(f"Could not compute seqeval metrics: {str(e)}")
    else:
        # Basic accuracy if no label list provided
        metrics['accuracy'] = accuracy_score(
            labels.flatten(),
            predictions.flatten()
        )
    
    return metrics


def print_detailed_report(
    predictions: List[List[int]],
    labels: List[List[int]],
    label_list: List[str],
    ignore_index: int = -100
) -> None:
    """
    Print detailed classification report.
    
    Args:
        predictions: Predicted sequences
        labels: True label sequences
        label_list: List of label names
        ignore_index: Label index to ignore
    """
    # Convert to label names
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        true_seq = []
        pred_seq_filtered = []
        
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != ignore_index:
                true_seq.append(label_list[label_id])
                pred_seq_filtered.append(label_list[pred_id])
        
        if true_seq:
            true_labels.append(true_seq)
            pred_labels.append(pred_seq_filtered)
    
    # Print report
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*80)
    print(seqeval_report(true_labels, pred_labels))
    print("="*80 + "\n")


class MetricsTracker:
    """Track metrics across multiple evaluations."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.history = []
    
    def add(self, metrics: Dict[str, float], epoch: int) -> None:
        """Add metrics for an epoch."""
        metrics_with_epoch = {'epoch': epoch, **metrics}
        self.history.append(metrics_with_epoch)
    
    def get_best(self, metric_name: str = 'f1') -> Tuple[int, Dict[str, float]]:
        """Get best epoch and metrics for a given metric."""
        if not self.history:
            return -1, {}
        
        best_epoch_data = max(
            self.history,
            key=lambda x: x.get(metric_name, float('-inf'))
        )
        
        return best_epoch_data['epoch'], best_epoch_data
    
    def get_latest(self) -> Dict[str, float]:
        """Get latest metrics."""
        return self.history[-1] if self.history else {}
    
    def save(self, path: str) -> None:
        """Save metrics history to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load metrics history from file."""
        import json
        with open(path, 'r') as f:
            self.history = json.load(f)
