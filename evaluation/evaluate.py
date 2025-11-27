"""
Evaluation Script
Evaluate trained model on test set with comprehensive metrics.
"""

from typing import Dict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml
import logging
import json
from tqdm import tqdm
import numpy as np

from models import LayoutLMv3ForTokenClassification
from preprocessing import InvoiceDataset, collate_fn
from evaluation.metrics import (
    compute_metrics,
    print_detailed_report,
    compute_table_detection_metrics,
    MetricsTracker
)
from transformers import LayoutLMv3Processor

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for trained LayoutLMv3 models."""
    
    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        device: str = "cuda"
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to training configuration
            device: Device to run evaluation on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get label mapping
        self.labels = self.config['labels']['token_classification']
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        # Initialize model
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            self.config['model']['name']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load processor
        self.processor = LayoutLMv3Processor.from_pretrained(
            self.config['model']['name']
        )
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False
    ) -> Dict:
        """
        Evaluate on a dataset.
        
        Args:
            dataloader: DataLoader for evaluation
            return_predictions: Whether to return raw predictions
            
        Returns:
            Dictionary with metrics and optionally predictions
        """
        all_predictions = []
        all_labels = []
        all_table_predictions = []
        all_table_labels = []
        total_loss = 0
        
        logger.info("Running evaluation...")
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }
            
            # Forward pass
            outputs = self.model(**batch)
            
            if outputs.loss is not None:
                total_loss += outputs.loss.item()
            
            # Get predictions
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Store predictions and labels
            all_predictions.append(predictions.cpu().numpy())
            if 'labels' in batch:
                all_labels.append(batch['labels'].cpu().numpy())
            
            # Table predictions if available
            if 'table_labels' in batch:
                all_table_labels.append(batch['table_labels'].cpu().numpy())
                # Assuming model has table prediction head
                # all_table_predictions.append(...)
        
        # Concatenate all batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0) if all_labels else None
        
        results = {}
        
        # Compute token classification metrics
        if all_labels is not None:
            metrics = compute_metrics(
                all_predictions,
                all_labels,
                self.labels,
                compute_seqeval=True
            )
            results['metrics'] = metrics
            
            # Print detailed report
            print_detailed_report(
                all_predictions.tolist(),
                all_labels.tolist(),
                self.labels
            )
            
            # Log key metrics
            logger.info("\nEvaluation Results:")
            logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
            logger.info(f"  Precision: {metrics.get('precision', 0):.4f}")
            logger.info(f"  Recall: {metrics.get('recall', 0):.4f}")
            logger.info(f"  F1 Score: {metrics.get('f1', 0):.4f}")
            logger.info(f"  SeqEval F1: {metrics.get('seqeval_f1', 0):.4f}")
        
        # Compute table metrics if available
        if all_table_labels and all_table_predictions:
            table_metrics = compute_table_detection_metrics(
                np.concatenate(all_table_predictions, axis=0),
                np.concatenate(all_table_labels, axis=0)
            )
            results['table_metrics'] = table_metrics
            
            logger.info("\nTable Detection Results:")
            for k, v in table_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
        
        results['avg_loss'] = total_loss / len(dataloader)
        
        if return_predictions:
            results['predictions'] = all_predictions
            results['labels'] = all_labels
        
        return results
    
    def evaluate_test_set(self, output_path: Optional[Path] = None) -> Dict:
        """
        Evaluate on test set.
        
        Args:
            output_path: Path to save evaluation results
            
        Returns:
            Evaluation results dictionary
        """
        # Create test dataset
        test_dataset = InvoiceDataset(
            Path(self.config['data']['test_data_path']),
            self.processor,
            self.label2id,
            max_length=self.config['data']['max_seq_length'],
            augmentation=None,
            mode="test"
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            collate_fn=collate_fn
        )
        
        # Run evaluation
        results = self.evaluate_dataset(test_loader, return_predictions=False)
        
        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Saved evaluation results to {output_path}")
        
        return results
    
    def predict_single_document(
        self,
        image_path: Path,
        ocr_data: Dict
    ) -> Dict:
        """
        Make predictions on a single document.
        
        Args:
            image_path: Path to document image
            ocr_data: OCR data with words and boxes
            
        Returns:
            Predictions with labels
        """
        from PIL import Image
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Extract OCR data
        words = [w['text'] for w in ocr_data['words']]
        boxes = [w['bbox'] for w in ocr_data['words']]
        
        # Normalize boxes
        width, height = image.size
        normalized_boxes = [
            [
                int(1000 * box[0] / width),
                int(1000 * box[1] / height),
                int(1000 * box[2] / width),
                int(1000 * box[3] / height)
            ]
            for box in boxes
        ]
        
        # Encode
        encoding = self.processor(
            image,
            words,
            boxes=normalized_boxes,
            padding="max_length",
            truncation=True,
            max_length=self.config['data']['max_seq_length'],
            return_tensors="pt"
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        predictions = torch.argmax(outputs.logits, dim=-1)
        predictions = predictions.squeeze(0).cpu().numpy()
        
        # Map back to labels
        predicted_labels = [self.id2label[pred] for pred in predictions[:len(words)]]
        
        # Combine with words and boxes
        results = []
        for word, box, label in zip(words, boxes, predicted_labels):
            results.append({
                'text': word,
                'bbox': box,
                'label': label
            })
        
        return {
            'predictions': results,
            'image_path': str(image_path)
        }


def evaluate_model(
    model_path: str,
    config_path: str = "./configs/training_config.yaml",
    output_path: Optional[str] = None
) -> None:
    """
    Evaluate trained model.
    
    Args:
        model_path: Path to model checkpoint
        config_path: Path to configuration file
        output_path: Path to save results
    """
    evaluator = Evaluator(
        Path(model_path),
        Path(config_path),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    results = evaluator.evaluate_test_set(
        Path(output_path) if output_path else None
    )
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate LayoutLMv3 model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/training_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./logs/test_results.json",
        help="Path to save results"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    evaluate_model(args.model, args.config, args.output)
