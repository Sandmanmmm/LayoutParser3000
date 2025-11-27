"""
Dataset Module - Production Grade
Custom dataset for LayoutLMv3 with invoice/PO documents.

Features:
- 115-label NER support (57 entity types)
- Multi-task labels: NER + cell detection + column classification
- Subword tokenization with proper label alignment
- Table structure annotation support
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import LayoutLMv3Processor
import logging

logger = logging.getLogger(__name__)


class InvoiceDataset(Dataset):
    """
    Production dataset for invoice and purchase order documents.
    
    Supports:
    - 115-label NER (57 entity types × 2 + O)
    - Binary cell detection (in-table vs not)
    - Column classification (0-15)
    - Subword tokenization with proper alignment
    """
    
    def __init__(
        self,
        data_dir: Path,
        processor: LayoutLMv3Processor,
        label2id: Dict[str, int],
        max_length: int = 512,
        augmentation=None,
        mode: str = "train"
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing JSON annotation files
            processor: LayoutLMv3 processor for tokenization
            label2id: Mapping from label names to IDs (115 labels)
            max_length: Maximum sequence length for truncation
            augmentation: Optional augmentation pipeline
            mode: train, val, or test
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.max_length = max_length
        self.augmentation = augmentation
        self.mode = mode
        
        # Load all annotation files
        self.samples = list(self.data_dir.glob("**/*.json"))
        if not self.samples:
            # Try JSONL format
            self.samples = list(self.data_dir.glob("**/*.jsonl"))
        
        logger.info(
            f"Loaded {len(self.samples)} samples from {data_dir} "
            f"with {len(label2id)} labels"
        )
        
        # Validate label schema
        if len(label2id) != 115:
            logger.warning(
                f"Expected 115 labels, got {len(label2id)}. "
                "Ensure label_list.txt has 57 entity types."
            )
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample with multi-task labels.
        
        Returns:
            Dictionary with:
            - input_ids, attention_mask, bbox, pixel_values
            - labels: NER labels (115 classes)
            - cell_labels: Binary cell detection (0/1)
            - col_labels: Column indices (0-15 or -100)
        """
        annotation_path = self.samples[idx]
        
        try:
            # Load annotation
            annotation = self._load_annotation(annotation_path)
            
            # Extract OCR tokens
            ocr_tokens = annotation.get('ocr', [])
            if not ocr_tokens:
                logger.warning(f"No OCR tokens in {annotation_path}")
                return self._get_dummy_sample()
            
            # Extract data
            words = [token['text'] for token in ocr_tokens]
            boxes = [token['bbox'] for token in ocr_tokens]
            ner_tags = annotation.get('ner_tags', ['O'] * len(words))
            tables = annotation.get('tables', [])
            
            # Load image
            image = self._load_image(annotation, annotation_path)
            if image is None:
                return self._get_dummy_sample()
            
            # Derive multi-task labels at word level
            word_cell_labels = self._derive_cell_labels(ocr_tokens, tables)
            word_col_labels = self._derive_col_labels(ocr_tokens, tables)
            
            # Apply augmentation if training
            if self.mode == "train" and self.augmentation:
                image, boxes, ner_tags = self._apply_augmentation(
                    image, boxes, ner_tags
                )
            
            # Convert NER tags to IDs
            word_ner_ids = [
                self.label2id.get(tag, 0) for tag in ner_tags
            ]
            
            # Normalize boxes to [0, 1000] scale
            width, height = image.size
            normalized_boxes = self._normalize_boxes(boxes, width, height)
            
            # Tokenize with LayoutLMv3 processor
            encoding = self.processor(
                image,
                words,
                boxes=normalized_boxes,
                word_labels=word_ner_ids,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Align word-level labels to subword tokens
            token_ner_labels = self._align_labels_to_tokens(
                encoding,
                word_ner_ids
            )
            token_cell_labels = self._align_labels_to_tokens(
                encoding,
                word_cell_labels
            )
            token_col_labels = self._align_labels_to_tokens(
                encoding,
                word_col_labels
            )
            
            # Build output
            item = {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'bbox': encoding['bbox'].squeeze(0),
                'pixel_values': encoding['pixel_values'].squeeze(0),
                'labels': token_ner_labels,
                'cell_labels': token_cell_labels,
                'col_labels': token_col_labels,
            }
            
            return item
            
        except Exception as e:
            logger.error(
                f"Error loading sample {annotation_path}: {str(e)}",
                exc_info=True
            )
            return self._get_dummy_sample()
    
    def _load_annotation(self, annotation_path: Path) -> Dict[str, Any]:
        """Load annotation from JSON or JSONL file."""
        with open(annotation_path, 'r', encoding='utf-8') as f:
            if annotation_path.suffix == '.jsonl':
                # JSONL: one sample per line, take first
                line = f.readline()
                return json.loads(line)
            else:
                return json.load(f)
    
    def _load_image(
        self,
        annotation: Dict[str, Any],
        annotation_path: Path
    ) -> Image.Image:
        """Load and validate document image."""
        # Try metadata first
        image_path = annotation.get('metadata', {}).get('file_name')
        
        # Try direct image_path
        if not image_path:
            image_path = annotation.get('image_path')
        
        # Build full path
        if image_path:
            image_path = Path(image_path)
            if not image_path.exists():
                # Try relative to annotation
                image_path = annotation_path.parent / image_path.name
            
            if not image_path.exists():
                # Try ../raw/ directory
                image_path = (
                    annotation_path.parent.parent /
                    'raw' /
                    image_path.name
                )
        
        if not image_path or not image_path.exists():
            logger.error(f"Image not found for {annotation_path}")
            return None
        
        return Image.open(image_path).convert("RGB")
    
    def _derive_cell_labels(
        self,
        ocr_tokens: List[Dict[str, Any]],
        tables: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Derive binary cell labels (in-table vs not).
        
        Args:
            ocr_tokens: List of OCR tokens with token_id
            tables: List of table annotations with cells
            
        Returns:
            List of 0/1 labels (same length as ocr_tokens)
        """
        cell_labels = [0] * len(ocr_tokens)
        
        # Build token_id to index mapping
        token_id_to_idx = {
            token['token_id']: idx
            for idx, token in enumerate(ocr_tokens)
        }
        
        # Mark tokens that are in table cells
        for table in tables:
            for cell in table.get('cells', []):
                for token_id in cell.get('token_ids', []):
                    if token_id in token_id_to_idx:
                        idx = token_id_to_idx[token_id]
                        cell_labels[idx] = 1
        
        return cell_labels
    
    def _derive_col_labels(
        self,
        ocr_tokens: List[Dict[str, Any]],
        tables: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Derive column index labels (0-15 or -100).
        
        Args:
            ocr_tokens: List of OCR tokens
            tables: List of table annotations
            
        Returns:
            List of column indices (-100 for non-table tokens)
        """
        col_labels = [-100] * len(ocr_tokens)
        
        # Build token_id to index mapping
        token_id_to_idx = {
            token['token_id']: idx
            for idx, token in enumerate(ocr_tokens)
        }
        
        # Assign column indices
        for table in tables:
            for cell in table.get('cells', []):
                col_idx = cell.get('col', -100)
                # Cap at 15 (our model supports 16 columns: 0-15)
                if col_idx > 15:
                    col_idx = 15
                
                for token_id in cell.get('token_ids', []):
                    if token_id in token_id_to_idx:
                        idx = token_id_to_idx[token_id]
                        col_labels[idx] = col_idx
        
        return col_labels
    
    def _apply_augmentation(
        self,
        image: Image.Image,
        boxes: List[List[int]],
        ner_tags: List[str]
    ) -> tuple:
        """Apply augmentation to image and annotations."""
        image_array = np.array(image)
        label_ids = [self.label2id.get(tag, 0) for tag in ner_tags]
        
        image_array, boxes, label_ids = self.augmentation.augment_image(
            image_array, boxes, label_ids
        )
        
        ner_tags = [self.id2label.get(lid, 'O') for lid in label_ids]
        image = Image.fromarray(image_array)
        
        return image, boxes, ner_tags
    
    def _normalize_boxes(
        self,
        boxes: List[List[int]],
        width: int,
        height: int
    ) -> List[List[int]]:
        """Normalize bounding boxes to [0, 1000] scale."""
        normalized = []
        for box in boxes:
            normalized.append([
                int(1000 * box[0] / width),
                int(1000 * box[1] / height),
                int(1000 * box[2] / width),
                int(1000 * box[3] / height)
            ])
        return normalized
    
    def _align_labels_to_tokens(
        self,
        encoding: Dict[str, Any],
        word_labels: List[int]
    ) -> torch.Tensor:
        """
        Align word-level labels to subword tokens.
        
        Uses -100 for special tokens and subword continuations.
        
        Args:
            encoding: Output from LayoutLMv3Processor
            word_labels: Labels for each word (before tokenization)
            
        Returns:
            Tensor of labels aligned to tokens
        """
        # Get word_ids mapping from encoding
        word_ids = encoding.word_ids(batch_index=0)
        
        aligned_labels = []
        previous_word_id = None
        
        for word_id in word_ids:
            if word_id is None:
                # Special token (CLS, SEP, PAD)
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                # First subword of a word
                if word_id < len(word_labels):
                    aligned_labels.append(word_labels[word_id])
                else:
                    aligned_labels.append(-100)
            else:
                # Continuation subword - use -100 to ignore in loss
                aligned_labels.append(-100)
            
            previous_word_id = word_id
        
        return torch.tensor(aligned_labels, dtype=torch.long)
    
    def _get_dummy_sample(self) -> Dict[str, Any]:
        """Return a dummy sample in case of loading error."""
        return {
            'input_ids': torch.zeros(self.max_length, dtype=torch.long),
            'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
            'bbox': torch.zeros((self.max_length, 4), dtype=torch.long),
            'pixel_values': torch.zeros((3, 224, 224), dtype=torch.float),
            'labels': torch.full((self.max_length,), -100, dtype=torch.long),
            'cell_labels': torch.full(
                (self.max_length,), -100, dtype=torch.long
            ),
            'col_labels': torch.full(
                (self.max_length,), -100, dtype=torch.long
            ),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching with multi-task labels.
    
    Args:
        batch: List of samples from InvoiceDataset
        
    Returns:
        Batched tensors with:
        - input_ids, attention_mask, bbox, pixel_values
        - labels (NER), cell_labels, col_labels
    """
    # Stack all tensors
    collated = {}
    
    # Expected keys
    expected_keys = [
        'input_ids',
        'attention_mask',
        'bbox',
        'pixel_values',
        'labels',
        'cell_labels',
        'col_labels'
    ]
    
    for key in expected_keys:
        if key in batch[0]:
            try:
                collated[key] = torch.stack([item[key] for item in batch])
            except Exception as e:
                logger.warning(f"Could not collate key {key}: {str(e)}")
                # Fallback: return as list
                collated[key] = [item[key] for item in batch]
        else:
            logger.warning(f"Key {key} not found in batch samples")
    
    return collated


def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    processor: LayoutLMv3Processor,
    label2id: Dict[str, int],
    batch_size: int = 8,
    num_workers: int = 4,
    train_augmentation=None
) -> tuple:
    """
    Create train and validation dataloaders.
    
    Args:
        train_dir: Training data directory
        val_dir: Validation data directory
        processor: LayoutLMv3 processor
        label2id: Label to ID mapping
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_augmentation: Augmentation for training
        
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader
    
    train_dataset = InvoiceDataset(
        train_dir,
        processor,
        label2id,
        augmentation=train_augmentation,
        mode="train"
    )
    
    val_dataset = InvoiceDataset(
        val_dir,
        processor,
        label2id,
        augmentation=None,
        mode="val"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
