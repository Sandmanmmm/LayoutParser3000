"""
Data Utilities
Helper functions for data processing and validation.
"""

from typing import List, Dict, Tuple
from pathlib import Path
import json
import shutil
from collections import defaultdict
import random
import logging

logger = logging.getLogger(__name__)


def split_dataset(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_key: str = None,
    seed: int = 42
) -> None:
    """
    Split dataset into train/val/test sets.
    
    Args:
        data_dir: Directory containing all data
        output_dir: Output directory for splits
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        stratify_key: Key to stratify splits on
        seed: Random seed
    """
    random.seed(seed)
    
    # Load all samples
    samples = list(data_dir.glob("**/*.json"))
    logger.info(f"Found {len(samples)} samples")
    
    # Stratify if needed
    if stratify_key:
        stratified_samples = defaultdict(list)
        for sample in samples:
            with open(sample, 'r') as f:
                data = json.load(f)
            key = data.get(stratify_key, 'unknown')
            stratified_samples[key].append(sample)
        
        # Split each stratum
        train_samples, val_samples, test_samples = [], [], []
        
        for key, stratum_samples in stratified_samples.items():
            random.shuffle(stratum_samples)
            n = len(stratum_samples)
            
            train_end = int(n * train_ratio)
            val_end = train_end + int(n * val_ratio)
            
            train_samples.extend(stratum_samples[:train_end])
            val_samples.extend(stratum_samples[train_end:val_end])
            test_samples.extend(stratum_samples[val_end:])
    else:
        # Simple random split
        random.shuffle(samples)
        n = len(samples)
        
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    for sample in train_samples:
        shutil.copy(sample, train_dir / sample.name)
    
    for sample in val_samples:
        shutil.copy(sample, val_dir / sample.name)
    
    for sample in test_samples:
        shutil.copy(sample, test_dir / sample.name)
    
    logger.info(f"Split complete:")
    logger.info(f"  Train: {len(train_samples)} samples")
    logger.info(f"  Val: {len(val_samples)} samples")
    logger.info(f"  Test: {len(test_samples)} samples")


def validate_annotations(
    annotation_path: Path,
    label_schema: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate annotation file.
    
    Args:
        annotation_path: Path to annotation file
        label_schema: List of valid labels
        
    Returns:
        Tuple of (is_valid, list of errors)
    """
    errors = []
    
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return False, ["Invalid JSON format"]
    
    # Check required fields
    required_fields = ['image_path', 'words']
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    # Validate words
    if 'words' in data:
        for i, word in enumerate(data['words']):
            # Check word structure
            if 'text' not in word:
                errors.append(f"Word {i} missing 'text' field")
            
            if 'bbox' not in word:
                errors.append(f"Word {i} missing 'bbox' field")
            elif len(word['bbox']) != 4:
                errors.append(f"Word {i} bbox must have 4 coordinates")
            
            # Validate label if present
            if 'label' in word:
                label = word['label']
                if label not in label_schema:
                    errors.append(f"Word {i} has invalid label: {label}")
            
            # Validate bbox coordinates
            if 'bbox' in word:
                bbox = word['bbox']
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    errors.append(f"Word {i} has invalid bbox coordinates")
    
    return len(errors) == 0, errors


def merge_bio_labels(predictions: List[Dict]) -> List[Dict]:
    """
    Merge B- and I- labels into complete entities.
    
    Args:
        predictions: List of predictions with BIO labels
        
    Returns:
        List of merged entities
    """
    entities = []
    current_entity = None
    
    for pred in predictions:
        label = pred['label']
        
        if label.startswith('B-'):
            # Start new entity
            if current_entity:
                entities.append(current_entity)
            
            entity_type = label[2:]
            current_entity = {
                'type': entity_type,
                'text': pred['text'],
                'bbox': pred['bbox'],
                'confidence': pred.get('confidence', 1.0)
            }
        
        elif label.startswith('I-') and current_entity:
            # Continue current entity
            entity_type = label[2:]
            if entity_type == current_entity['type']:
                current_entity['text'] += ' ' + pred['text']
                # Expand bbox
                bbox = current_entity['bbox']
                pred_bbox = pred['bbox']
                current_entity['bbox'] = [
                    min(bbox[0], pred_bbox[0]),
                    min(bbox[1], pred_bbox[1]),
                    max(bbox[2], pred_bbox[2]),
                    max(bbox[3], pred_bbox[3])
                ]
        else:
            # O label - end current entity
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    # Add last entity
    if current_entity:
        entities.append(current_entity)
    
    return entities


def calculate_dataset_statistics(data_dir: Path) -> Dict:
    """
    Calculate statistics for a dataset.
    
    Args:
        data_dir: Directory containing annotation files
        
    Returns:
        Dictionary of statistics
    """
    from collections import Counter
    
    samples = list(data_dir.glob("**/*.json"))
    
    stats = {
        'num_samples': len(samples),
        'num_words': 0,
        'num_entities': 0,
        'entity_distribution': Counter(),
        'avg_words_per_sample': 0,
        'avg_entities_per_sample': 0
    }
    
    for sample_path in samples:
        with open(sample_path, 'r') as f:
            data = json.load(f)
        
        words = data.get('words', [])
        stats['num_words'] += len(words)
        
        for word in words:
            label = word.get('label', 'O')
            if label != 'O':
                entity_type = label.split('-')[1] if '-' in label else label
                if label.startswith('B-'):
                    stats['num_entities'] += 1
                stats['entity_distribution'][entity_type] += 1
    
    if stats['num_samples'] > 0:
        stats['avg_words_per_sample'] = stats['num_words'] / stats['num_samples']
        stats['avg_entities_per_sample'] = stats['num_entities'] / stats['num_samples']
    
    return stats


def print_dataset_statistics(stats: Dict) -> None:
    """Print dataset statistics in readable format."""
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    print(f"Number of samples: {stats['num_samples']}")
    print(f"Total words: {stats['num_words']}")
    print(f"Total entities: {stats['num_entities']}")
    print(f"Average words per sample: {stats['avg_words_per_sample']:.2f}")
    print(f"Average entities per sample: {stats['avg_entities_per_sample']:.2f}")
    print("\nEntity Distribution:")
    for entity_type, count in stats['entity_distribution'].most_common():
        print(f"  {entity_type}: {count}")
    print("="*80 + "\n")
