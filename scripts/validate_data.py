#!/usr/bin/env python3
"""
Data Validation Script
======================
Validates training data format and consistency.

Checks:
- JSON format and structure
- Required fields present
- Label schema consistency
- Bounding box validity
- Table structure correctness
- Token-label alignment

Usage:
    python scripts/validate_data.py [--data-dir PATH] [--split train|val|test]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_labels(label_file: Path) -> List[str]:
    """Load label list from file."""
    with open(label_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_data(data_file: Path) -> List[Dict]:
    """Load data from JSON or JSONL file."""
    samples = []
    
    if data_file.suffix == '.jsonl':
        with open(data_file, 'r') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    else:
        with open(data_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            else:
                samples = [data]
    
    return samples


def validate_sample_structure(sample: Dict, sample_idx: int) -> List[str]:
    """Validate the structure of a single sample."""
    issues = []
    
    # Check required top-level fields
    required_fields = ['metadata', 'ocr', 'ner_tags']
    for field in required_fields:
        if field not in sample:
            issues.append(f"Sample {sample_idx}: Missing required field '{field}'")
    
    # Check metadata
    if 'metadata' in sample:
        required_meta = ['file_name', 'document_type', 'pages']
        for field in required_meta:
            if field not in sample['metadata']:
                issues.append(f"Sample {sample_idx}: Missing metadata field '{field}'")
    
    # Check OCR structure
    if 'ocr' in sample:
        if not isinstance(sample['ocr'], list):
            issues.append(f"Sample {sample_idx}: 'ocr' must be a list")
        else:
            for i, token in enumerate(sample['ocr']):
                required_token_fields = ['text', 'bbox', 'page']
                for field in required_token_fields:
                    if field not in token:
                        issues.append(f"Sample {sample_idx}, token {i}: Missing '{field}'")
                
                # Validate bbox
                if 'bbox' in token:
                    bbox = token['bbox']
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        issues.append(f"Sample {sample_idx}, token {i}: Invalid bbox format (must be [x0,y0,x1,y1])")
                    elif bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                        issues.append(f"Sample {sample_idx}, token {i}: Invalid bbox coordinates (x0<x1, y0<y1)")
    
    # Check NER tags
    if 'ner_tags' in sample and 'ocr' in sample:
        if not isinstance(sample['ner_tags'], list):
            issues.append(f"Sample {sample_idx}: 'ner_tags' must be a list")
        elif len(sample['ner_tags']) != len(sample['ocr']):
            issues.append(f"Sample {sample_idx}: ner_tags length ({len(sample['ner_tags'])}) != ocr length ({len(sample['ocr'])})")
    
    # Check tables (optional but if present, validate structure)
    if 'tables' in sample and sample['tables']:
        for table_idx, table in enumerate(sample['tables']):
            required_table_fields = ['table_id', 'bbox', 'cells']
            for field in required_table_fields:
                if field not in table:
                    issues.append(f"Sample {sample_idx}, table {table_idx}: Missing '{field}'")
            
            if 'cells' in table:
                for cell_idx, cell in enumerate(table['cells']):
                    required_cell_fields = ['row', 'col', 'bbox', 'token_ids']
                    for field in required_cell_fields:
                        if field not in cell:
                            issues.append(f"Sample {sample_idx}, table {table_idx}, cell {cell_idx}: Missing '{field}'")
    
    return issues


def validate_labels(sample: Dict, sample_idx: int, valid_labels: List[str]) -> List[str]:
    """Validate label consistency."""
    issues = []
    
    if 'ner_tags' not in sample:
        return issues
    
    for i, label in enumerate(sample['ner_tags']):
        if label not in valid_labels:
            issues.append(f"Sample {sample_idx}, token {i}: Invalid label '{label}'")
    
    return issues


def validate_bio_consistency(sample: Dict, sample_idx: int) -> List[str]:
    """Validate BIO tag consistency."""
    issues = []
    
    if 'ner_tags' not in sample:
        return issues
    
    tags = sample['ner_tags']
    prev_tag = 'O'
    
    for i, tag in enumerate(tags):
        if tag.startswith('I-'):
            entity = tag[2:]
            # I- tag must follow B- or I- of same entity
            if prev_tag == 'O':
                issues.append(f"Sample {sample_idx}, token {i}: I-{entity} tag without preceding B-{entity}")
            elif prev_tag.startswith('B-') or prev_tag.startswith('I-'):
                prev_entity = prev_tag[2:]
                if prev_entity != entity:
                    issues.append(f"Sample {sample_idx}, token {i}: I-{entity} follows {prev_tag}")
        
        prev_tag = tag
    
    return issues


def compute_statistics(samples: List[Dict], valid_labels: List[str]) -> Dict:
    """Compute dataset statistics."""
    stats = {
        'num_samples': len(samples),
        'num_tokens': 0,
        'num_entities': 0,
        'label_distribution': Counter(),
        'entity_distribution': Counter(),
        'avg_tokens_per_sample': 0,
        'samples_with_tables': 0,
        'total_tables': 0,
        'total_cells': 0
    }
    
    for sample in samples:
        if 'ocr' in sample:
            stats['num_tokens'] += len(sample['ocr'])
        
        if 'ner_tags' in sample:
            for tag in sample['ner_tags']:
                stats['label_distribution'][tag] += 1
                if tag.startswith('B-'):
                    entity = tag[2:]
                    stats['entity_distribution'][entity] += 1
                    stats['num_entities'] += 1
        
        if 'tables' in sample and sample['tables']:
            stats['samples_with_tables'] += 1
            stats['total_tables'] += len(sample['tables'])
            for table in sample['tables']:
                if 'cells' in table:
                    stats['total_cells'] += len(table['cells'])
    
    if stats['num_samples'] > 0:
        stats['avg_tokens_per_sample'] = stats['num_tokens'] / stats['num_samples']
    
    return stats


def print_statistics(stats: Dict, valid_labels: List[str]):
    """Print dataset statistics."""
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    print(f"\n📊 Overall:")
    print(f"   Total samples: {stats['num_samples']}")
    print(f"   Total tokens: {stats['num_tokens']:,}")
    print(f"   Total entities: {stats['num_entities']:,}")
    print(f"   Avg tokens/sample: {stats['avg_tokens_per_sample']:.1f}")
    
    print(f"\n🏷️  Label Distribution:")
    for label in valid_labels[:10]:  # Show first 10 labels
        count = stats['label_distribution'].get(label, 0)
        percentage = (count / stats['num_tokens'] * 100) if stats['num_tokens'] > 0 else 0
        print(f"   {label:30s}: {count:6,} ({percentage:5.2f}%)")
    
    if len(valid_labels) > 10:
        print(f"   ... and {len(valid_labels) - 10} more labels")
    
    print(f"\n📋 Entity Distribution (Top 10):")
    for entity, count in stats['entity_distribution'].most_common(10):
        print(f"   {entity:30s}: {count:6,} occurrences")
    
    print(f"\n📑 Tables:")
    print(f"   Samples with tables: {stats['samples_with_tables']}")
    print(f"   Total tables: {stats['total_tables']}")
    print(f"   Total cells: {stats['total_cells']}")
    
    # Check for low-frequency entities
    print(f"\n⚠️  Low-frequency entities (<10 examples):")
    low_freq = [(e, c) for e, c in stats['entity_distribution'].items() if c < 10]
    if low_freq:
        for entity, count in sorted(low_freq, key=lambda x: x[1]):
            print(f"   {entity:30s}: {count} (⚠️  insufficient)")
    else:
        print("   None - all entities have sufficient examples ✅")
    
    print()


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description='Validate training data')
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing data files')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'],
                       default='all', help='Which split to validate')
    parser.add_argument('--label-file', type=str,
                       default='configs/label_list_production.txt',
                       help='Path to label list file')
    parser.add_argument('--max-issues', type=int, default=50,
                       help='Maximum issues to report per check')
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    data_dir = base_path / args.data_dir
    label_file = base_path / args.label_file
    
    print("\n" + "=" * 80)
    print("DATA VALIDATION")
    print("=" * 80)
    
    # Load labels
    if not label_file.exists():
        print(f"❌ Label file not found: {label_file}")
        sys.exit(1)
    
    valid_labels = load_labels(label_file)
    print(f"\n✅ Loaded {len(valid_labels)} labels from {label_file.name}")
    
    # Determine which splits to validate
    splits = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    
    all_issues = []
    all_stats = {}
    
    for split in splits:
        print(f"\n{'=' * 80}")
        print(f"VALIDATING {split.upper()} SPLIT")
        print(f"{'=' * 80}")
        
        # Try both .jsonl and .json extensions
        data_file = data_dir / f'{split}.jsonl'
        if not data_file.exists():
            data_file = data_dir / f'{split}.json'
        
        if not data_file.exists():
            print(f"⚠️  Data file not found: {data_file}")
            continue
        
        # Load data
        try:
            samples = load_data(data_file)
            print(f"✅ Loaded {len(samples)} samples from {data_file.name}")
        except Exception as e:
            print(f"❌ Failed to load data: {e}")
            continue
        
        # Validate each sample
        print("\n🔍 Validating sample structure...")
        structure_issues = []
        for i, sample in enumerate(samples):
            structure_issues.extend(validate_sample_structure(sample, i))
        
        if structure_issues:
            print(f"❌ Found {len(structure_issues)} structure issues:")
            for issue in structure_issues[:args.max_issues]:
                print(f"   {issue}")
            if len(structure_issues) > args.max_issues:
                print(f"   ... and {len(structure_issues) - args.max_issues} more")
        else:
            print("✅ All samples have valid structure")
        
        # Validate labels
        print("\n🔍 Validating label consistency...")
        label_issues = []
        for i, sample in enumerate(samples):
            label_issues.extend(validate_labels(sample, i, valid_labels))
        
        if label_issues:
            print(f"❌ Found {len(label_issues)} label issues:")
            for issue in label_issues[:args.max_issues]:
                print(f"   {issue}")
            if len(label_issues) > args.max_issues:
                print(f"   ... and {len(label_issues) - args.max_issues} more")
        else:
            print("✅ All labels are valid")
        
        # Validate BIO consistency
        print("\n🔍 Validating BIO tag consistency...")
        bio_issues = []
        for i, sample in enumerate(samples):
            bio_issues.extend(validate_bio_consistency(sample, i))
        
        if bio_issues:
            print(f"❌ Found {len(bio_issues)} BIO consistency issues:")
            for issue in bio_issues[:args.max_issues]:
                print(f"   {issue}")
            if len(bio_issues) > args.max_issues:
                print(f"   ... and {len(bio_issues) - args.max_issues} more")
        else:
            print("✅ BIO tags are consistent")
        
        # Compute statistics
        stats = compute_statistics(samples, valid_labels)
        all_stats[split] = stats
        print_statistics(stats, valid_labels)
        
        all_issues.extend(structure_issues + label_issues + bio_issues)
    
    # Final summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if all_issues:
        print(f"\n❌ Found {len(all_issues)} total issues")
        print("   Please fix these issues before training")
    else:
        print("\n✅ All validations passed!")
        print("   Data is ready for training")
    
    # Print combined statistics
    if len(all_stats) > 1:
        print(f"\n📊 Combined Statistics:")
        total_samples = sum(s['num_samples'] for s in all_stats.values())
        total_tokens = sum(s['num_tokens'] for s in all_stats.values())
        total_entities = sum(s['num_entities'] for s in all_stats.values())
        print(f"   Total samples: {total_samples:,}")
        print(f"   Total tokens: {total_tokens:,}")
        print(f"   Total entities: {total_entities:,}")
    
    print()
    
    sys.exit(0 if not all_issues else 1)


if __name__ == '__main__':
    main()
