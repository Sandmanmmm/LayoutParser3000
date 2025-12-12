#!/usr/bin/env python3
"""
Training Preparation Script
===========================
Prepares the LayoutParser3000 system for model training by:
1. Validating the environment and dependencies
2. Checking configuration files
3. Preparing data directories
4. Creating sample dataset (if needed for testing)
5. Validating label schema consistency
6. Providing training readiness report

Usage:
    python scripts/prepare_training.py [--create-sample-data] [--validate-only]
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_environment() -> Tuple[bool, List[str]]:
    """Check if required packages are installed."""
    print("=" * 80)
    print("STEP 1: Environment Check")
    print("=" * 80)
    
    issues = []
    required_packages = [
        'torch',
        'transformers',
        'PIL',
        'yaml',
        'numpy',
        'albumentations',
        'seqeval'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package:20s} - installed")
        except ImportError:
            print(f"❌ {package:20s} - NOT FOUND")
            issues.append(f"Missing package: {package}")
    
    print()
    return len(issues) == 0, issues


def check_config_files(base_path: Path) -> Tuple[bool, List[str]]:
    """Check if all required configuration files exist."""
    print("=" * 80)
    print("STEP 2: Configuration Files Check")
    print("=" * 80)
    
    issues = []
    required_configs = [
        'configs/training_config_production.yaml',
        'configs/label_list_production.txt',
        'configs/output_mapping_production.yaml'
    ]
    
    for config_path in required_configs:
        full_path = base_path / config_path
        if full_path.exists():
            print(f"✅ {config_path}")
        else:
            print(f"❌ {config_path} - NOT FOUND")
            issues.append(f"Missing config: {config_path}")
    
    print()
    return len(issues) == 0, issues


def validate_label_schema(base_path: Path) -> Tuple[bool, List[str], int]:
    """Validate the label schema is correctly formatted."""
    print("=" * 80)
    print("STEP 3: Label Schema Validation")
    print("=" * 80)
    
    issues = []
    label_file = base_path / 'configs' / 'label_list_production.txt'
    
    if not label_file.exists():
        issues.append("Label file not found")
        return False, issues, 0
    
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    num_labels = len(labels)
    print(f"Total labels: {num_labels}")
    
    # Check for 'O' label
    if 'O' not in labels:
        issues.append("Missing 'O' (Outside) label")
        print("❌ Missing 'O' label")
    else:
        print("✅ 'O' label present")
    
    # Check B-/I- pairing
    b_labels = [l for l in labels if l.startswith('B-')]
    i_labels = [l for l in labels if l.startswith('I-')]
    
    print(f"B- labels: {len(b_labels)}")
    print(f"I- labels: {len(i_labels)}")
    
    if len(b_labels) != len(i_labels):
        issues.append(f"Mismatch: {len(b_labels)} B- labels vs {len(i_labels)} I- labels")
        print(f"❌ Label mismatch")
    else:
        print("✅ B-/I- labels match")
    
    # Verify pairing
    for b_label in b_labels:
        entity = b_label[2:]  # Remove 'B-'
        i_label = f'I-{entity}'
        if i_label not in i_labels:
            issues.append(f"Missing I- pair for {b_label}")
            print(f"❌ Missing {i_label}")
    
    if not issues:
        print("✅ All B-/I- pairs validated")
    
    print()
    return len(issues) == 0, issues, num_labels


def check_data_directories(base_path: Path) -> Tuple[bool, List[str], Dict]:
    """Check data directory structure and count files."""
    print("=" * 80)
    print("STEP 4: Data Directory Check")
    print("=" * 80)
    
    issues = []
    stats = {
        'train': 0,
        'val': 0,
        'test': 0
    }
    
    processed_dir = base_path / 'data' / 'processed'
    
    # Check if processed directory exists
    if not processed_dir.exists():
        print(f"⚠️  Data directory not found: {processed_dir}")
        print("   Creating directory structure...")
        processed_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {processed_dir}")
    
    # Check for data files
    for split in ['train', 'val', 'test']:
        jsonl_file = processed_dir / f'{split}.jsonl'
        json_file = processed_dir / f'{split}.json'
        
        if jsonl_file.exists():
            # Count lines in JSONL
            with open(jsonl_file, 'r') as f:
                stats[split] = sum(1 for _ in f)
            print(f"✅ {split}.jsonl - {stats[split]} samples")
        elif json_file.exists():
            # Check if it's a JSON array
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    stats[split] = len(data)
                else:
                    stats[split] = 1
            print(f"✅ {split}.json - {stats[split]} samples")
        else:
            print(f"⚠️  No data file found for {split} split")
            issues.append(f"Missing {split} data")
    
    total_samples = sum(stats.values())
    print(f"\nTotal samples: {total_samples}")
    
    if total_samples == 0:
        print("⚠️  WARNING: No training data found!")
        print("   You need to prepare training data before training.")
    
    print()
    return total_samples > 0, issues, stats


def create_sample_dataset(base_path: Path, num_samples: int = 10):
    """Create a sample dataset for testing the training pipeline."""
    print("=" * 80)
    print("STEP 5: Creating Sample Dataset")
    print("=" * 80)
    
    # Load the test invoice as template
    test_invoice_path = base_path / 'data' / 'test_dataset' / 'test_invoice.json'
    
    if not test_invoice_path.exists():
        print(f"❌ Template file not found: {test_invoice_path}")
        return False
    
    with open(test_invoice_path, 'r') as f:
        template = json.load(f)
    
    processed_dir = base_path / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create splits
    splits = {
        'train': int(num_samples * 0.7),
        'val': int(num_samples * 0.15),
        'test': int(num_samples * 0.15)
    }
    
    # Ensure at least 1 sample per split
    for split in splits:
        if splits[split] == 0:
            splits[split] = 1
    
    for split, count in splits.items():
        samples = []
        for i in range(count):
            # Create a modified copy
            sample = template.copy()
            sample['metadata']['file_name'] = f'{split}_sample_{i}.png'
            samples.append(sample)
        
        # Write as JSONL
        output_file = processed_dir / f'{split}.jsonl'
        with open(output_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"✅ Created {split}.jsonl with {count} samples")
    
    print("\n⚠️  NOTE: This is SAMPLE DATA for testing only!")
    print("   For actual training, you need real annotated documents.")
    print()
    return True


def check_model_files(base_path: Path) -> Tuple[bool, List[str]]:
    """Check if model architecture files exist."""
    print("=" * 80)
    print("STEP 6: Model Architecture Check")
    print("=" * 80)
    
    issues = []
    required_files = [
        'models/__init__.py',
        'models/layoutlmv3_model.py',
        'training/__init__.py',
        'training/trainer.py',
        'preprocessing/__init__.py',
        'preprocessing/dataset.py'
    ]
    
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            issues.append(f"Missing file: {file_path}")
    
    print()
    return len(issues) == 0, issues


def load_training_config(base_path: Path) -> Dict:
    """Load and validate training configuration."""
    config_path = base_path / 'configs' / 'training_config_production.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def print_training_summary(config: Dict, stats: Dict, num_labels: int):
    """Print a summary of the training configuration."""
    print("=" * 80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Dataset:")
    print(f"   Train samples: {stats.get('train', 0)}")
    print(f"   Val samples:   {stats.get('val', 0)}")
    print(f"   Test samples:  {stats.get('test', 0)}")
    
    print(f"\n🏷️  Labels:")
    print(f"   Total labels: {num_labels}")
    print(f"   Config expects: {config['model']['num_labels']}")
    
    if num_labels != config['model']['num_labels']:
        print(f"   ⚠️  WARNING: Label count mismatch!")
    
    training_cfg = config['training']
    print(f"\n⚙️  Training Settings:")
    print(f"   Epochs: {training_cfg['num_epochs']}")
    print(f"   Batch size: {training_cfg['per_device_train_batch_size']}")
    print(f"   Gradient accumulation: {training_cfg['gradient_accumulation_steps']}")
    effective_batch = training_cfg['per_device_train_batch_size'] * training_cfg['gradient_accumulation_steps']
    print(f"   Effective batch size: {effective_batch}")
    print(f"   Learning rate: {training_cfg['learning_rate']}")
    print(f"   FP16: {training_cfg['fp16']}")
    
    print(f"\n💾 Output:")
    print(f"   Checkpoint dir: {training_cfg['output_dir']}")
    print(f"   Best metric: {training_cfg['metric_for_best_model']}")
    
    print()


def print_next_steps(has_data: bool):
    """Print next steps for the user."""
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    
    if not has_data:
        print("\n🚨 ACTION REQUIRED: Prepare Training Data")
        print("\nOption 1: Create sample data for testing")
        print("   python scripts/prepare_training.py --create-sample-data")
        
        print("\nOption 2: Prepare real training data")
        print("   1. Collect invoice/PO documents")
        print("   2. Run OCR extraction")
        print("   3. Annotate with Label Studio or similar tool")
        print("   4. Export to JSON format")
        print("   5. Place in data/processed/ as train.jsonl, val.jsonl, test.jsonl")
        
        print("\nOption 3: Use existing dataset (e.g., CORD)")
        print("   1. Download CORD dataset")
        print("   2. Convert annotations to 49-label schema")
        print("   3. Place in data/processed/")
    else:
        print("\n✅ System is ready for training!")
        print("\nTo start training:")
        print("   python scripts/start_training.py")
        print("\nOr use the trainer directly:")
        print("   python training/trainer.py --config configs/training_config_production.yaml")
        
        print("\nTo monitor training:")
        print("   tensorboard --logdir logs/tensorboard")
    
    print()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Prepare system for model training')
    parser.add_argument('--create-sample-data', action='store_true',
                       help='Create sample dataset for testing')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate, do not create anything')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of sample data to create (default: 10)')
    
    args = parser.parse_args()
    
    base_path = Path(__file__).parent.parent
    
    print("\n" + "=" * 80)
    print("LAYOUTPARSER3000 - TRAINING PREPARATION")
    print("=" * 80)
    print()
    
    all_checks_passed = True
    
    # Step 1: Environment
    env_ok, env_issues = check_environment()
    all_checks_passed &= env_ok
    
    # Step 2: Config files
    config_ok, config_issues = check_config_files(base_path)
    all_checks_passed &= config_ok
    
    # Step 3: Label schema
    label_ok, label_issues, num_labels = validate_label_schema(base_path)
    all_checks_passed &= label_ok
    
    # Step 4: Data directories
    data_ok, data_issues, stats = check_data_directories(base_path)
    
    # Step 5: Create sample data if requested
    if args.create_sample_data and not args.validate_only:
        create_sample_dataset(base_path, args.num_samples)
        # Re-check data
        data_ok, data_issues, stats = check_data_directories(base_path)
    
    # Step 6: Model files
    model_ok, model_issues = check_model_files(base_path)
    all_checks_passed &= model_ok
    
    # Load configuration
    if config_ok:
        config = load_training_config(base_path)
        print_training_summary(config, stats, num_labels)
    
    # Print status
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    all_issues = env_issues + config_issues + label_issues + data_issues + model_issues
    
    if all_checks_passed and data_ok:
        print("✅ ALL CHECKS PASSED - System is ready for training!")
    elif all_checks_passed:
        print("⚠️  SYSTEM CONFIGURED - Waiting for training data")
    else:
        print("❌ ISSUES FOUND:")
        for issue in all_issues:
            print(f"   - {issue}")
    
    print()
    
    # Print next steps
    print_next_steps(data_ok)
    
    # Exit code
    sys.exit(0 if all_checks_passed else 1)


if __name__ == '__main__':
    main()
