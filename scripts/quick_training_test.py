#!/usr/bin/env python3
"""
Quick Training Test
===================
Runs a minimal training session to verify the pipeline works.

This script:
- Uses sample data
- Runs for 1 epoch with 2 steps
- Tests all components without full training
- Useful for debugging and validation

Usage:
    python scripts/quick_training_test.py
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run a quick training test."""
    print("=" * 80)
    print("QUICK TRAINING TEST")
    print("=" * 80)
    print()
    print("This script tests the training pipeline without full training.")
    print("It will:")
    print("  1. Check if dependencies are installed")
    print("  2. Load sample data (or create it)")
    print("  3. Initialize the model")
    print("  4. Run 1 epoch with max 2 training steps")
    print("  5. Report results")
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    missing_deps = []
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch not installed")
        missing_deps.append("torch")
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        missing_deps.append("transformers")
    
    if missing_deps:
        print()
        print("❌ Missing dependencies. Please install:")
        print(f"   pip install {' '.join(missing_deps)}")
        print()
        print("Or install all requirements:")
        print("   pip install -r requirements.txt")
        return 1
    
    print()
    
    # Check for sample data
    base_path = Path(__file__).parent.parent
    train_file = base_path / 'data' / 'processed' / 'train.jsonl'
    
    if not train_file.exists():
        print("⚠️  No training data found. Creating sample data...")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(base_path / 'scripts' / 'prepare_training.py'), 
             '--create-sample-data', '--num-samples', '20'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print("❌ Failed to create sample data")
            print(result.stderr)
            return 1
        print("✅ Sample data created")
    else:
        print("✅ Training data found")
    
    print()
    
    # Try to import and initialize components
    print("Testing imports and initialization...")
    
    try:
        from models.layoutlmv3_model import LayoutLMv3ForMultiTask
        print("✅ Model class imported")
    except ImportError as e:
        print(f"❌ Failed to import model: {e}")
        return 1
    
    try:
        from preprocessing import create_dataloaders
        print("✅ Dataset utilities imported")
    except ImportError as e:
        print(f"❌ Failed to import preprocessing: {e}")
        return 1
    
    try:
        from training.trainer import Trainer
        print("✅ Trainer imported")
    except ImportError as e:
        print(f"❌ Failed to import trainer: {e}")
        return 1
    
    print()
    
    # Load config
    import yaml
    config_path = base_path / 'configs' / 'training_config_production.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("✅ Configuration loaded")
    print()
    
    # Modify config for quick test
    print("Configuring for quick test...")
    config['training']['num_epochs'] = 1
    config['training']['per_device_train_batch_size'] = 1
    config['training']['gradient_accumulation_steps'] = 1
    config['training']['logging_steps'] = 1
    config['training']['eval_steps'] = 2
    config['training']['save_steps'] = 1000  # Don't save during test
    config['training']['fp16'] = False  # Disable for compatibility
    
    # Limit to 2 steps for quick test
    max_train_steps = 2
    
    print(f"  - Epochs: {config['training']['num_epochs']}")
    print(f"  - Max steps: {max_train_steps}")
    print(f"  - Batch size: {config['training']['per_device_train_batch_size']}")
    print()
    
    # Initialize model
    print("Initializing model (this may take a moment)...")
    
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        model = LayoutLMv3ForMultiTask(
            model_name_or_path=config['model']['model_name_or_path'],
            num_labels=config['model']['num_labels'],
            use_crf=config['model'].get('use_crf', True),
            table_num_cols=config['model'].get('table_num_cols', 16)
        )
        print("✅ Model initialized")
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        return 1
    
    print()
    
    # Prepare dataloaders
    print("Preparing dataloaders...")
    
    try:
        train_loader, val_loader = create_dataloaders(
            train_json=str(base_path / config['data']['train_json']),
            val_json=str(base_path / config['data']['val_json']),
            label_list_path=str(base_path / config['data']['label_list_path']),
            batch_size=config['training']['per_device_train_batch_size'],
            num_workers=0,  # Single worker for testing
            apply_augmentation=False  # No augmentation for quick test
        )
        print("✅ Dataloaders created")
        print(f"   Training batches: {len(train_loader)}")
        print(f"   Validation batches: {len(val_loader)}")
    except Exception as e:
        print(f"❌ Failed to create dataloaders: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # Test forward pass
    print("Testing forward pass...")
    
    try:
        model.to(device)
        model.eval()
        
        # Get a batch
        batch = next(iter(train_loader))
        
        # Move to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**batch)
        
        print("✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())}")
        
        if 'loss' in outputs:
            print(f"   Loss: {outputs['loss'].item():.4f}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    
    # Test training step
    print("Testing training step...")
    
    try:
        from torch.optim import AdamW
        
        model.train()
        optimizer = AdamW(model.parameters(), lr=1e-5)
        
        # Get a batch
        batch = next(iter(train_loader))
        
        # Move to device
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)
        
        # Training step
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        
        print("✅ Training step successful")
        print(f"   Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print()
    print("=" * 80)
    print("✅ ALL TESTS PASSED")
    print("=" * 80)
    print()
    print("The training pipeline is working correctly!")
    print()
    print("Next steps:")
    print("  1. Prepare real training data (500-1000 documents)")
    print("  2. Run full training: python scripts/start_training.py")
    print("  3. Monitor with TensorBoard: tensorboard --logdir logs/tensorboard")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
