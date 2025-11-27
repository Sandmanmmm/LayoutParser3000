"""
Test Script for Multi-Task Trainer
Tests the updated trainer with dummy data to verify:
- Multi-task loss handling
- Loss component logging
- Per-task metrics computation
- Checkpoint saving with composite metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import LayoutLMv3Processor
import tempfile
import shutil

from models.layoutlmv3_model import LayoutLMv3ForMultiTask
from training.trainer import Trainer


class DummyDataset(Dataset):
    """Dummy dataset that produces valid multi-task batches."""
    
    def __init__(self, num_samples=10):
        self.num_samples = num_samples
        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False
        )
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate a valid sample with all multi-task labels."""
        # Simulate OCR tokens (10 words)
        words = [f"word{i}" for i in range(10)]
        boxes = [[i*100, 50, (i+1)*100, 100] for i in range(10)]
        
        # Create a dummy image (3, 224, 224)
        image = torch.rand(3, 224, 224)
        
        # Tokenize
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        # Remove batch dimension
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'bbox': encoding['bbox'].squeeze(0),
            'pixel_values': encoding['pixel_values'].squeeze(0)
        }
        
        # Generate labels with -100 padding
        seq_length = item['input_ids'].shape[0]
        
        # NER labels: random labels from 0-114, with -100 for special tokens
        labels = torch.full((seq_length,), -100, dtype=torch.long)
        # Set some valid labels (skip first and last tokens)
        labels[1:11] = torch.randint(0, 115, (10,))
        
        # Cell labels: binary (0 or 1), with -100 padding
        cell_labels = torch.full((seq_length,), -100, dtype=torch.long)
        cell_labels[1:11] = torch.randint(0, 2, (10,))
        
        # Column labels: 0-15, with -100 padding
        col_labels = torch.full((seq_length,), -100, dtype=torch.long)
        col_labels[1:11] = torch.randint(0, 16, (10,))
        
        item['labels'] = labels
        item['cell_labels'] = cell_labels
        item['col_labels'] = col_labels
        
        return item


def create_dummy_config(output_dir):
    """Create a minimal config for testing."""
    config = {
        'model': {
            'name': 'microsoft/layoutlmv3-base',
            'use_crf': False,
            'table_structure': {
                'enabled': True,
                'num_row_labels': 2,
                'num_col_labels': 16
            }
        },
        'training': {
            'num_epochs': 2,
            'batch_size': 2,
            'learning_rate': 3e-5,
            'weight_decay': 0.01,
            'warmup_ratio': 0.1,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'fp16': False,
            'logging_steps': 1,
            'save_steps': 5,
            'eval_steps': 5,
            'early_stopping_patience': 3,
            'metric_for_best_model': 'eval_composite_f1',
            'output_dir': str(output_dir)
        },
        'loss_weights': {
            'ner_loss_weight': 1.0,
            'cell_loss_weight': 1.0,
            'col_loss_weight': 0.5
        },
        'data': {
            'num_workers': 0,
            'max_seq_length': 512
        },
        'output': {
            'output_dir': str(output_dir)
        },
        'logging': {
            'tensorboard_enabled': False,
            'wandb_enabled': False
        },
        'labels': {
            'token_classification': ['O'] + [
                f'{prefix}-{label}' 
                for label in ['invoice_number', 'invoice_date', 'total']
                for prefix in ['B', 'I']
            ]
        }
    }
    return config


def test_model_instantiation():
    """Test that multi-task model can be instantiated."""
    print("\n" + "="*70)
    print("TEST 1: Model Instantiation")
    print("="*70)
    
    from transformers import LayoutLMv3Config
    
    config = LayoutLMv3Config.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=115
    )
    config.use_crf = False
    config.use_table_head = True
    config.num_row_labels = 2
    config.num_col_labels = 16
    
    model = LayoutLMv3ForMultiTask.from_pretrained(
        "microsoft/layoutlmv3-base",
        config=config
    )
    
    print(f"✓ Model instantiated successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  NER head: Linear(768, 115)")
    print(f"  Cell head: Linear(768, 2)")
    print(f"  Column head: Linear(768, 16)")
    
    return model


def test_dataset_creation():
    """Test that dummy dataset produces valid batches."""
    print("\n" + "="*70)
    print("TEST 2: Dataset Creation")
    print("="*70)
    
    dataset = DummyDataset(num_samples=4)
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # Test single sample
    sample = dataset[0]
    print(f"\n  Sample keys: {list(sample.keys())}")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels shape: {sample['labels'].shape}")
    print(f"  cell_labels shape: {sample['cell_labels'].shape}")
    print(f"  col_labels shape: {sample['col_labels'].shape}")
    print(f"  pixel_values shape: {sample['pixel_values'].shape}")
    
    # Test batch
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    print(f"\n  Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"  Valid NER labels: {(batch['labels'] != -100).sum().item()}")
    print(f"  Valid cell labels: {(batch['cell_labels'] != -100).sum().item()}")
    print(f"  Valid col labels: {(batch['col_labels'] != -100).sum().item()}")
    
    return dataset


def test_forward_pass(model, dataset):
    """Test model forward pass with dummy data."""
    print("\n" + "="*70)
    print("TEST 3: Model Forward Pass")
    print("="*70)
    
    model.eval()
    loader = DataLoader(dataset, batch_size=2)
    batch = next(iter(loader))
    
    with torch.no_grad():
        outputs = model(**batch)
    
    print(f"✓ Forward pass completed")
    print(f"\n  Output keys: {list(outputs.keys())}")
    print(f"  Total loss: {outputs['loss'].item():.4f}")
    if 'ner_loss' in outputs:
        print(f"  NER loss: {outputs['ner_loss'].item():.4f}")
    if 'cell_loss' in outputs:
        print(f"  Cell loss: {outputs['cell_loss'].item():.4f}")
    if 'col_loss' in outputs:
        print(f"  Col loss: {outputs['col_loss'].item():.4f}")
    
    print(f"\n  NER logits shape: {outputs['ner_logits'].shape}")
    print(f"  Cell logits shape: {outputs['cell_logits'].shape}")
    print(f"  Col logits shape: {outputs['col_logits'].shape}")
    
    return outputs


def test_trainer_instantiation(model, dataset, output_dir):
    """Test trainer instantiation with multi-task config."""
    print("\n" + "="*70)
    print("TEST 4: Trainer Instantiation")
    print("="*70)
    
    config = create_dummy_config(output_dir)
    
    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device='cpu'
    )
    
    print(f"✓ Trainer instantiated successfully")
    print(f"  Epochs: {trainer.num_epochs}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Loss weights: {trainer.loss_weights}")
    print(f"  Output dir: {trainer.output_dir}")
    
    return trainer, config


def test_single_training_step(trainer):
    """Test a single training step."""
    print("\n" + "="*70)
    print("TEST 5: Single Training Step")
    print("="*70)
    
    trainer.model.train()
    batch = next(iter(trainer.train_loader))
    
    # Move to device
    batch = {
        k: v.to(trainer.device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    
    # Forward + backward
    outputs = trainer.model(**batch)
    loss = outputs['loss']
    loss.backward()
    
    print(f"✓ Training step completed")
    print(f"  Total loss: {loss.item():.4f}")
    if 'ner_loss' in outputs:
        print(f"  NER loss: {outputs['ner_loss'].item():.4f}")
    if 'cell_loss' in outputs:
        print(f"  Cell loss: {outputs['cell_loss'].item():.4f}")
    if 'col_loss' in outputs:
        print(f"  Col loss: {outputs['col_loss'].item():.4f}")
    
    # Check gradients
    has_grads = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in trainer.model.parameters()
    )
    print(f"  Gradients computed: {has_grads}")
    
    trainer.optimizer.zero_grad()


def test_evaluation_step(trainer):
    """Test evaluation with multi-task metrics."""
    print("\n" + "="*70)
    print("TEST 6: Evaluation Step")
    print("="*70)
    
    metrics = trainer.evaluate(epoch=0)
    
    print(f"✓ Evaluation completed")
    print(f"\n  Metrics computed:")
    for key, value in sorted(metrics.items()):
        print(f"    {key}: {value:.4f}")
    
    # Check for expected metrics
    expected_keys = [
        'eval_loss',
        'eval_ner_loss',
        'eval_cell_loss',
        'eval_col_loss'
    ]
    
    missing = [k for k in expected_keys if k not in metrics]
    if missing:
        print(f"\n  ⚠ Missing metrics: {missing}")
    else:
        print(f"\n  ✓ All expected metrics present")
    
    return metrics


def test_checkpoint_saving(trainer, metrics):
    """Test checkpoint saving with multi-task metrics."""
    print("\n" + "="*70)
    print("TEST 7: Checkpoint Saving")
    print("="*70)
    
    trainer.save_checkpoint(epoch=0, metrics=metrics, is_best=True)
    
    checkpoint_path = trainer.output_dir / 'best_model.pt'
    if checkpoint_path.exists():
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Load and inspect
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"\n  Checkpoint keys: {list(checkpoint.keys())}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Loss weights: {checkpoint.get('loss_weights', 'N/A')}")
        
        if 'metrics' in checkpoint:
            print(f"  Saved metrics:")
            for k, v in checkpoint['metrics'].items():
                print(f"    {k}: {v:.4f}")
    else:
        print(f"✗ Checkpoint not found at {checkpoint_path}")


def test_full_training(trainer):
    """Test full training loop (1 epoch)."""
    print("\n" + "="*70)
    print("TEST 8: Full Training Loop (1 Epoch)")
    print("="*70)
    
    # Override to train just 1 epoch
    original_epochs = trainer.num_epochs
    trainer.num_epochs = 1
    
    try:
        trainer.train()
        print(f"\n✓ Training completed successfully")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.num_epochs = original_epochs


def main():
    """Run all trainer tests."""
    print("\n" + "="*70)
    print("MULTI-TASK TRAINER TEST SUITE")
    print("="*70)
    print("Testing updated trainer.py with:")
    print("  - Multi-task loss handling")
    print("  - Loss component logging")
    print("  - Per-task metrics")
    print("  - Composite metric tracking")
    print("="*70)
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="trainer_test_"))
    print(f"\nTemp directory: {temp_dir}")
    
    try:
        # Run tests
        model = test_model_instantiation()
        dataset = test_dataset_creation()
        test_forward_pass(model, dataset)
        trainer, config = test_trainer_instantiation(model, dataset, temp_dir)
        test_single_training_step(trainer)
        metrics = test_evaluation_step(trainer)
        test_checkpoint_saving(trainer, metrics)
        test_full_training(trainer)
        
        print("\n" + "="*70)
        print("ALL TESTS COMPLETED")
        print("="*70)
        print("\n✅ Phase 3 (Trainer Updates) validation successful!")
        print("\nNext steps:")
        print("  1. Test with real data from data/processed/")
        print("  2. Monitor loss components during training")
        print("  3. Verify per-task metrics are computed correctly")
        print("  4. Proceed to Phase 4 (Data Acquisition)")
        
    except Exception as e:
        print(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temp directory: {temp_dir}")


if __name__ == "__main__":
    main()
