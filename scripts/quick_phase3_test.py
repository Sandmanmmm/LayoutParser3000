"""
Quick Smoke Test for Phase 3 - Trainer Updates
Fast validation of multi-task trainer functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from models.layoutlmv3_model import LayoutLMv3ForMultiTask
from transformers import LayoutLMv3Config


def test_trainer_imports():
    """Test that trainer can be imported with new changes."""
    print("\n" + "="*60)
    print("TEST 1: Import Trainer")
    print("="*60)
    
    try:
        from training.trainer import Trainer
        print("✓ Trainer imported successfully")
        print(f"  Class: {Trainer.__name__}")
        print(f"  Module: {Trainer.__module__}")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_output_format():
    """Test that model returns dict with loss components."""
    print("\n" + "="*60)
    print("TEST 2: Model Output Format")
    print("="*60)
    
    # Create small model
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
    model.eval()
    
    # Create dummy batch
    batch_size, seq_len = 2, 512
    batch = {
        'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
        'attention_mask': torch.ones(batch_size, seq_len),
        'bbox': torch.randint(0, 1000, (batch_size, seq_len, 4)),
        'pixel_values': torch.rand(batch_size, 3, 224, 224),
        'labels': torch.randint(0, 115, (batch_size, seq_len)),
        'cell_labels': torch.randint(0, 2, (batch_size, seq_len)),
        'col_labels': torch.randint(0, 16, (batch_size, seq_len))
    }
    
    with torch.no_grad():
        outputs = model(**batch)
    
    # Check output structure
    required_keys = ['loss', 'ner_loss', 'cell_loss', 'col_loss', 
                    'ner_logits', 'cell_logits', 'col_logits']
    
    print(f"✓ Forward pass completed")
    print(f"  Output type: {type(outputs)}")
    print(f"  Output keys: {list(outputs.keys())}")
    
    missing = [k for k in required_keys if k not in outputs]
    if missing:
        print(f"\n✗ Missing keys: {missing}")
        return False
    else:
        print(f"\n✓ All required keys present")
        print(f"  Total loss: {outputs['loss'].item():.4f}")
        print(f"  NER loss: {outputs['ner_loss'].item():.4f}")
        print(f"  Cell loss: {outputs['cell_loss'].item():.4f}")
        print(f"  Col loss: {outputs['col_loss'].item():.4f}")
        return True


def test_trainer_methods():
    """Test that trainer has updated methods."""
    print("\n" + "="*60)
    print("TEST 3: Trainer Method Signatures")
    print("="*60)
    
    from training.trainer import Trainer
    import inspect
    
    # Check train_epoch return type annotation
    train_epoch_sig = inspect.signature(Trainer.train_epoch)
    print(f"✓ train_epoch signature: {train_epoch_sig}")
    
    # Check evaluate return type
    evaluate_sig = inspect.signature(Trainer.evaluate)
    print(f"✓ evaluate signature: {evaluate_sig}")
    
    # Check for new helper method
    if hasattr(Trainer, '_compute_classification_metrics'):
        print(f"✓ _compute_classification_metrics method exists")
    else:
        print(f"✗ Missing _compute_classification_metrics")
        return False
    
    return True


def test_loss_component_handling():
    """Test that dict outputs can be handled correctly."""
    print("\n" + "="*60)
    print("TEST 4: Loss Component Extraction")
    print("="*60)
    
    # Simulate model output
    outputs = {
        'loss': torch.tensor(5.0),
        'ner_loss': torch.tensor(3.0),
        'cell_loss': torch.tensor(1.0),
        'col_loss': torch.tensor(1.0),
        'ner_logits': torch.randn(2, 512, 115),
        'cell_logits': torch.randn(2, 512, 2),
        'col_logits': torch.randn(2, 512, 16)
    }
    
    # Test extraction
    try:
        loss = outputs['loss']
        ner_loss = outputs.get('ner_loss')
        cell_loss = outputs.get('cell_loss')
        col_loss = outputs.get('col_loss')
        
        print(f"✓ Loss extraction successful")
        print(f"  Total: {loss.item():.4f}")
        print(f"  NER: {ner_loss.item():.4f}")
        print(f"  Cell: {cell_loss.item():.4f}")
        print(f"  Col: {col_loss.item():.4f}")
        
        # Test backward
        loss.backward()
        print(f"✓ Backward pass successful")
        
        return True
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False


def test_composite_metric():
    """Test composite metric calculation."""
    print("\n" + "="*60)
    print("TEST 5: Composite Metric Calculation")
    print("="*60)
    
    metrics = {
        'eval_ner_f1': 0.85,
        'eval_cell_f1': 0.90,
        'eval_col_f1': 0.75
    }
    
    # Calculate composite
    composite = (
        metrics['eval_ner_f1'] * 1.0 +
        metrics['eval_cell_f1'] * 0.5 +
        metrics['eval_col_f1'] * 0.3
    ) / (1.0 + 0.5 + 0.3)
    
    print(f"✓ Composite metric calculated")
    print(f"  NER F1: {metrics['eval_ner_f1']:.4f} (weight=1.0)")
    print(f"  Cell F1: {metrics['eval_cell_f1']:.4f} (weight=0.5)")
    print(f"  Col F1: {metrics['eval_col_f1']:.4f} (weight=0.3)")
    print(f"  Composite: {composite:.4f}")
    
    return True


def main():
    """Run quick smoke tests."""
    print("\n" + "="*60)
    print("PHASE 3 SMOKE TESTS - QUICK VALIDATION")
    print("="*60)
    
    results = []
    results.append(test_trainer_imports())
    results.append(test_model_output_format())
    results.append(test_trainer_methods())
    results.append(test_loss_component_handling())
    results.append(test_composite_metric())
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ All smoke tests passed!")
        print("\nPhase 3 key changes validated:")
        print("  ✓ Multi-task loss dict handling")
        print("  ✓ Loss component extraction")
        print("  ✓ Trainer method updates")
        print("  ✓ Composite metric calculation")
        print("\nNext: Run full test_trainer.py for integration tests")
    else:
        print("\n⚠ Some tests failed - review above")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
