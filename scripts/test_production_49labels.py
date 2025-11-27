"""
Production 49-Label Compatibility Test
Validates that the system works with the production 49-label schema.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from transformers import LayoutLMv3Config
from models.layoutlmv3_model import LayoutLMv3ForMultiTask


def load_production_labels():
    """Load the 49-label production schema."""
    label_file = Path("configs/label_list_production.txt")
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def test_label_count():
    """Verify production label list has exactly 49 labels."""
    print("\n" + "="*60)
    print("TEST 1: Production Label Count")
    print("="*60)
    
    labels = load_production_labels()
    print(f"Labels loaded: {len(labels)}")
    
    if len(labels) == 49:
        print(f"✓ Correct count: 49 labels")
        print(f"  First 5: {labels[:5]}")
        print(f"  Last 5: {labels[-5:]}")
        return True
    else:
        print(f"✗ Expected 49, got {len(labels)}")
        return False


def test_label_schema():
    """Verify label schema structure (O + B-/I- pairs)."""
    print("\n" + "="*60)
    print("TEST 2: Label Schema Structure")
    print("="*60)
    
    labels = load_production_labels()
    
    # Check first label is O
    if labels[0] != 'O':
        print(f"✗ First label should be 'O', got '{labels[0]}'")
        return False
    print(f"✓ First label is 'O'")
    
    # Check B-/I- pairing
    entity_labels = labels[1:]
    if len(entity_labels) % 2 != 0:
        print(f"✗ Entity labels should be even (B-/I- pairs), got {len(entity_labels)}")
        return False
    
    # Check pairs
    num_entities = len(entity_labels) // 2
    print(f"✓ Entity label count is even: {len(entity_labels)}")
    print(f"  Entity types: {num_entities}")
    
    # Verify B-/I- pattern
    for i in range(0, len(entity_labels), 2):
        b_label = entity_labels[i]
        i_label = entity_labels[i+1]
        
        if not b_label.startswith('B-') or not i_label.startswith('I-'):
            print(f"✗ Invalid pair at position {i}: {b_label}, {i_label}")
            return False
        
        entity_name_b = b_label[2:]
        entity_name_i = i_label[2:]
        
        if entity_name_b != entity_name_i:
            print(f"✗ Mismatched entity names: {b_label} vs {i_label}")
            return False
    
    print(f"✓ All B-/I- pairs are valid")
    return True


def test_entity_coverage():
    """Verify all required production entities are present."""
    print("\n" + "="*60)
    print("TEST 3: Entity Coverage")
    print("="*60)
    
    labels = load_production_labels()
    
    # Required document-level entities
    doc_entities = [
        'SUPPLIER_NAME', 'SUPPLIER_ADDRESS', 'BUYER_NAME', 'BUYER_ADDRESS',
        'DOCUMENT_NUMBER', 'DOCUMENT_DATE', 'DUE_DATE', 'CURRENCY',
        'SUBTOTAL', 'TAX', 'SHIPPING', 'TOTAL_AMOUNT',
        'PAYMENT_TERMS', 'ORDER_REFERENCE', 'VENDOR_TAX_ID', 'ACCOUNT_NUMBER'
    ]
    
    # Required line-item entities
    item_entities = [
        'ITEM_DESCRIPTION', 'SKU', 'QUANTITY', 'UOM',
        'UNIT_PRICE', 'LINE_TOTAL', 'PACK_SIZE', 'UNIT_COST'
    ]
    
    all_required = doc_entities + item_entities
    
    missing = []
    for entity in all_required:
        b_label = f'B-{entity}'
        i_label = f'I-{entity}'
        if b_label not in labels or i_label not in labels:
            missing.append(entity)
    
    if missing:
        print(f"✗ Missing entities: {missing}")
        return False
    
    print(f"✓ All 24 required entities present")
    print(f"  Document-level: {len(doc_entities)}")
    print(f"  Line-item: {len(item_entities)}")
    return True


def test_model_instantiation_49():
    """Test model instantiation with 49 labels."""
    print("\n" + "="*60)
    print("TEST 4: Model Instantiation (49 labels)")
    print("="*60)
    
    labels = load_production_labels()
    
    config = LayoutLMv3Config.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=49
    )
    config.use_crf = True
    config.use_table_head = True
    config.num_row_labels = 2
    config.num_col_labels = 16
    
    try:
        model = LayoutLMv3ForMultiTask.from_pretrained(
            "microsoft/layoutlmv3-base",
            config=config
        )
        
        print(f"✓ Model instantiated successfully")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  NER head: Linear(768, 49)")
        print(f"  Cell head: Linear(768, 2)")
        print(f"  Column head: Linear(768, 16)")
        
        # Check classifier shapes
        ner_out_features = model.ner_classifier.out_features
        if ner_out_features != 49:
            print(f"✗ NER classifier has {ner_out_features} outputs, expected 49")
            return False
        
        print(f"  ✓ NER classifier output size: {ner_out_features}")
        return True
        
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return False


def test_forward_pass_49():
    """Test forward pass with 49-label data."""
    print("\n" + "="*60)
    print("TEST 5: Forward Pass (49 labels)")
    print("="*60)
    
    labels = load_production_labels()
    
    config = LayoutLMv3Config.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=49
    )
    config.use_crf = False  # Faster for testing
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
        'labels': torch.randint(0, 49, (batch_size, seq_len)),  # 49 labels
        'cell_labels': torch.randint(0, 2, (batch_size, seq_len)),
        'col_labels': torch.randint(0, 16, (batch_size, seq_len))
    }
    
    try:
        with torch.no_grad():
            outputs = model(**batch)
        
        print(f"✓ Forward pass completed")
        print(f"  Total loss: {outputs['loss'].item():.4f}")
        print(f"  NER loss: {outputs['ner_loss'].item():.4f}")
        print(f"  Cell loss: {outputs['cell_loss'].item():.4f}")
        print(f"  Col loss: {outputs['col_loss'].item():.4f}")
        
        # Check logits shape
        ner_logits = outputs['ner_logits']
        if ner_logits.shape[-1] != 49:
            print(f"✗ NER logits have shape {ner_logits.shape}, expected (..., 49)")
            return False
        
        print(f"  ✓ NER logits shape: {ner_logits.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_production_config():
    """Test loading production configuration file."""
    print("\n" + "="*60)
    print("TEST 6: Production Config Loading")
    print("="*60)
    
    config_file = Path("configs/training_config_production.yaml")
    
    if not config_file.exists():
        print(f"✗ Config file not found: {config_file}")
        return False
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Config file loaded")
        
        # Check num_labels
        model_config = config.get('model', {})
        num_labels = model_config.get('num_labels')
        
        if num_labels != 49:
            print(f"✗ num_labels is {num_labels}, expected 49")
            return False
        
        print(f"  ✓ num_labels: {num_labels}")
        
        # Check other key settings
        training_config = config.get('training', {})
        print(f"  num_epochs: {training_config.get('num_epochs')}")
        print(f"  batch_size: {training_config.get('per_device_train_batch_size')}")
        print(f"  learning_rate: {training_config.get('learning_rate')}")
        
        # Check loss weights
        loss_weights = config.get('loss_weights', {})
        print(f"  loss_weights: {loss_weights}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_output_mapping():
    """Test output mapping configuration."""
    print("\n" + "="*60)
    print("TEST 7: Output Mapping Config")
    print("="*60)
    
    mapping_file = Path("configs/output_mapping_production.yaml")
    
    if not mapping_file.exists():
        print(f"✗ Mapping file not found: {mapping_file}")
        return False
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = yaml.safe_load(f)
        
        print(f"✓ Mapping file loaded")
        
        # Check purchase_order_mapping
        po_mapping = mapping.get('purchase_order_mapping', {})
        print(f"  Purchase order fields: {len(po_mapping)}")
        
        # Check line_item_mapping
        li_mapping = mapping.get('line_item_mapping', {})
        print(f"  Line item fields: {len(li_mapping)}")
        
        # Verify key fields
        required_po = ['DOCUMENT_NUMBER', 'SUPPLIER_NAME', 'TOTAL_AMOUNT']
        required_li = ['ITEM_DESCRIPTION', 'QUANTITY', 'UNIT_PRICE']
        
        missing_po = [f for f in required_po if f not in po_mapping]
        missing_li = [f for f in required_li if f not in li_mapping]
        
        if missing_po or missing_li:
            print(f"✗ Missing mappings: PO={missing_po}, LI={missing_li}")
            return False
        
        print(f"  ✓ All required mappings present")
        return True
        
    except Exception as e:
        print(f"✗ Mapping loading failed: {e}")
        return False


def main():
    """Run all production 49-label tests."""
    print("\n" + "="*60)
    print("PRODUCTION 49-LABEL COMPATIBILITY TEST SUITE")
    print("="*60)
    print("Validating system readiness for production deployment")
    print("="*60)
    
    results = []
    results.append(test_label_count())
    results.append(test_label_schema())
    results.append(test_entity_coverage())
    results.append(test_model_instantiation_49())
    results.append(test_forward_pass_49())
    results.append(test_production_config())
    results.append(test_output_mapping())
    
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED!")
        print("\nProduction System Status:")
        print("  ✓ 49-label schema validated")
        print("  ✓ Model compatible with 49 labels")
        print("  ✓ Forward pass working")
        print("  ✓ Production configs ready")
        print("  ✓ Output mapping complete")
        print("\n🚀 System is READY FOR TRAINING")
        print("   Next: Acquire annotated training data")
        return 0
    else:
        print("\n⚠ Some tests failed - review above")
        return 1


if __name__ == "__main__":
    exit(main())
