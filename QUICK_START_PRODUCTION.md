# Production Training - Quick Reference Card

## 🚀 System Status: READY FOR DATA

**Version**: 1.0.0-production  
**Label Schema**: 49 labels (24 entity types)  
**Tests**: 7/7 passing ✅

---

## 📋 Production Configuration

### Label Schema (49 labels)
```
O (1) + 24 entity types × 2 (B-/I-) = 49 labels
```

**File**: `configs/label_list_production.txt`

**Entities**:
- **Document (16)**: SUPPLIER_NAME, SUPPLIER_ADDRESS, BUYER_NAME, BUYER_ADDRESS, DOCUMENT_NUMBER, DOCUMENT_DATE, DUE_DATE, CURRENCY, SUBTOTAL, TAX, SHIPPING, TOTAL_AMOUNT, PAYMENT_TERMS, ORDER_REFERENCE, VENDOR_TAX_ID, ACCOUNT_NUMBER
- **Line-Item (8)**: ITEM_DESCRIPTION, SKU, QUANTITY, UOM, UNIT_PRICE, LINE_TOTAL, PACK_SIZE, UNIT_COST

---

## 🔧 Configuration Files

| File | Purpose | Status |
|------|---------|--------|
| `configs/label_list_production.txt` | 49-label schema | ✅ Ready |
| `configs/training_config_production.yaml` | Training settings | ✅ Ready |
| `configs/output_mapping_production.yaml` | Database mapping | ✅ Ready |

---

## 🎯 Training Command

```bash
# Activate environment
.\.venv\Scripts\activate

# Run training
python training/trainer.py --config configs/training_config_production.yaml

# Monitor
tensorboard --logdir logs/tensorboard
```

---

## 📊 Key Settings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `num_labels` | 49 | Production schema |
| `use_crf` | true | Better boundaries |
| `batch_size` | 2 | GPU memory |
| `gradient_accumulation` | 8 | Effective=16 |
| `learning_rate` | 3e-5 | Fine-tuning |
| `num_epochs` | 12 | Convergence |
| `fp16` | true | Speed + memory |

**Loss Weights**: NER=1.0, Cell=1.0, Col=0.5  
**Metric**: eval_composite_f1

---

## 📝 Annotation Format

```json
{
  "ocr": [{"token_id": 0, "text": "...", "bbox": [...], "page": 1}],
  "ner_tags": ["O", "B-DOCUMENT_NUMBER", "I-DOCUMENT_NUMBER", ...],
  "tables": [{"cells": [{"row": 0, "col": 0, "token_ids": [...]}]}],
  "ground_truth": {"DOCUMENT_NUMBER": "...", "TOTAL_AMOUNT": "..."}
}
```

---

## ✅ Test Validation

```bash
# Quick test (30 seconds)
python scripts/test_production_49labels.py
# Expected: 7/7 tests passing
```

---

## 📦 Data Requirements

| Split | Documents | Purpose |
|-------|-----------|---------|
| Train | 500-1000 | Training |
| Val | 100-200 | Validation |
| Test | 100-200 | Evaluation |

**Coverage**: 20+ examples per entity type

---

## 🎓 Expected Accuracy

| Metric | Target | Critical |
|--------|--------|----------|
| Composite F1 | >0.85 | ✅ |
| NER F1 | >0.85 | ✅ |
| DOCUMENT_NUMBER | >0.90 | 🔴 |
| SUPPLIER_NAME | >0.85 | 🔴 |
| TOTAL_AMOUNT | >0.85 | 🔴 |

---

## ⏱️ Training Time

- **RTX 3060**: 3-4 hours
- **RTX 3090**: 1.5-2 hours
- **V100**: 1-1.5 hours
- **A100**: 40-60 min

---

## 🚧 Blocking Issue

**TRAINING DATA NEEDED**

Options:
1. **CORD Dataset**: Quick start, needs label conversion
2. **Custom Data**: Production-ready, needs annotation

---

## 🔍 Database Mapping

**PurchaseOrder** (16 fields):
```yaml
DOCUMENT_NUMBER → number
SUPPLIER_NAME → supplierName
TOTAL_AMOUNT → totalAmount
...
```

**POLineItem** (8 fields):
```yaml
ITEM_DESCRIPTION → productName
QUANTITY → quantity
UNIT_PRICE → unitCost
...
```

---

## 📚 Documentation

- `PRODUCTION_TRAINING_COMPLETE.md` - Full guide
- `PRODUCTION_READINESS_ANALYSIS.md` - System audit
- `PHASE1_COMPLETE.md` - Model architecture
- `PHASE2_COMPLETE.md` - Dataset updates
- `PHASE3_COMPLETE.md` - Trainer updates

---

## ✨ Quick Commands

```bash
# Test production compatibility
python scripts/test_production_49labels.py

# Test Phase 3 updates
python scripts/quick_phase3_test.py

# Check environment
python scripts/setup_check.py

# Train with production config
python training/trainer.py --config configs/training_config_production.yaml
```

---

**Status**: ✅ PRODUCTION READY (pending data)  
**Next**: Acquire training data (Phase 4)
