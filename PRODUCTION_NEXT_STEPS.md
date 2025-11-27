# Production Training Configuration - Next Steps Analysis

## Current Status ✅

### Completed
1. ✅ **Production Label Schema**: Updated `training_config.yaml` with 57 labels (28 entities × 2 + O)
2. ✅ **Label List File**: Created `configs/label_list.txt` with all production labels
3. ✅ **Output Mapping**: Created `configs/output_mapping.yaml` for DB field mapping
4. ✅ **Annotation Documentation**: Created `docs/ANNOTATION_FORMAT.md` with complete schema
5. ✅ **Multi-task Configuration**: Added table detection settings (cell + column classification)
6. ✅ **Loss Weights**: Configured weighted multi-task loss (NER: 1.0, Cell: 1.0, Col: 0.5)

### Configuration Summary
- **Model**: LayoutLMv3-base with CRF
- **Labels**: 57 (O + 28 entities in BIO format)
- **Batch Size**: Effective 16 (2 per device × 8 grad accum)
- **Epochs**: 12
- **Learning Rate**: 3e-5 with 6% warmup, cosine schedule
- **Tasks**: Token classification (NER) + Table structure (cell + column)

## Critical Next Steps 🚀

### Phase 1: Code Implementation (Before Training)

#### 1. Install torchcrf Dependency ⚠️ REQUIRED
```powershell
.\.venv\Scripts\python.exe -m pip install pytorch-crf
```

**Why**: CRF layer requires `torchcrf` or `pytorch-crf` package. Without it, training will fail.

**Status**: Not installed yet

---

#### 2. Update Multi-Task Model Architecture 🔴 CRITICAL

**File**: `models/layoutlmv3_model.py`

**Current State**: Basic token classification with CRF
**Required State**: Multi-task with table detection heads

**Changes Needed**:

```python
# Add to LayoutLMv3ForTokenClassification class:

# New heads (add to __init__)
self.cell_classifier = nn.Linear(hidden_size, 2)  # in-cell vs not
self.col_classifier = nn.Linear(hidden_size, config.table_num_cols)  # column index

# Update forward() to return cell_logits, col_logits
# Compute multi-task loss: ner_loss + cell_loss + col_loss
```

**Key Implementation Points**:
- Add `cell_classifier` head (binary: in-table-cell or not)
- Add `col_classifier` head (predicts column index 0-15)
- Combine losses with weights from config
- Handle optional labels (cell_labels, col_labels can be None)

**Estimated Time**: 2-3 hours

---

#### 3. Update Dataset for Multi-Task Training 🔴 CRITICAL

**File**: `preprocessing/dataset.py`

**Current State**: Loads `input_ids`, `bbox`, `images`, `labels`
**Required State**: Also loads `cell_labels`, `col_labels`

**Changes Needed**:

```python
class InvoiceDataset:
    def __getitem__(self, idx):
        # Existing: load OCR, ner_tags
        
        # NEW: Derive cell labels
        cell_labels = self._get_cell_labels(annotation["ocr"], annotation["tables"])
        
        # NEW: Derive column labels
        col_labels = self._get_col_labels(annotation["ocr"], annotation["tables"])
        
        # Align to subword tokens (use -100 for padding)
        
        return {
            "input_ids": ...,
            "bbox": ...,
            "images": ...,
            "labels": ...,
            "cell_labels": ...,  # NEW
            "col_labels": ...    # NEW
        }
```

**Key Implementation Points**:
- Map `token_ids` from table cells to tokens
- Handle subword tokenization (align cell/col labels to wordpieces)
- Use `-100` for ignored tokens in labels
- Validate annotation format

**Estimated Time**: 3-4 hours

---

#### 4. Update Trainer for Multi-Task Loss 🟡 MEDIUM PRIORITY

**File**: `training/trainer.py`

**Current State**: Expects `loss` from model
**Required State**: Handles multi-task losses dict

**Changes Needed**:

```python
# In train_epoch():
outputs = model(**batch)
loss = outputs["loss"]  # Combined multi-task loss
losses_dict = outputs.get("losses", {})  # Individual losses

# Log individual losses
if losses_dict:
    for loss_name, loss_value in losses_dict.items():
        wandb.log({f"train/{loss_name}": loss_value})
```

**Estimated Time**: 1-2 hours

---

#### 5. Update Evaluation Metrics 🟡 MEDIUM PRIORITY

**File**: `evaluation/metrics.py`

**Current State**: Token-level NER metrics (precision, recall, F1)
**Required State**: + Line-item extraction + Table IoU

**New Metrics Needed**:

1. **Line-Item Extraction Metrics**:
   - Treat each extracted row as a structured record
   - Match predicted line items to ground truth by SKU/description
   - Compute precision/recall/F1 for complete records

2. **Table Detection IoU**:
   - If predicting table bboxes: compute IoU with ground truth
   - Otherwise: compute cell-level accuracy

**Estimated Time**: 2-3 hours

---

#### 6. Update Inference Pipeline 🟢 LOW PRIORITY (Post-training)

**File**: `scripts/inference.py`

**Current State**: Decodes NER tags only
**Required State**: Decodes NER + groups tokens into table rows → line items

**Changes Needed**:
- Decode `cell_logits` and `col_logits`
- Cluster tokens by row (vertical bbox grouping)
- Form line items from row tokens
- Generate output JSON matching `output_mapping.yaml`

**Estimated Time**: 4-5 hours

**Priority**: Can be done after initial training

---

### Phase 2: Data Acquisition (Parallel with Code)

#### Option A: Public Dataset (FASTEST - 2-3 hours)

**CORD Dataset**: https://github.com/clovaai/cord
- 800+ annotated receipts
- Similar to invoices (line items, totals)
- Already has bboxes + entity labels

**Steps**:
1. Download CORD dataset
2. Convert to our JSON format (script needed)
3. Map CORD labels to our 57-label schema
4. Split into train/val/test

**Conversion Script Needed**: 30 min to write

---

#### Option B: Minimal Test Dataset (1-2 days)

**Collect 10-20 sample documents:**
1. Find 10-20 invoice/PO PDFs online or from business
2. Run OCR: `python preprocessing/ocr_processor.py --input data/raw/invoices --output data/ocr_results`
3. Create annotations manually or with Label Studio
4. Follow `docs/ANNOTATION_FORMAT.md` schema

**Pros**: Tests full pipeline, customized to exact use case
**Cons**: Time-consuming, small dataset (limited model performance)

---

#### Option C: Production Dataset (2-4 weeks)

**For serious deployment:**
- 500-1000+ varied documents
- Professional annotation service or internal team
- Full coverage of all 28 entity types
- Diverse suppliers, formats, qualities

**Not recommended for initial prototype**

---

### Phase 3: Pre-Training Validation

Before starting training, run these checks:

```powershell
# 1. Environment check
.\.venv\Scripts\python.exe scripts/setup_check.py

# 2. Annotation validation
.\.venv\Scripts\python.exe utils/validate_annotations.py --annotations data/annotations/

# 3. Dataset statistics
.\.venv\Scripts\python.exe utils/data_utils.py stats --annotations data/annotations/

# 4. Test data loading
.\.venv\Scripts\python.exe preprocessing/dataset.py --test --config configs/training_config.yaml
```

---

## Recommended Immediate Action Plan

### Today (4-6 hours)

1. **Install torchcrf** (5 min):
   ```powershell
   .\.venv\Scripts\python.exe -m pip install pytorch-crf
   ```

2. **Update Model Architecture** (2-3 hours):
   - Add cell_classifier and col_classifier heads
   - Implement multi-task loss computation
   - Test forward pass with dummy data

3. **Update Dataset** (2-3 hours):
   - Add cell_labels and col_labels derivation
   - Test with sample annotation
   - Validate shapes and alignment

### Tomorrow (2-4 hours)

4. **Update Trainer** (1-2 hours):
   - Handle multi-task losses
   - Add logging for individual losses

5. **Data Acquisition Decision**:
   - **RECOMMENDED**: Start with CORD dataset conversion (2-3 hours)
   - OR: Begin collecting minimal test set (ongoing)

### Day 3+ (Train!)

6. **Convert Data** (if using CORD):
   - Write conversion script
   - Generate train/val/test splits
   - Validate annotations

7. **Start Training**:
   ```powershell
   .\.venv\Scripts\python.exe training/trainer.py --config configs/training_config.yaml
   ```

8. **Monitor**:
   ```powershell
   .\.venv\Scripts\tensorboard.exe --logdir logs/
   ```

---

## Expected Timeline

| Task | Priority | Time | Status |
|------|----------|------|--------|
| Install torchcrf | 🔴 Critical | 5 min | ⬜ Not started |
| Update model architecture | 🔴 Critical | 2-3 hours | ⬜ Not started |
| Update dataset.py | 🔴 Critical | 3-4 hours | ⬜ Not started |
| Update trainer.py | 🟡 High | 1-2 hours | ⬜ Not started |
| Update metrics.py | 🟡 Medium | 2-3 hours | ⬜ Not started |
| Acquire training data | 🔴 Critical | 2-3 hours (CORD) | ⬜ Not started |
| Update inference.py | 🟢 Low | 4-5 hours | ⬜ Post-training |
| **TOTAL (Before Training)** | | **10-15 hours** | |

---

## Risks & Mitigations

### Risk 1: No Training Data
**Impact**: Cannot train model
**Mitigation**: Start with CORD dataset conversion (fastest path)
**Status**: 🔴 Critical blocker

### Risk 2: OOM Errors
**Impact**: Training crashes
**Mitigation**: Reduce `per_device_train_batch_size` to 1, increase `gradient_accumulation_steps`
**Status**: 🟡 Potential issue (CPU-only, limited RAM)

### Risk 3: Model Complexity
**Impact**: Training unstable or slow convergence
**Mitigation**: Start with NER-only (disable table heads), add multi-task incrementally
**Status**: 🟢 Low risk

### Risk 4: Annotation Quality
**Impact**: Poor model performance
**Mitigation**: Validate all annotations, spot-check 10% manually
**Status**: 🟡 Depends on data source

---

## Success Metrics (After Training)

### Minimum Viable Model
- **Token-level F1**: > 0.70 (all entities)
- **Line-item extraction F1**: > 0.60 (complete records)
- **Inference time**: < 5 seconds per page

### Production-Ready Model
- **Token-level F1**: > 0.85 (all entities)
- **Line-item extraction F1**: > 0.80 (complete records)
- **Per-entity F1**: > 0.80 for critical fields (TOTAL_AMOUNT, DOCUMENT_NUMBER, SKU)
- **Confidence calibration**: Auto-approve threshold at 0.75 confidence

---

## Next Commands to Run

```powershell
# 1. Install CRF dependency
.\.venv\Scripts\python.exe -m pip install pytorch-crf

# 2. Test model architecture changes (after implementation)
.\.venv\Scripts\python.exe -c "from models.layoutlmv3_model import LayoutLMv3ForTokenClassification; print('Model import successful')"

# 3. Test dataset loading (after implementation)
.\.venv\Scripts\python.exe -c "from preprocessing.dataset import InvoiceDataset; print('Dataset import successful')"

# 4. Validate environment
.\.venv\Scripts\python.exe scripts/setup_check.py

# 5. When ready: Start training
.\.venv\Scripts\python.exe training/trainer.py --config configs/training_config.yaml
```

---

## Questions to Answer

1. **Data Source**: Use CORD dataset (fast) or collect custom data (slow)?
   - **Recommendation**: CORD for prototype, custom for production

2. **GPU Access**: Can you get GPU access for training?
   - **Current**: CPU-only (very slow)
   - **Recommendation**: Use Google Colab (free GPU) or Azure VM

3. **Annotation Tool**: Manual JSON or Label Studio?
   - **Recommendation**: Label Studio for > 10 documents

4. **Timeline**: Prototype in 1 week or production in 1 month?
   - **Prototype**: CORD dataset + MVP model
   - **Production**: Custom data + full pipeline

---

## Summary

**Current Blocker**: No training data yet

**Critical Path**: 
1. Install torchcrf (5 min)
2. Update model + dataset code (5-7 hours)
3. Convert CORD dataset (2-3 hours)
4. Start training (overnight)

**ETA to Training**: 1-2 days (if starting now)

**Documentation Status**: ✅ Complete (configs, schemas, mappings all ready)
