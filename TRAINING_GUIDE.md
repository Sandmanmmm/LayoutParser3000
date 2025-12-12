# Training Guide - LayoutParser3000

## 🚀 Quick Start Training

This guide will help you prepare and start training the LayoutLMv3 model for invoice and purchase order processing.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Preparation](#data-preparation)
4. [Training Execution](#training-execution)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **GPU**: CUDA-compatible GPU with 8GB+ VRAM (recommended)
  - RTX 3060 (12GB) or better
  - Training on CPU is possible but very slow
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space for models and checkpoints

### Required Software
- Python 3.8+
- CUDA 11.7+ (for GPU training)
- Tesseract OCR (optional, for OCR processing)

---

## Environment Setup

### 1. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install all other requirements
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Check Python packages
python scripts/prepare_training.py --validate-only

# Quick system check
python scripts/setup_check.py
```

---

## Data Preparation

The model requires training data in JSON/JSONL format with OCR tokens, NER labels, and table structures.

### Option 1: Create Sample Data (For Testing)

Create sample data to test the training pipeline:

```bash
python scripts/prepare_training.py --create-sample-data --num-samples 20
```

**⚠️ Note**: This creates synthetic data for testing only. For real training, use actual annotated documents.

### Option 2: Prepare Real Training Data

#### Step 1: Collect Documents
Gather invoice and purchase order documents (PDF or images):
- Minimum 500 documents recommended
- Split: 70% train, 15% validation, 15% test

#### Step 2: Extract OCR Data
Use Tesseract, Azure, Google Vision, or AWS Textract:

```bash
# Example with Tesseract
python preprocessing/ocr_processor.py \
    --input data/raw/documents \
    --output data/ocr_results \
    --engine tesseract
```

#### Step 3: Annotate Documents
Use Label Studio or similar tool to annotate:
1. Import OCR results
2. Tag entities with 49-label schema (see `configs/label_list_production.txt`)
3. Mark table structures
4. Export to JSON format

#### Step 4: Format Data
Convert annotations to the required JSON schema:

```json
{
  "metadata": {
    "file_name": "invoice_001.pdf",
    "document_type": "invoice",
    "pages": 1
  },
  "ocr": [
    {
      "token_id": 0,
      "text": "Invoice",
      "bbox": [50, 50, 150, 80],
      "page": 1,
      "confidence": 0.99
    }
  ],
  "ner_tags": ["O", "B-DOCUMENT_NUMBER", ...],
  "tables": [...],
  "ground_truth": {...}
}
```

#### Step 5: Organize Data
Place formatted data in `data/processed/`:
- `train.jsonl` - Training set (500-1000 samples)
- `val.jsonl` - Validation set (100-200 samples)
- `test.jsonl` - Test set (100-200 samples)

#### Step 6: Validate Data

```bash
python scripts/validate_data.py --split all
```

### Option 3: Use Public Dataset (CORD)

Download and convert the CORD dataset:

```bash
# 1. Download CORD
git clone https://github.com/clovaai/cord.git data/cord

# 2. Convert to production schema (create conversion script)
python scripts/convert_cord_to_production.py \
    --input data/cord \
    --output data/processed

# 3. Validate
python scripts/validate_data.py
```

---

## Training Execution

### Configuration

The production configuration is in `configs/training_config_production.yaml`:

**Key Settings:**
- **Labels**: 49 (O + 24 entity types × 2)
- **Batch size**: 2 (effective 16 with gradient accumulation)
- **Epochs**: 12
- **Learning rate**: 3e-5
- **FP16**: Enabled (mixed precision)
- **CRF**: Enabled (better boundary detection)

### Start Training

**Simple method:**
```bash
python scripts/start_training.py
```

**With custom config:**
```bash
python scripts/start_training.py --config configs/training_config_production.yaml
```

**Direct trainer call:**
```bash
python training/trainer.py --config configs/training_config_production.yaml
```

### What Happens During Training

1. **Initialization** (1-2 minutes)
   - Loads configuration
   - Initializes LayoutLMv3 model (125M parameters)
   - Prepares dataloaders
   - Sets up optimizer and scheduler

2. **Training Loop** (1-4 hours depending on GPU)
   - Trains on training set
   - Validates on validation set every 500 steps
   - Saves checkpoints every 500 steps
   - Logs metrics to TensorBoard

3. **Completion**
   - Saves best model (based on `eval_composite_f1`)
   - Saves final checkpoint
   - Generates training summary

---

## Monitoring

### TensorBoard

Monitor training progress in real-time:

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard --port 6006

# Open in browser
# http://localhost:6006
```

**Metrics to Watch:**
- `train_loss` - Should decrease steadily
- `eval_composite_f1` - Overall performance metric
- `eval_ner_f1` - Entity recognition performance
- `eval_cell_f1` - Table cell detection
- `learning_rate` - Should follow cosine schedule

### Weights & Biases (Optional)

Enable W&B in config:
```yaml
logging:
  wandb_enabled: true
  wandb_project: layoutlmv3-invoice-production
```

Then run:
```bash
wandb login
python scripts/start_training.py
```

### Training Logs

Logs are saved to:
- `logs/training.log` - Detailed text logs
- `logs/tensorboard/` - TensorBoard events
- `models/checkpoints/` - Model checkpoints

---

## Expected Results

### Training Time

| GPU | Training Time (12 epochs) |
|-----|---------------------------|
| RTX 3060 (12GB) | 3-4 hours |
| RTX 3090 (24GB) | 1.5-2 hours |
| V100 (32GB) | 1-1.5 hours |
| A100 (40GB) | 40-60 minutes |

### Target Metrics (With Good Data)

| Metric | Target | Critical |
|--------|--------|----------|
| Composite F1 | >0.85 | ✅ |
| NER F1 | >0.85 | ✅ |
| Cell F1 | >0.88 | ✅ |
| Column Accuracy | >0.75 | ⚠️ |

### Per-Entity Goals

| Entity | Target F1 | Priority |
|--------|-----------|----------|
| DOCUMENT_NUMBER | >0.90 | 🔴 Critical |
| SUPPLIER_NAME | >0.85 | 🔴 Critical |
| TOTAL_AMOUNT | >0.85 | 🔴 Critical |
| DOCUMENT_DATE | >0.80 | 🔴 Critical |
| ITEM_DESCRIPTION | >0.80 | 🟡 Important |
| QUANTITY | >0.85 | 🟡 Important |
| UNIT_PRICE | >0.85 | 🟡 Important |

---

## Troubleshooting

### CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in config:
   ```yaml
   training:
     per_device_train_batch_size: 1  # Reduce from 2
   ```

2. Increase gradient accumulation:
   ```yaml
   training:
     gradient_accumulation_steps: 16  # Increase from 8
   ```

3. Disable FP16 (not recommended):
   ```yaml
   training:
     fp16: false
   ```

4. Use smaller model:
   ```yaml
   model:
     model_name_or_path: microsoft/layoutlmv3-base  # Already using base
   ```

### Slow Training

**Symptoms**: Training takes much longer than expected

**Check**:
1. CUDA is being used:
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   ```

2. FP16 is enabled (config)

3. Data loading is not bottleneck:
   ```yaml
   training:
     dataloader_num_workers: 4  # Increase if CPU-bound
   ```

### Poor Model Performance

**Symptoms**: Low F1 scores after training

**Possible causes**:
1. **Insufficient training data**
   - Need 500+ annotated documents
   - Each entity type needs 20+ examples

2. **Inconsistent annotations**
   - Run validation: `python scripts/validate_data.py`
   - Check for BIO tag errors
   - Verify label consistency

3. **Wrong hyperparameters**
   - Try higher learning rate: 5e-5
   - Train for more epochs: 20-30
   - Adjust loss weights if multi-task performance is unbalanced

4. **Data quality issues**
   - Low OCR quality
   - Inconsistent document formats
   - Poor table annotations

### Model Not Converging

**Symptoms**: Loss plateaus or increases

**Solutions**:
1. Check learning rate:
   ```yaml
   training:
     learning_rate: 3e-5  # Try 2e-5 or 5e-5
   ```

2. Enable gradient clipping:
   ```yaml
   training:
     max_grad_norm: 1.0
   ```

3. Adjust warmup:
   ```yaml
   training:
     warmup_ratio: 0.1  # Try 0.15
   ```

### Data Loading Errors

**Symptoms**: Errors during data loading

**Check**:
1. Data format is correct:
   ```bash
   python scripts/validate_data.py
   ```

2. All required fields present

3. Label schema matches config

---

## Advanced Topics

### Resume Training from Checkpoint

```bash
python scripts/start_training.py --resume models/checkpoints/checkpoint-1000
```

### Custom Label Schema

To use a different label schema:

1. Create new label file: `configs/label_list_custom.txt`
2. Update config:
   ```yaml
   model:
     num_labels: YOUR_NUM_LABELS
   data:
     label_list_path: configs/label_list_custom.txt
   ```

### Multi-GPU Training

For multiple GPUs, use PyTorch's DistributedDataParallel:

```bash
torchrun --nproc_per_node=2 training/trainer.py \
    --config configs/training_config_production.yaml
```

### Hyperparameter Tuning

Key hyperparameters to tune:
- `learning_rate` (2e-5 to 5e-5)
- `num_epochs` (10-30)
- `loss_weights` (balance NER vs table tasks)
- `warmup_ratio` (0.05-0.15)

---

## Post-Training

### Evaluate on Test Set

```bash
python evaluation/evaluate.py \
    --model models/checkpoints/layoutlmv3_production/best_model.pt \
    --config configs/training_config_production.yaml \
    --output logs/test_results.json
```

### Export Model

For deployment, export to ONNX:

```bash
python scripts/export_to_onnx.py \
    --model models/checkpoints/layoutlmv3_production/best_model.pt \
    --output models/production/model.onnx
```

### Inference

Test the trained model:

```bash
python scripts/inference.py \
    --model models/checkpoints/layoutlmv3_production/best_model.pt \
    --input data/test_documents/invoice.pdf \
    --output predictions.json
```

---

## Checklist

### Pre-Training
- [ ] Environment set up (Python, CUDA, dependencies)
- [ ] Data prepared (train/val/test splits)
- [ ] Data validated (no errors)
- [ ] Configuration reviewed
- [ ] Label schema confirmed (49 labels)

### During Training
- [ ] TensorBoard running
- [ ] Metrics improving
- [ ] No CUDA OOM errors
- [ ] Checkpoints saving

### Post-Training
- [ ] Test set evaluation
- [ ] Per-entity F1 scores acceptable
- [ ] Model exported for deployment
- [ ] Inference tested

---

## Support

For issues or questions:
1. Check logs in `logs/training.log`
2. Review configuration in `configs/`
3. Run validation scripts
4. Check documentation in `docs/`

---

**Ready to train? Run:**
```bash
python scripts/start_training.py
```

Good luck! 🚀
