# Quick Reference Guide

## 🚀 Quick Start Commands

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify environment
python scripts/setup_check.py
```

### Data Preparation
```bash
# Run OCR
python preprocessing/ocr_processor.py \
    --input data/raw/invoices \
    --output data/ocr_results

# Split dataset (after annotation)
python -c "from utils import split_dataset; from pathlib import Path; \
    split_dataset(Path('data/annotations'), Path('data/processed'))"
```

### Training
```bash
# Train model
python training/trainer.py --config configs/training_config.yaml

# Monitor training
tensorboard --logdir logs/tensorboard
```

### Evaluation
```bash
# Evaluate on test set
python evaluation/evaluate.py \
    --model models/checkpoints/best_model.pt \
    --config configs/training_config.yaml
```

### Inference
```bash
# Single document
python scripts/inference.py \
    --model models/checkpoints/best_model.pt \
    --image path/to/invoice.png

# Batch processing
python scripts/inference.py \
    --model models/checkpoints/best_model.pt \
    --images-dir data/new_invoices \
    --output predictions/
```

## 📊 Key Configuration Parameters

### Training (`configs/training_config.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 30 | Number of training epochs |
| `batch_size` | 4 | Batch size per GPU |
| `learning_rate` | 5e-5 | Learning rate |
| `fp16` | true | Use mixed precision |
| `gradient_accumulation_steps` | 4 | Accumulation steps |
| `use_crf` | true | Enable CRF layer |

### Model (`configs/training_config.yaml`)

| Parameter | Options | Description |
|-----------|---------|-------------|
| `name` | `layoutlmv3-base/large` | Base model |
| `num_labels` | 13 | Number of entity types |
| `use_crf` | true/false | CRF for sequences |

## 📁 Directory Structure

```
├── configs/              # Configuration files
├── data/                 # Data directory
│   ├── raw/             # Raw documents
│   ├── processed/       # Train/val/test splits
│   └── annotations/     # Annotation files
├── models/              # Model code & checkpoints
├── preprocessing/       # OCR & data processing
├── training/            # Training scripts
├── evaluation/          # Evaluation scripts
├── utils/               # Utilities
├── scripts/             # End-to-end scripts
├── docs/                # Documentation
└── logs/                # Training logs
```

## 🏷️ Label Schema (BIO Format)

### Entity Types (Example)
- `B-DATE` / `I-DATE`: Invoice/PO dates
- `B-INVOICE_NUMBER` / `I-INVOICE_NUMBER`: Invoice number
- `B-VENDOR_NAME` / `I-VENDOR_NAME`: Vendor name
- `B-TOTAL_AMOUNT` / `I-TOTAL_AMOUNT`: Total amount
- `O`: Outside any entity

### Annotation Format
```json
{
  "image_path": "invoice.png",
  "words": [
    {"text": "Invoice", "bbox": [x1, y1, x2, y2], "label": "O"},
    {"text": "INV-123", "bbox": [x1, y1, x2, y2], "label": "B-INVOICE_NUMBER"}
  ]
}
```

## 🔧 Common Tasks

### Change Learning Rate
Edit `configs/training_config.yaml`:
```yaml
training:
  learning_rate: 3e-5  # Change this
```

### Enable Weights & Biases
```yaml
logging:
  use_wandb: true
  wandb_project: "your-project"
```

### Reduce Memory Usage
```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8

hardware:
  gradient_checkpointing: true
```

### Add New Entity Type
1. Add to `configs/training_config.yaml`:
```yaml
labels:
  token_classification:
    - "B-NEW_ENTITY"
    - "I-NEW_ENTITY"
```
2. Update `num_labels` accordingly
3. Annotate data with new labels
4. Retrain model

## 🐛 Troubleshooting

### CUDA Out of Memory
```yaml
# Reduce batch size
training:
  batch_size: 2
  gradient_accumulation_steps: 8
```

### Low OCR Quality
```yaml
# Adjust in configs/data_config.yaml
ocr:
  preprocessing:
    denoise: true
    deskew: true
    resize_dpi: 300
```

### Model Not Learning
- Check learning rate (try 3e-5 to 7e-5)
- Verify data quality and labels
- Increase training epochs
- Check for class imbalance

### Slow Training
- Enable FP16: `fp16: true`
- Increase workers: `num_workers: 4`
- Use gradient checkpointing: `gradient_checkpointing: true`

## 📊 Metrics Explained

| Metric | What it measures |
|--------|------------------|
| **Accuracy** | Overall token classification accuracy |
| **Precision** | % of predicted entities that are correct |
| **Recall** | % of actual entities that are found |
| **F1** | Harmonic mean of precision and recall |
| **SeqEval F1** | F1 using proper NER evaluation (recommended) |

## 💡 Best Practices

### Data
- ✅ Diverse templates (multiple vendors)
- ✅ Balanced entity types
- ✅ Consistent labeling
- ✅ 100+ annotated documents minimum

### Training
- ✅ Start with base model (faster)
- ✅ Use cosine LR scheduling
- ✅ Enable early stopping
- ✅ Monitor validation metrics
- ✅ Save checkpoints regularly

### Evaluation
- ✅ Test on diverse holdout set
- ✅ Check per-entity F1 scores
- ✅ Visualize predictions
- ✅ Analyze failure cases

## 🔗 Useful Resources

- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [Hugging Face Docs](https://huggingface.co/docs/transformers)
- [Complete README](../README.md)
- [Data Preparation Guide](DATA_PREPARATION.md)
- [Training Guide](TRAINING_GUIDE.md)
- [Example Workflows](EXAMPLE_WORKFLOW.md)

## 🆘 Getting Help

1. Check documentation in `docs/`
2. Review configuration files in `configs/`
3. Check logs in `logs/`
4. Verify with `python scripts/setup_check.py`

## ⚡ Performance Tips

### GPU Utilization
- Increase batch size (if memory allows)
- Use FP16 mixed precision
- Enable pinned memory

### Data Loading
- Use multiple workers
- Enable prefetching
- Cache preprocessed data

### Training Speed
- Use cosine scheduler
- Start with smaller model
- Profile with TensorBoard

## 📈 Typical Results

On invoice dataset with ~1000 samples:

| Metric | Expected Range |
|--------|----------------|
| Overall F1 | 85-95% |
| Invoice Number F1 | 90-98% |
| Date F1 | 85-95% |
| Total Amount F1 | 85-95% |
| Training Time | 2-5 hours (V100) |

Results vary based on:
- Data quality
- Template diversity
- Annotation consistency
- Model size
- Hyperparameters
