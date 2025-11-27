# Training Guide

Complete guide for training LayoutLMv3 models on invoice and PO data.

## Prerequisites

Before training:
1. ✅ Data is prepared and annotated
2. ✅ Data is split into train/val/test sets
3. ✅ Configuration files are reviewed
4. ✅ Dependencies are installed

## Configuration

### Training Hyperparameters

Edit `configs/training_config.yaml`:

```yaml
training:
  # Basic settings
  num_epochs: 30
  batch_size: 4  # Per GPU
  gradient_accumulation_steps: 4  # Effective batch = 16
  
  # Learning rate
  learning_rate: 5e-5
  warmup_ratio: 0.1  # 10% warmup
  lr_scheduler_type: "cosine"
  
  # Optimization
  weight_decay: 0.01
  max_grad_norm: 1.0
  
  # Mixed precision
  fp16: true
  
  # Early stopping
  early_stopping_patience: 5
  metric_for_best_model: "eval_f1"
```

### Model Configuration

```yaml
model:
  name: "microsoft/layoutlmv3-base"
  num_labels: 13
  use_crf: true  # Improves sequential predictions
  
  # Enable table detection
  table_structure:
    enabled: true
    num_row_labels: 3
    num_col_labels: 3
```

## Training Commands

### Basic Training

```bash
python training/trainer.py --config configs/training_config.yaml
```

### Resume Training

```bash
python training/trainer.py \
    --config configs/training_config.yaml \
    --resume models/checkpoints/checkpoint_epoch_10.pt
```

### Distributed Training (Multi-GPU)

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    training/trainer.py \
    --config configs/training_config.yaml
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/tensorboard --port 6006
```

View at: http://localhost:6006

Metrics logged:
- Training loss
- Validation loss
- Learning rate
- Accuracy, Precision, Recall, F1
- Per-entity F1 scores

### Weights & Biases

Enable in config:
```yaml
logging:
  use_wandb: true
  wandb_project: "layoutlmv3-invoice-po"
  wandb_entity: "your-username"
```

Login and start training:
```bash
wandb login
python training/trainer.py --config configs/training_config.yaml
```

### Console Logs

Training progress is logged to:
- Console output
- `logs/training_TIMESTAMP.log`

## Training Features

### Mixed Precision (FP16)

Automatically enabled when GPU supports it:
- 2x faster training
- 50% less memory usage
- Minimal accuracy impact

### Gradient Accumulation

Simulates larger batch sizes:
```yaml
batch_size: 4
gradient_accumulation_steps: 4
# Effective batch size = 16
```

### Learning Rate Scheduling

Available schedulers:
- **cosine**: Smooth decay (recommended)
- **linear**: Linear decay
- **polynomial**: Polynomial decay
- **constant**: No decay

With warmup:
```yaml
warmup_ratio: 0.1  # 10% of total steps
```

### Gradient Checkpointing

Save memory at cost of speed:
```yaml
hardware:
  gradient_checkpointing: true
```

### CRF Layer

Improves sequence predictions:
```yaml
model:
  use_crf: true
```

Benefits:
- Better handling of label dependencies
- Enforces valid tag sequences (e.g., B-TAG followed by I-TAG)
- +2-3% F1 improvement typically

## Hyperparameter Tuning

### Learning Rate

Start with: 5e-5 (LayoutLMv3-base), 3e-5 (LayoutLMv3-large)

Too high: Loss spikes, unstable training
Too low: Slow convergence, suboptimal results

### Batch Size

Recommendations:
- LayoutLMv3-base: 8-16 effective batch size
- LayoutLMv3-large: 4-8 effective batch size

Adjust based on GPU memory.

### Number of Epochs

Typical range: 20-50 epochs

Use early stopping to prevent overfitting:
```yaml
early_stopping_patience: 5
```

### Weight Decay

Prevents overfitting:
- Start with: 0.01
- Increase if overfitting: 0.05-0.1
- Decrease if underfitting: 0.001-0.005

## Common Issues

### Out of Memory (OOM)

Solutions:
1. Reduce batch_size
2. Increase gradient_accumulation_steps
3. Enable gradient_checkpointing
4. Use smaller model (base vs large)
5. Reduce max_seq_length

```yaml
training:
  batch_size: 2
  gradient_accumulation_steps: 8

hardware:
  gradient_checkpointing: true
```

### Slow Training

Optimizations:
1. Enable fp16
2. Increase num_workers for data loading
3. Use faster augmentation pipeline
4. Pin memory

```yaml
training:
  fp16: true

data:
  num_workers: 4
  prefetch_factor: 2

hardware:
  pin_memory: true
```

### Overfitting

Signs:
- Train loss decreasing, val loss increasing
- Large gap between train and val metrics

Solutions:
1. Increase weight_decay
2. Enable data augmentation
3. Add dropout
4. Reduce model size
5. Get more training data

```yaml
model:
  dropout: 0.1

augmentation:
  enabled: true

training:
  weight_decay: 0.05
```

### Underfitting

Signs:
- Both train and val loss high
- Poor performance on training set

Solutions:
1. Increase model capacity (use large variant)
2. Train longer
3. Increase learning rate
4. Reduce regularization
5. Check data quality

```yaml
model:
  name: "microsoft/layoutlmv3-large"

training:
  num_epochs: 50
  learning_rate: 5e-5
  weight_decay: 0.001
```

### Poor Entity Recognition

For specific entity types:
1. Check annotation quality
2. Ensure sufficient training examples
3. Use class weights
4. Enable CRF layer

```yaml
model:
  use_crf: true

data:
  use_class_weights: true
```

## Checkpointing

### Automatic Checkpointing

Models are saved:
- Every epoch
- When validation metric improves (best model)

```yaml
training:
  save_strategy: "epoch"
  save_total_limit: 3  # Keep only 3 latest
```

### Manual Checkpoint Loading

```python
import torch
from models import LayoutLMv3ForTokenClassification

# Load checkpoint
checkpoint = torch.load('models/checkpoints/best_model.pt')

# Initialize model
model = LayoutLMv3ForTokenClassification.from_pretrained(
    config['model']['name']
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
```

## Evaluation During Training

Validation runs:
- After each epoch
- Metrics logged to TensorBoard/W&B
- Best model saved automatically

Metrics computed:
- Token-level: Accuracy, P/R/F1
- Sequence-level: SeqEval F1
- Per-entity: F1 for each entity type
- Table detection: P/R/F1 for tables

## Training Time Estimates

Approximate times on NVIDIA V100:

**LayoutLMv3-base:**
- 1000 samples: ~2 hours (30 epochs)
- 5000 samples: ~10 hours (30 epochs)
- 10000 samples: ~20 hours (30 epochs)

**LayoutLMv3-large:**
- 2-3x slower than base

Factors affecting speed:
- Image resolution
- Sequence length
- Batch size
- GPU type
- Data augmentation

## Next Steps

After training:
1. Run full evaluation: `python evaluation/evaluate.py`
2. Visualize predictions
3. Analyze errors
4. Iterate on hyperparameters
5. Deploy model

## Advanced Topics

### Custom Loss Functions

Implement in `models/layoutlmv3_model.py`:

```python
def custom_loss(self, logits, labels, ...):
    # Your custom loss logic
    return loss
```

### Multi-Task Learning

Enable multiple heads:
```yaml
model:
  token_classification:
    enabled: true
  
  table_structure:
    enabled: true
  
  # Add custom heads here
```

### Transfer Learning

Fine-tune from your checkpoint:
```yaml
model:
  name: "./models/checkpoints/best_model_hf"  # Your checkpoint
```

## Resources

- [LayoutLMv3 Paper](https://arxiv.org/abs/2204.08387)
- [Hugging Face Docs](https://huggingface.co/docs/transformers)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
