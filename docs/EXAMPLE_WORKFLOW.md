# Example: Complete Training Workflow

This example demonstrates the complete workflow from data preparation to deployment.

## Sample Data Structure

```
data/
├── raw/
│   └── invoices/
│       ├── invoice_001.png
│       ├── invoice_002.png
│       └── ...
├── annotations/
│   ├── invoice_001.json
│   ├── invoice_002.json
│   └── ...
└── processed/
    ├── train/
    ├── val/
    └── test/
```

## Step-by-Step Example

### 1. Environment Setup

```bash
# Run environment check
python scripts/setup_check.py
```

### 2. Data Preparation

```python
# prepare_data.py
from pathlib import Path
from preprocessing.ocr_processor import OCRProcessor
from utils import split_dataset

# Process documents with OCR
processor = OCRProcessor("./configs/data_config.yaml")

invoice_dir = Path("data/raw/invoices")
for invoice in invoice_dir.glob("*.png"):
    print(f"Processing {invoice.name}")
    result = processor.process_document(invoice)
    
    if result and result['quality_pass']:
        # Save OCR result
        output_file = Path("data/ocr_results") / f"{invoice.stem}.json"
        with open(output_file, 'w') as f:
            import json
            json.dump(result, f, indent=2)

# After annotation, split dataset
split_dataset(
    data_dir=Path("data/annotations"),
    output_dir=Path("data/processed"),
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)
```

### 3. Training

```bash
# Start training
python training/trainer.py --config configs/training_config.yaml

# Monitor with TensorBoard
tensorboard --logdir logs/tensorboard
```

### 4. Evaluation

```bash
# Evaluate on test set
python evaluation/evaluate.py \
    --model models/checkpoints/best_model.pt \
    --config configs/training_config.yaml \
    --output logs/test_results.json
```

### 5. Inference on New Documents

```python
# inference_example.py
from pathlib import Path
from evaluation.evaluate import Evaluator
from preprocessing.ocr_processor import OCRProcessor
from utils.visualization import visualize_predictions

# Initialize
evaluator = Evaluator(
    Path("models/checkpoints/best_model.pt"),
    Path("configs/training_config.yaml")
)

ocr_processor = OCRProcessor()

# Process new invoice
new_invoice = Path("data/new/invoice_new.png")
ocr_result = ocr_processor.process_document(new_invoice)

# Run inference
predictions = evaluator.predict_single_document(
    new_invoice,
    ocr_result
)

# Extract entities
entities = {}
current_entity = None

for pred in predictions['predictions']:
    label = pred['label']
    
    if label.startswith('B-'):
        entity_type = label[2:]
        current_entity = {
            'type': entity_type,
            'text': pred['text'],
            'bbox': pred['bbox']
        }
        
        if entity_type not in entities:
            entities[entity_type] = []
        entities[entity_type].append(current_entity)
    
    elif label.startswith('I-') and current_entity:
        current_entity['text'] += ' ' + pred['text']

# Print extracted information
print("\nExtracted Information:")
print("=" * 50)

for entity_type, entity_list in entities.items():
    print(f"\n{entity_type}:")
    for entity in entity_list:
        print(f"  {entity['text']}")

# Visualize
visualize_predictions(
    new_invoice,
    predictions['predictions'],
    save_path="output/visualization.png"
)
```

## Batch Processing Example

```python
# batch_inference.py
from pathlib import Path
from scripts.inference import batch_inference

# Process all invoices in a directory
batch_inference(
    model_path="models/checkpoints/best_model.pt",
    images_dir="data/new_invoices",
    output_dir="output/predictions",
    config_path="configs/training_config.yaml"
)
```

## Custom Training Loop Example

```python
# custom_training.py
import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor
from pathlib import Path
import yaml

from models import LayoutLMv3ForTokenClassification
from preprocessing import InvoiceDataset, DocumentAugmentation
from training import Trainer

# Load config
with open("configs/training_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = LayoutLMv3Processor.from_pretrained(config['model']['name'])

# Create label mapping
labels = config['labels']['token_classification']
label2id = {label: i for i, label in enumerate(labels)}

# Create datasets
augmentation = DocumentAugmentation()

train_dataset = InvoiceDataset(
    Path("data/processed/train"),
    processor,
    label2id,
    augmentation=augmentation,
    mode="train"
)

val_dataset = InvoiceDataset(
    Path("data/processed/val"),
    processor,
    label2id,
    mode="val"
)

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config['training']['batch_size'],
    shuffle=False,
    num_workers=4
)

# Initialize model
model = LayoutLMv3ForTokenClassification.from_pretrained(
    config['model']['name'],
    num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id=label2id
)

# Train
trainer = Trainer(
    model,
    train_loader,
    val_loader,
    config,
    device=device
)

trainer.train()
```

## Production Deployment Example

```python
# deploy.py
from fastapi import FastAPI, File, UploadFile
from pathlib import Path
import shutil

from evaluation.evaluate import Evaluator
from preprocessing.ocr_processor import OCRProcessor

app = FastAPI()

# Load model once
evaluator = Evaluator(
    Path("models/checkpoints/best_model.pt"),
    Path("configs/training_config.yaml")
)

ocr_processor = OCRProcessor()

@app.post("/extract")
async def extract_invoice(file: UploadFile = File(...)):
    """Extract information from invoice."""
    
    # Save uploaded file
    temp_path = Path(f"temp/{file.filename}")
    temp_path.parent.mkdir(exist_ok=True)
    
    with temp_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process
        ocr_result = ocr_processor.process_document(temp_path)
        predictions = evaluator.predict_single_document(temp_path, ocr_result)
        
        # Extract entities
        entities = {}
        for pred in predictions['predictions']:
            label = pred['label']
            if label != 'O':
                entity_type = label.split('-')[1]
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(pred['text'])
        
        return {
            "status": "success",
            "entities": entities,
            "confidence": ocr_result['average_confidence']
        }
    
    finally:
        # Cleanup
        temp_path.unlink()

# Run with: uvicorn deploy:app --reload
```

## Tips for Success

### 1. Start Small
- Begin with 50-100 annotated samples
- Validate pipeline end-to-end
- Gradually scale up

### 2. Monitor Quality
```python
from utils import calculate_dataset_statistics

stats = calculate_dataset_statistics(Path("data/processed/train"))
print(f"Average entities per doc: {stats['avg_entities_per_sample']:.1f}")
```

### 3. Iterative Improvement
1. Train initial model
2. Identify failure cases
3. Add more training examples for weak entities
4. Retrain and evaluate
5. Repeat

### 4. Hyperparameter Search
```python
# Try different learning rates
for lr in [3e-5, 5e-5, 7e-5]:
    config['training']['learning_rate'] = lr
    # Train and evaluate
```

### 5. Error Analysis
```python
from evaluation import Evaluator

evaluator = Evaluator(model_path, config_path)

# Get predictions
predictions = evaluator.evaluate_test_set(return_predictions=True)

# Find errors
for i, (pred, label) in enumerate(zip(predictions['predictions'], predictions['labels'])):
    if pred != label and label != -100:
        print(f"Sample {i}: Predicted {pred}, Expected {label}")
```

## Common Workflows

### Quick Test
```bash
# Test on small dataset
python training/trainer.py --config configs/training_config.yaml --max-samples 100
```

### Full Production Pipeline
```bash
# Complete pipeline
python scripts/run_pipeline.py \
    --raw-data data/raw \
    --output output \
    --config configs/training_config.yaml
```

### Continuous Training
```bash
# Retrain with new data
python training/trainer.py \
    --config configs/training_config.yaml \
    --resume models/checkpoints/best_model.pt \
    --additional-data data/new_annotations
```

## Next Steps

1. Review the complete [README.md](../README.md)
2. Follow [DATA_PREPARATION.md](DATA_PREPARATION.md) for data setup
3. Check [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for training details
4. Start with the quick start example above
5. Scale to your full dataset
