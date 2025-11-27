# Data Preparation Guide

This guide walks you through preparing your invoice and purchase order data for training.

## Step 1: Organize Raw Data

Place your documents in the appropriate directories:

```
data/raw/
├── invoices/
│   ├── invoice_001.pdf
│   ├── invoice_002.pdf
│   └── ...
└── purchase_orders/
    ├── po_001.pdf
    ├── po_002.pdf
    └── ...
```

## Step 2: Convert PDFs to Images (if needed)

If you have PDFs, convert them to images first:

```python
from pdf2image import convert_from_path

images = convert_from_path('invoice.pdf', dpi=300)
for i, image in enumerate(images):
    image.save(f'invoice_page_{i+1}.png', 'PNG')
```

## Step 3: Run OCR Extraction

```bash
python preprocessing/ocr_processor.py \
    --input data/raw/invoices \
    --output data/ocr_results/invoices \
    --config configs/data_config.yaml
```

This will:
- Preprocess images (deskew, denoise, binarize)
- Extract text with bounding boxes
- Calculate confidence scores
- Save results as JSON files

## Step 4: Annotation

### Option A: Using LabelStudio

1. Install LabelStudio:
```bash
pip install label-studio
```

2. Start LabelStudio:
```bash
label-studio start
```

3. Create a new project with the following template:

```xml
<View>
  <Image name="image" value="$image"/>
  <Labels name="label" toName="image">
    <Label value="invoice_number" background="red"/>
    <Label value="date" background="green"/>
    <Label value="vendor_name" background="blue"/>
    <Label value="total_amount" background="orange"/>
    <Label value="line_item" background="purple"/>
  </Labels>
  <Rectangle name="bbox" toName="image" strokeWidth="3"/>
</View>
```

4. Export annotations in JSON format

### Option B: Manual Annotation Format

Create JSON files following this structure:

```json
{
  "image_path": "data/raw/invoices/invoice_001.png",
  "document_type": "invoice",
  "words": [
    {
      "text": "INVOICE",
      "bbox": [100, 50, 200, 80],
      "label": "O",
      "confidence": 0.98
    },
    {
      "text": "INV-12345",
      "bbox": [100, 100, 250, 130],
      "label": "B-INVOICE_NUMBER",
      "confidence": 0.95
    }
  ],
  "table_labels": [0, 0, 1, 1, 2, ...],
  "metadata": {
    "vendor": "ACME Corp",
    "template_id": "template_A"
  }
}
```

## Step 5: Label Schema

Use BIO tagging scheme:
- **B-** : Beginning of entity
- **I-** : Inside entity (continuation)
- **O** : Outside (not an entity)

Common entity types for invoices:
- DATE
- INVOICE_NUMBER
- PURCHASE_ORDER_NUMBER
- VENDOR_NAME
- VENDOR_ADDRESS
- CUSTOMER_NAME
- CUSTOMER_ADDRESS
- LINE_ITEM_DESCRIPTION
- LINE_ITEM_QUANTITY
- LINE_ITEM_PRICE
- SUBTOTAL
- TAX
- TOTAL_AMOUNT
- PAYMENT_TERMS

## Step 6: Quality Control

Validate your annotations:

```python
from utils import validate_annotations

labels = ['O', 'B-DATE', 'I-DATE', 'B-INVOICE_NUMBER', ...]

for annotation_file in annotation_files:
    is_valid, errors = validate_annotations(annotation_file, labels)
    
    if not is_valid:
        print(f"Errors in {annotation_file}:")
        for error in errors:
            print(f"  - {error}")
```

## Step 7: Data Splitting

Split your annotated data:

```python
from utils import split_dataset

split_dataset(
    data_dir='data/annotations',
    output_dir='data/processed',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    stratify_key='document_type',  # Ensure balanced splits
    seed=42
)
```

## Step 8: Calculate Statistics

```python
from utils import calculate_dataset_statistics, print_dataset_statistics

for split in ['train', 'val', 'test']:
    stats = calculate_dataset_statistics(f'data/processed/{split}')
    print(f"\n{split.upper()} SET:")
    print_dataset_statistics(stats)
```

## Best Practices

### Data Quality

1. **Diverse Templates**: Include invoices from multiple vendors with different layouts
2. **Varied Quality**: Include both high-quality scans and degraded documents
3. **Balanced Labels**: Ensure all entity types are well-represented
4. **Minimum Samples**: Aim for at least 100-200 annotated documents per document type

### Annotation Guidelines

1. **Consistency**: Use consistent labeling across all documents
2. **Complete Spans**: Always annotate complete entities (don't split dates, amounts)
3. **Bounding Boxes**: Ensure bounding boxes tightly fit the text
4. **Verification**: Have multiple annotators verify a subset for quality

### Data Augmentation

The training pipeline includes automatic augmentation:
- Image augmentations (brightness, blur, rotation)
- Text augmentations (OCR error simulation)
- Enable in `configs/training_config.yaml`

## Troubleshooting

### Poor OCR Quality

If OCR quality is low:
1. Increase image DPI (300+ recommended)
2. Adjust preprocessing in `configs/data_config.yaml`
3. Try different OCR engines (Azure, Google Cloud Vision)

### Imbalanced Labels

If some entities are rare:
1. Use class weights in training config
2. Apply oversampling to minority classes
3. Collect more samples with rare entities

### Inconsistent Annotations

Run validation checks:
```bash
python -c "
from pathlib import Path
from utils import validate_annotations

labels = ['O', 'B-DATE', ...]  # Your label schema

errors_found = []
for file in Path('data/annotations').glob('*.json'):
    is_valid, errors = validate_annotations(file, labels)
    if not is_valid:
        errors_found.append((file, errors))

print(f'Found {len(errors_found)} files with errors')
"
```

## Next Steps

After data preparation:
1. Review `configs/training_config.yaml`
2. Start training: `python training/trainer.py`
3. Monitor progress with TensorBoard
4. Evaluate on test set
5. Iterate based on results
