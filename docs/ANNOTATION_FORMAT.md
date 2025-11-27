# Annotation Format Documentation

## Overview

This document describes the JSON annotation schema for training LayoutLMv3 on invoice and purchase order documents. The schema supports multi-task learning: token classification (NER) + table structure detection.

## Annotation File Format

Each document has a corresponding JSON annotation file with the following structure:

```json
{
  "metadata": {...},
  "ocr": [...],
  "ner_tags": [...],
  "tables": [...],
  "ground_truth": {...}
}
```

## Schema Sections

### 1. Metadata

Document-level information:

```json
{
  "metadata": {
    "file_name": "PO-1234.pdf",
    "merchant_id": "merchant_abc",
    "document_type": "purchase_order",  // or "invoice"
    "pages": 1,
    "image_width": 2480,
    "image_height": 3508,
    "annotator_id": "annotator_01",
    "annotation_date": "2025-11-26",
    "version": "1.0"
  }
}
```

### 2. OCR Tokens

Array of tokens extracted from the document with bounding boxes:

```json
{
  "ocr": [
    {
      "token_id": 0,
      "text": "Invoice",
      "bbox": [10, 10, 60, 30],  // [x0, y0, x1, y1] in pixels
      "page": 1,
      "confidence": 0.99
    },
    {
      "token_id": 1,
      "text": "No:",
      "bbox": [65, 10, 90, 30],
      "page": 1,
      "confidence": 0.98
    },
    {
      "token_id": 2,
      "text": "PO-1234",
      "bbox": [95, 10, 160, 30],
      "page": 1,
      "confidence": 0.99
    }
  ]
}
```

**Key Points:**
- Bounding boxes are in pixel coordinates (not normalized)
- `token_id` is sequential index (0-based)
- Each token represents a word or number

### 3. NER Tags

BIO labels for each token (same length as `ocr` array):

```json
{
  "ner_tags": [
    "O",
    "O",
    "B-DOCUMENT_NUMBER",
    "O",
    "B-DOCUMENT_DATE",
    "O",
    "B-SUPPLIER_NAME",
    "I-SUPPLIER_NAME",
    "I-SUPPLIER_NAME",
    "B-TOTAL_AMOUNT"
  ]
}
```

**Label Schema:**
- `O`: Outside any entity
- `B-<ENTITY>`: Beginning of entity
- `I-<ENTITY>`: Inside (continuation) of entity

**Supported Entities:**

**Document-level (single value):**
- SUPPLIER_NAME, SUPPLIER_ADDRESS
- BUYER_NAME, BUYER_ADDRESS
- DOCUMENT_NUMBER, DOCUMENT_DATE, DUE_DATE
- CURRENCY, SUBTOTAL, TAX, SHIPPING, TOTAL_AMOUNT
- PAYMENT_TERMS, ORDER_REFERENCE
- VENDOR_TAX_ID, ACCOUNT_NUMBER

**Line-item level (repeatable):**
- ITEM_DESCRIPTION, SKU, QUANTITY, UOM
- UNIT_PRICE, LINE_TOTAL, PACK_SIZE

### 4. Tables

Table structure with cells, rows, and columns:

```json
{
  "tables": [
    {
      "table_id": "t1",
      "bbox": [20, 120, 780, 420],
      "page": 1,
      "num_rows": 5,
      "num_cols": 6,
      "cells": [
        {
          "cell_id": "t1_r0_c0",
          "row": 0,
          "col": 0,
          "bbox": [22, 122, 200, 150],
          "text": "SKU",
          "token_ids": [12],
          "is_header": true
        },
        {
          "cell_id": "t1_r0_c1",
          "row": 0,
          "col": 1,
          "bbox": [202, 122, 500, 150],
          "text": "Description",
          "token_ids": [13],
          "is_header": true
        },
        {
          "cell_id": "t1_r1_c0",
          "row": 1,
          "col": 0,
          "bbox": [22, 152, 200, 190],
          "text": "ABC-100",
          "token_ids": [20],
          "is_header": false
        },
        {
          "cell_id": "t1_r1_c1",
          "row": 1,
          "col": 1,
          "bbox": [202, 152, 500, 190],
          "text": "Widget 10 pack",
          "token_ids": [21, 22, 23],
          "is_header": false
        }
      ]
    }
  ]
}
```

**Key Points:**
- Multiple tables can exist per document
- Cells reference tokens via `token_ids` array
- Row/col indices are 0-based
- `is_header` distinguishes header rows from data rows

### 5. Ground Truth

High-level extracted fields for evaluation:

```json
{
  "ground_truth": {
    "DOCUMENT_NUMBER": "PO-1234",
    "SUPPLIER_NAME": "Acme Supplies Inc.",
    "DOCUMENT_DATE": "2025-11-01",
    "TOTAL_AMOUNT": "1234.56",
    "CURRENCY": "USD",
    "line_items": [
      {
        "SKU": "ABC-100",
        "ITEM_DESCRIPTION": "Widget 10 pack (blue)",
        "QUANTITY": "5",
        "UNIT_PRICE": "10.00",
        "LINE_TOTAL": "50.00"
      },
      {
        "SKU": "XYZ-200",
        "ITEM_DESCRIPTION": "Gadget X",
        "QUANTITY": "2",
        "UNIT_PRICE": "200.00",
        "LINE_TOTAL": "400.00"
      }
    ]
  }
}
```

## Complete Example

```json
{
  "metadata": {
    "file_name": "PO-1234.pdf",
    "merchant_id": "merchant_abc",
    "document_type": "purchase_order",
    "pages": 1,
    "image_width": 2480,
    "image_height": 3508
  },
  "ocr": [
    {"token_id": 0, "text": "Purchase", "bbox": [50, 50, 150, 80], "page": 1, "confidence": 0.99},
    {"token_id": 1, "text": "Order", "bbox": [155, 50, 220, 80], "page": 1, "confidence": 0.99},
    {"token_id": 2, "text": "#", "bbox": [225, 50, 240, 80], "page": 1, "confidence": 0.95},
    {"token_id": 3, "text": "PO-1234", "bbox": [245, 50, 330, 80], "page": 1, "confidence": 0.99},
    {"token_id": 4, "text": "Date:", "bbox": [50, 90, 100, 110], "page": 1, "confidence": 0.98},
    {"token_id": 5, "text": "2025-11-01", "bbox": [105, 90, 200, 110], "page": 1, "confidence": 0.99},
    {"token_id": 6, "text": "Supplier:", "bbox": [50, 130, 130, 150], "page": 1, "confidence": 0.98},
    {"token_id": 7, "text": "Acme", "bbox": [135, 130, 190, 150], "page": 1, "confidence": 0.99},
    {"token_id": 8, "text": "Supplies", "bbox": [195, 130, 270, 150], "page": 1, "confidence": 0.99},
    {"token_id": 9, "text": "Inc.", "bbox": [275, 130, 310, 150], "page": 1, "confidence": 0.97},
    {"token_id": 10, "text": "Total:", "bbox": [600, 400, 650, 420], "page": 1, "confidence": 0.98},
    {"token_id": 11, "text": "$1,234.56", "bbox": [655, 400, 730, 420], "page": 1, "confidence": 0.99}
  ],
  "ner_tags": [
    "O", "O", "O", "B-DOCUMENT_NUMBER",
    "O", "B-DOCUMENT_DATE",
    "O", "B-SUPPLIER_NAME", "I-SUPPLIER_NAME", "I-SUPPLIER_NAME",
    "O", "B-TOTAL_AMOUNT"
  ],
  "tables": [
    {
      "table_id": "t1",
      "bbox": [50, 200, 750, 380],
      "page": 1,
      "num_rows": 3,
      "num_cols": 5,
      "cells": [
        {"cell_id": "t1_r0_c0", "row": 0, "col": 0, "bbox": [50, 200, 150, 230], "text": "SKU", "token_ids": [12], "is_header": true},
        {"cell_id": "t1_r0_c1", "row": 0, "col": 1, "bbox": [155, 200, 400, 230], "text": "Description", "token_ids": [13], "is_header": true},
        {"cell_id": "t1_r0_c2", "row": 0, "col": 2, "bbox": [405, 200, 500, 230], "text": "Qty", "token_ids": [14], "is_header": true},
        {"cell_id": "t1_r0_c3", "row": 0, "col": 3, "bbox": [505, 200, 620, 230], "text": "Unit Price", "token_ids": [15, 16], "is_header": true},
        {"cell_id": "t1_r0_c4", "row": 0, "col": 4, "bbox": [625, 200, 745, 230], "text": "Total", "token_ids": [17], "is_header": true}
      ]
    }
  ],
  "ground_truth": {
    "DOCUMENT_NUMBER": "PO-1234",
    "DOCUMENT_DATE": "2025-11-01",
    "SUPPLIER_NAME": "Acme Supplies Inc.",
    "TOTAL_AMOUNT": "1234.56",
    "CURRENCY": "USD"
  }
}
```

## Token-to-Cell Alignment

For training the table structure head, we need to derive:

1. **Cell labels** (per token): Is this token inside a table cell? (binary: 0/1)
2. **Column labels** (per token): Which column does this token belong to? (0-15, or -1 for non-table)

### Deriving Cell Labels

```python
cell_labels = []
for token in ocr_tokens:
    token_id = token["token_id"]
    in_cell = False
    for table in tables:
        for cell in table["cells"]:
            if token_id in cell["token_ids"]:
                in_cell = True
                break
        if in_cell:
            break
    cell_labels.append(1 if in_cell else 0)
```

### Deriving Column Labels

```python
col_labels = []
for token in ocr_tokens:
    token_id = token["token_id"]
    col_idx = -1
    for table in tables:
        for cell in table["cells"]:
            if token_id in cell["token_ids"]:
                col_idx = cell["col"]
                break
        if col_idx != -1:
            break
    col_labels.append(col_idx)
```

## File Naming Convention

```
data/
  annotations/
    PO-1234.json
    INV-5678.json
    PO-9999.json
  raw/
    invoices/
      PO-1234.pdf
      INV-5678.pdf
    purchase_orders/
      PO-9999.pdf
  processed/
    train.jsonl
    val.jsonl
    test.jsonl
```

## JSONL Format for Training

The `train.jsonl`, `val.jsonl`, and `test.jsonl` files contain one JSON object per line:

```jsonl
{"metadata": {...}, "ocr": [...], "ner_tags": [...], "tables": [...], "ground_truth": {...}}
{"metadata": {...}, "ocr": [...], "ner_tags": [...], "tables": [...], "ground_truth": {...}}
```

## Annotation Tools

Recommended tools for creating these annotations:

1. **Label Studio**: https://labelstud.io/
   - Supports OCR import, bbox annotation, NER labeling
   - Export to custom JSON format

2. **CVAT**: https://www.cvat.ai/
   - Good for bbox annotation
   - Requires custom export script

3. **Custom Scripts**:
   - Use `preprocessing/ocr_processor.py` to generate initial OCR
   - Manually create JSON annotations
   - Use provided template in `docs/annotation_template.json`

## Validation Script

Use the validation script to check annotation quality:

```bash
python utils/validate_annotations.py --annotations data/annotations/ --verbose
```

This checks for:
- Consistent token_id sequences
- Matching lengths of `ocr` and `ner_tags`
- Valid BIO tag sequences
- Cell token_ids referencing valid tokens
- Overlapping bboxes
- Required ground_truth fields

## Quality Guidelines

1. **Accurate Bounding Boxes**: Tight boxes around text (no excessive padding)
2. **Consistent Tokenization**: Match OCR output exactly
3. **Valid BIO Sequences**: No I- without preceding B-
4. **Complete Tables**: All cells labeled with row/col indices
5. **Ground Truth**: Double-check extracted values against document
6. **Edge Cases**: Include challenging examples (rotated, low-quality scans, multi-page)

## Next Steps

After creating annotations:

1. Split into train/val/test sets (70/15/15)
2. Run validation: `python utils/validate_annotations.py`
3. Generate statistics: `python utils/data_utils.py stats --annotations data/annotations/`
4. Start training: `python training/trainer.py --config configs/training_config.yaml`
