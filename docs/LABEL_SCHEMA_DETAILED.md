# Production-Grade Label Schema Documentation

## Overview

This document describes the **3-layer production label architecture** for enterprise-grade invoice and purchase order extraction. The schema contains **95 labels** (47 entity types × 2 BIO tags + O).

## Design Philosophy

The schema is designed to:
1. **Extract all critical business data** from invoices/POs
2. **Understand document structure** (tables, blocks, noise)
3. **Group multi-token entities** correctly (addresses, line items)
4. **Suppress irrelevant content** (headers, footers, watermarks)
5. **Support multi-task learning** (NER + table detection + row grouping)

## Three-Layer Architecture

```
┌─────────────────────────────────────────┐
│  LAYER 1: GLOBAL HEADER FIELDS         │
│  (Single-value document-level)          │
│  - Invoice metadata                     │
│  - Vendor information                   │
│  - Buyer information                    │
│  - Financial totals                     │
│  - Payment details                      │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  LAYER 2: LINE-ITEM FIELDS              │
│  (Repeatable per table row)             │
│  - Product description                  │
│  - SKU/Item code                        │
│  - Quantity & UOM                       │
│  - Pricing (unit & total)               │
│  - Tax & discount per item              │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│  LAYER 3: STRUCTURAL LABELS             │
│  (Layout understanding & grouping)      │
│  - Table boundaries                     │
│  - Row grouping                         │
│  - Address blocks                       │
│  - Noise suppression                    │
└─────────────────────────────────────────┘
```

## Label Inventory (95 Total)

### LAYER 1: Global Header Fields (68 labels)

#### Invoice/PO Identifiers (14 labels)
| Label | Description | Example |
|-------|-------------|---------|
| `INVOICE_NUMBER` | Invoice or document number | INV-2025-001234 |
| `PO_NUMBER` | Purchase order reference | PO-98765 |
| `INVOICE_TYPE` | Document type | Invoice, Credit Memo, Pro Forma |

#### Dates (12 labels)
| Label | Description | Example |
|-------|-------------|---------|
| `INVOICE_DATE` | Issue date | 2025-11-26 |
| `DUE_DATE` | Payment due date | 2025-12-26 |
| `DELIVERY_DATE` | Expected delivery | 2025-12-01 |

#### Vendor Information (28 labels)
| Label | Description | Example |
|-------|-------------|---------|
| `VENDOR_NAME` | Supplier/vendor name | Acme Supplies Inc. |
| `VENDOR_ADDRESS` | Street address | 123 Main St, Suite 100 |
| `VENDOR_CITY` | City | San Francisco |
| `VENDOR_COUNTRY` | Country | USA |
| `VENDOR_POSTAL` | Zip/postal code | 94102 |
| `VENDOR_TAX_ID` | Tax ID/VAT number | VAT-123456789 |
| `VENDOR_PHONE` | Contact phone | +1-555-0100 |
| `VENDOR_EMAIL` | Contact email | sales@acme.com |

#### Buyer Information (16 labels)
| Label | Description | Example |
|-------|-------------|---------|
| `BUYER_NAME` | Customer/buyer name | XYZ Corporation |
| `BUYER_ADDRESS` | Bill-to address | 456 Oak Ave |
| `BUYER_CITY` | City | New York |
| `BUYER_COUNTRY` | Country | USA |
| `BUYER_POSTAL` | Zip/postal code | 10001 |
| `BUYER_TAX_ID` | Customer tax ID | EIN-987654321 |

#### Ship-To Information (8 labels)
| Label | Description | Example |
|-------|-------------|---------|
| `SHIP_TO_NAME` | Shipping recipient | XYZ Warehouse |
| `SHIP_TO_ADDRESS` | Shipping address | 789 Industrial Blvd |

#### Financial Totals (22 labels)
| Label | Description | Example |
|-------|-------------|---------|
| `SUBTOTAL` | Pre-tax total | 1,000.00 |
| `TAX_AMOUNT` | Total tax | 80.00 |
| `TAX_RATE` | Tax percentage | 8% |
| `SHIPPING_COST` | Shipping fee | 25.00 |
| `DISCOUNT_AMOUNT` | Total discount | -50.00 |
| `TOTAL_AMOUNT` | Final amount due | 1,055.00 |
| `CURRENCY` | Currency code | USD, EUR, GBP |

#### Payment Information (20 labels)
| Label | Description | Example |
|-------|-------------|---------|
| `PAYMENT_TERMS` | Payment terms | Net 30, Due on Receipt |
| `PAYMENT_METHOD` | Payment method | Bank Transfer, Credit Card |
| `BANK_ACCOUNT` | Bank account number | 1234567890 |
| `IBAN` | IBAN code | GB29 NWBK 6016 1331 9268 19 |
| `SWIFT_CODE` | SWIFT/BIC code | CHASUS33 |

#### References (12 labels)
| Label | Description | Example |
|-------|-------------|---------|
| `ORDER_REFERENCE` | Related order number | ORD-55555 |
| `CUSTOMER_ID` | Customer account ID | CUST-12345 |
| `CONTRACT_NUMBER` | Contract reference | CNT-2025-001 |

---

### LAYER 2: Line-Item Fields (18 labels)

These labels appear **repeatedly** in table rows. Each row represents a purchased item.

| Label | Description | Example | Required? |
|-------|-------------|---------|-----------|
| `ITEM_DESCRIPTION` | Product name/description | Blue Widget 10-pack | ✅ Yes |
| `ITEM_SKU` | Product SKU/code | WDG-BLU-10 | ✅ Yes |
| `ITEM_QUANTITY` | Quantity ordered | 5 | ✅ Yes |
| `ITEM_UOM` | Unit of measure | each, box, kg | Optional |
| `ITEM_UNIT_PRICE` | Price per unit | 10.00 | ✅ Yes |
| `ITEM_TOTAL_AMOUNT` | Line total (qty × price) | 50.00 | ✅ Yes |
| `ITEM_TAX` | Tax for this line | 4.00 | Optional |
| `ITEM_DISCOUNT` | Discount for this line | -2.00 | Optional |
| `ITEM_CODE` | Alternative product code | UPC-123456 | Optional |

**Example Table Row:**
```
| SKU        | Description        | Qty | Unit Price | Total   |
|------------|--------------------|-----|------------|---------|
| WDG-BLU-10 | Blue Widget 10-pack| 5   | $10.00     | $50.00  |
   ↑              ↑                 ↑       ↑           ↑
ITEM_SKU    ITEM_DESCRIPTION   ITEM_QTY  ITEM_UNIT   ITEM_TOTAL
```

---

### LAYER 3: Structural Labels (18 labels)

These labels help the model **understand layout** and **group tokens** correctly.

#### Table Structure (12 labels)
| Label | Purpose | Usage |
|-------|---------|-------|
| `TABLE` | Table boundary | Marks entire table region |
| `TABLE_HEADER` | Column headers | "SKU", "Description", "Qty", "Price" |
| `LINE_ITEM_ROW` | Data row | Entire row of item data |

**Why it matters:**
- Without `TABLE`, model can't distinguish table from text
- Without `TABLE_HEADER`, model confuses headers with data
- Without `LINE_ITEM_ROW`, line items aren't grouped correctly

#### Address Blocks (8 labels)
| Label | Purpose | Usage |
|-------|---------|-------|
| `ADDRESS_BLOCK_VENDOR` | Vendor address grouping | Groups multi-line vendor address |
| `ADDRESS_BLOCK_BUYER` | Buyer address grouping | Groups multi-line buyer address |

**Why it matters:**
- Addresses often span 3-5 lines
- Without block grouping, model extracts incomplete addresses
- Helps distinguish vendor vs buyer addresses in similar layouts

#### Special Sections (8 labels)
| Label | Purpose | Usage |
|-------|---------|-------|
| `PAYMENT_TERMS_BLOCK` | Payment terms section | Groups payment instructions |
| `NOTES_BLOCK` | Notes/terms section | Legal text, special instructions |

#### Noise Suppression (10 labels)
| Label | Purpose | Usage |
|-------|---------|-------|
| `PAGE_HEADER` | Header text | Company logos, "Invoice" title |
| `PAGE_FOOTER` | Footer text | Page numbers, disclaimers |
| `WATERMARK` | Watermark text | "COPY", "DRAFT", "PAID" |
| `NOISE` | Irrelevant text | Barcodes, artifacts, stamps |

**Why it matters:**
- Prevents model from extracting noise as data
- Improves precision by ignoring decorative elements
- Reduces false positives for amount extraction

---

## BIO Tagging Scheme

Each entity uses **BIO format**:
- `B-ENTITY`: **Begin** - First token of entity
- `I-ENTITY`: **Inside** - Continuation of entity
- `O`: **Outside** - Not part of any entity

### Examples

#### Example 1: Multi-word Vendor Name
```
Tokens:  ["Acme", "Supplies", "Inc."]
Labels:  [B-VENDOR_NAME, I-VENDOR_NAME, I-VENDOR_NAME]
```

#### Example 2: Address with Multiple Entities
```
Tokens:  ["123", "Main", "St", "San", "Francisco", "CA", "94102"]
Labels:  [B-VENDOR_ADDRESS, I-VENDOR_ADDRESS, I-VENDOR_ADDRESS,
          B-VENDOR_CITY, I-VENDOR_CITY, O, B-VENDOR_POSTAL]
```

#### Example 3: Table Row
```
Tokens:  ["WDG-100", "Widget", "5", "$10.00", "$50.00"]
Labels:  [B-ITEM_SKU, B-ITEM_DESCRIPTION, B-ITEM_QUANTITY,
          B-ITEM_UNIT_PRICE, B-ITEM_TOTAL_AMOUNT]
```

#### Example 4: Structural Labels
```
Tokens:  ["SKU", "Description", "Qty", "Price"]
Labels:  [B-TABLE_HEADER, I-TABLE_HEADER, I-TABLE_HEADER, I-TABLE_HEADER]
```

---

## Label Hierarchy & Relationships

### Nested Structures

Some labels work together hierarchically:

```
ADDRESS_BLOCK_VENDOR (outer)
  ├── VENDOR_NAME
  ├── VENDOR_ADDRESS
  ├── VENDOR_CITY
  └── VENDOR_POSTAL

TABLE (outer)
  ├── TABLE_HEADER
  └── LINE_ITEM_ROW (repeatable)
      ├── ITEM_SKU
      ├── ITEM_DESCRIPTION
      ├── ITEM_QUANTITY
      ├── ITEM_UNIT_PRICE
      └── ITEM_TOTAL_AMOUNT
```

### Mutual Exclusivity Rules

- A token cannot be both `VENDOR_ADDRESS` and `BUYER_ADDRESS`
- A token cannot be both `TABLE_HEADER` and `LINE_ITEM_ROW`
- A token cannot be both semantic (e.g., `INVOICE_NUMBER`) and noise (e.g., `PAGE_HEADER`)

---

## Annotation Guidelines

### Priority Order

When multiple labels could apply, use this priority:

1. **Structural labels** (TABLE, TABLE_HEADER, LINE_ITEM_ROW) - annotate first
2. **Semantic labels** (VENDOR_NAME, ITEM_SKU, etc.) - annotate within structure
3. **Noise labels** (PAGE_HEADER, NOISE) - annotate remaining

### Edge Cases

#### Case 1: Total appears in table AND header
```
In table: Label as ITEM_TOTAL_AMOUNT
In header: Label as TOTAL_AMOUNT
```

#### Case 2: Same text in multiple locations
```
"Acme Supplies" in header → VENDOR_NAME
"Acme Supplies" in footer → PAGE_FOOTER or NOISE
```

#### Case 3: Multi-page documents
```
Page 1 header: PAGE_HEADER
Page 2+ headers: PAGE_HEADER (repeat)
```

---

## Training Strategies

### Class Imbalance

- `O` (outside) is most frequent (~70% of tokens)
- Header fields appear once per document
- Line items repeat 5-50 times
- Structural labels are sparse

**Solution:**
- Use class weights in loss function
- Oversample rare entities
- Use CRF layer to capture dependencies

### Multi-Task Learning

Train simultaneously:
1. **NER head**: Token classification (95 classes)
2. **Table head**: Cell detection (binary: in-table / not-in-table)
3. **Column head**: Column classification (0-15 or -1)
4. **Row grouping**: CRF for sequence dependencies

---

## Expected Performance

### Target Metrics (Production-Ready)

| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| Overall Token F1 | > 0.85 | > 0.75 |
| Header Fields F1 | > 0.90 | > 0.80 |
| Line Items F1 | > 0.80 | > 0.70 |
| Structural Labels F1 | > 0.75 | > 0.65 |
| Complete Document Extraction | > 85% | > 70% |

### Per-Entity Targets

**Critical fields (must be accurate):**
- `INVOICE_NUMBER`: F1 > 0.95
- `TOTAL_AMOUNT`: F1 > 0.95
- `VENDOR_NAME`: F1 > 0.90
- `ITEM_SKU`: F1 > 0.85

**Medium priority:**
- Addresses: F1 > 0.80
- Dates: F1 > 0.85
- Line item fields: F1 > 0.80

**Lower priority (optional fields):**
- Payment details: F1 > 0.70
- Structural labels: F1 > 0.70

---

## Comparison to Existing Systems

| Feature | Our Schema | SAP | Oracle | Coupa |
|---------|-----------|-----|--------|-------|
| Total Labels | 95 | ~60 | ~75 | ~50 |
| Structural Labels | ✅ Yes | ❌ No | ⚠️ Limited | ❌ No |
| Multi-task Training | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Address Grouping | ✅ Yes | ⚠️ Limited | ⚠️ Limited | ❌ No |
| Noise Suppression | ✅ Yes | ❌ No | ❌ No | ❌ No |
| Line Item Rows | ✅ Explicit | ⚠️ Implicit | ⚠️ Implicit | ⚠️ Implicit |

---

## Next Steps

1. **Annotation**: Use this schema to label 500-1000 documents
2. **Validation**: Check label consistency and completeness
3. **Training**: Fine-tune LayoutLMv3 with multi-task objectives
4. **Evaluation**: Measure per-entity F1 and complete extraction rate
5. **Iteration**: Identify failure modes and add more training data

---

## References

- LayoutLMv3: https://arxiv.org/abs/2204.08387
- BIO Tagging: https://en.wikipedia.org/wiki/Inside–outside–beginning_(tagging)
- SROIE Dataset: https://rrc.cvc.uab.es/?ch=13
- CORD Dataset: https://github.com/clovaai/cord
