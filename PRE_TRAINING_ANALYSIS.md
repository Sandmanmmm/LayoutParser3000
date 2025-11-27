# Pre-Training Requirements Analysis

**Analysis Date**: November 26, 2025  
**Workspace**: Layoutparser3 (NEW)  
**Status**: Setup Complete, Not Ready for Training

---

## 🔍 Current State Analysis

### ✅ What's Already Done

1. **Complete Code Base** ✅
   - All Python modules implemented (18 files)
   - Model architecture with CRF layer
   - Training pipeline with FP16 support
   - Evaluation framework
   - Data processing utilities

2. **Configuration Files** ✅
   - `training_config.yaml` - Ready
   - `data_config.yaml` - Ready

3. **Documentation** ✅
   - Comprehensive guides
   - Code examples
   - Quick reference

4. **Project Structure** ✅
   - All directories created
   - Proper organization

---

## ❌ What's Missing (Blocking Training)

### 🔴 CRITICAL - Must Have Before Training

#### 1. **Python Dependencies Not Installed**
**Status**: ❌ BLOCKING  
**Priority**: CRITICAL

Missing packages:
```bash
transformers>=4.35.0          # REQUIRED - LayoutLMv3 model
datasets>=2.14.0              # REQUIRED - Data handling
layoutparser>=0.3.4           # REQUIRED - Document processing
detectron2>=0.6               # REQUIRED - Layout detection
pytesseract>=0.3.10           # REQUIRED - OCR
albumentations>=1.3.1         # REQUIRED - Augmentation
seqeval>=1.2.2                # REQUIRED - NER metrics
tensorboard>=2.14.0           # Monitoring
wandb>=0.15.0                 # Optional monitoring
```

**Action Required**:
```bash
pip install -r requirements.txt
```

**Estimated Time**: 5-15 minutes  
**Potential Issues**: 
- `detectron2` may need manual installation on Windows
- May need Visual Studio Build Tools for some packages

---

#### 2. **No Training Data**
**Status**: ❌ BLOCKING  
**Priority**: CRITICAL

Current state:
- `data/raw/invoices/` - Empty (only .gitkeep)
- `data/annotations/` - Empty
- `data/processed/train/` - Empty
- `data/processed/val/` - Empty
- `data/processed/test/` - Empty

**What's Needed**:

**Option A: Use Existing Dataset**
- Download public invoice dataset (CORD, SROIE, etc.)
- Typical size: 800-1,000 documents
- Already annotated

**Option B: Create Custom Dataset**
- Minimum: 100-200 annotated invoices
- Recommended: 500-1,000 documents
- Need diverse templates and vendors

**Annotation Format Required**:
```json
{
  "image_path": "data/raw/invoices/invoice_001.png",
  "words": [
    {
      "text": "INVOICE",
      "bbox": [100, 50, 200, 80],
      "label": "O"
    },
    {
      "text": "INV-12345",
      "bbox": [100, 100, 250, 130],
      "label": "B-INVOICE_NUMBER"
    }
  ]
}
```

**Action Required**:
1. Acquire invoice images (PDF or PNG/JPG)
2. Run OCR: `python preprocessing/ocr_processor.py`
3. Annotate with labels (use LabelStudio, CVAT, or manual)
4. Split dataset: train/val/test (70/15/15)

**Estimated Time**: 
- With existing dataset: 1-2 hours (download + format)
- Manual annotation: 1-3 weeks (100-500 documents)

---

#### 3. **Tesseract OCR Not Verified**
**Status**: ⚠️ WARNING  
**Priority**: HIGH

The setup check couldn't verify Tesseract installation.

**Action Required**:
```bash
# Check if installed
tesseract --version

# If not installed:
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH after installation
```

**Estimated Time**: 5-10 minutes

---

### 🟡 IMPORTANT - Should Have for Best Results

#### 4. **CUDA/GPU Support**
**Status**: ⚠️ CPU ONLY  
**Priority**: HIGH (for speed)

Current: PyTorch 2.1.0+cpu (CPU-only version)

**Impact**:
- CPU training will be **10-50x slower**
- 1000 samples, 30 epochs: 
  - GPU: 2-5 hours
  - CPU: 20-100+ hours

**Action Required** (if GPU available):
```bash
# Uninstall CPU version
pip uninstall torch torchvision

# Install CUDA version (check your CUDA version first)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Check CUDA availability**:
```bash
nvidia-smi  # Check if NVIDIA GPU present
```

---

#### 5. **Label Schema Configuration**
**Status**: ⚠️ NEEDS REVIEW  
**Priority**: MEDIUM

Current config has 13 labels (example):
```yaml
labels:
  token_classification:
    - "O"
    - "B-DATE"
    - "I-DATE"
    - "B-INVOICE_NUMBER"
    - "I-INVOICE_NUMBER"
    - "B-VENDOR_NAME"
    - "I-VENDOR_NAME"
    # ... etc
```

**Action Required**:
1. Review `configs/training_config.yaml`
2. Adjust labels to match your actual entity types
3. Update `num_labels` to match label count

**Estimated Time**: 10-15 minutes

---

### 🟢 OPTIONAL - Nice to Have

#### 6. **Experiment Tracking**
**Status**: ✅ CONFIGURED  
**Priority**: LOW

TensorBoard: Built-in ✅  
Weights & Biases: Optional

**Action** (if using W&B):
```bash
wandb login
# Update configs/training_config.yaml with your project name
```

---

## 📋 Pre-Training Checklist

### Immediate Prerequisites (Before Any Training)

- [ ] **Install all Python dependencies**
  ```bash
  pip install -r requirements.txt
  ```
  
- [ ] **Verify Tesseract OCR installed**
  ```bash
  tesseract --version
  ```

- [ ] **Acquire training data**
  - [ ] Get invoice/PO images (100+ documents)
  - [ ] Place in `data/raw/invoices/`
  
- [ ] **Process and annotate data**
  - [ ] Run OCR extraction
  - [ ] Create annotations in JSON format
  - [ ] Validate annotation format
  
- [ ] **Split dataset**
  - [ ] Create train/val/test splits
  - [ ] Verify splits in `data/processed/`

- [ ] **Configure labels**
  - [ ] Update label schema in config
  - [ ] Match num_labels to actual count

### Optional But Recommended

- [ ] **Setup GPU support** (if available)
  - [ ] Install CUDA PyTorch version
  - [ ] Verify GPU detection

- [ ] **Configure experiment tracking**
  - [ ] Setup TensorBoard
  - [ ] (Optional) Setup Weights & Biases

- [ ] **Test data pipeline**
  - [ ] Run with small subset first
  - [ ] Verify data loading works

---

## 🚀 Quick Start Path (Fastest to Training)

### Path 1: Use Public Dataset (2-3 hours)
1. Install dependencies (15 min)
2. Download CORD or SROIE dataset (30 min)
3. Convert to required format (1 hour)
4. Configure labels (15 min)
5. Start training ✅

### Path 2: Minimal Custom Dataset (1-2 days)
1. Install dependencies (15 min)
2. Collect 50 invoice samples (1 hour)
3. Run OCR (30 min)
4. Quick annotation of key fields only (3-4 hours)
5. Split dataset (5 min)
6. Start training ✅

### Path 3: Full Custom Dataset (2-4 weeks)
1. Install dependencies (15 min)
2. Collect 500+ invoice samples (varies)
3. Run OCR (1-2 hours)
4. Full annotation (1-3 weeks)
5. Split dataset (5 min)
6. Start training ✅

---

## 🛠️ Detailed Setup Steps

### Step 1: Install Dependencies (REQUIRED)

```bash
# Install all at once
pip install -r requirements.txt

# Or install in stages:
# Core (required)
pip install torch torchvision transformers datasets

# OCR (required)
pip install pytesseract pdf2image Pillow opencv-python

# Data processing (required)
pip install numpy pandas scikit-learn albumentations

# Evaluation (required)
pip install seqeval

# Monitoring (recommended)
pip install tensorboard tqdm

# Optional
pip install wandb
```

**Troubleshooting**:
- If `detectron2` fails on Windows, it can be skipped for basic training
- If `layoutparser` fails, continue without it (only needed for complex layouts)

---

### Step 2: Get Sample Data (REQUIRED)

**Quick Start Option - Use Public Dataset**:

```python
# Download CORD dataset (simplified)
from datasets import load_dataset

dataset = load_dataset("naver-clova-ix/cord-v2")
# Convert to our format and save to data/
```

**Or manually**:
1. Find 10-20 sample invoices (PDFs or images)
2. Place in `data/raw/invoices/`
3. Continue to Step 3

---

### Step 3: Process Data (REQUIRED)

```bash
# Run OCR on your invoices
python preprocessing/ocr_processor.py \
    --input data/raw/invoices \
    --output data/ocr_results \
    --config configs/data_config.yaml
```

---

### Step 4: Annotate (REQUIRED)

**Option A: Use LabelStudio** (recommended):
```bash
pip install label-studio
label-studio start
# Create project and import images
# Export as JSON
```

**Option B: Manual annotation**:
Edit JSON files with labels following the format shown above.

---

### Step 5: Split Dataset (REQUIRED)

```python
from utils import split_dataset
from pathlib import Path

split_dataset(
    data_dir=Path('data/annotations'),
    output_dir=Path('data/processed'),
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)
```

---

### Step 6: Verify Setup

```bash
# Run setup check
python scripts/setup_check.py

# Should show all green checkmarks
```

---

### Step 7: Start Training! 🎉

```bash
python training/trainer.py --config configs/training_config.yaml
```

---

## ⏱️ Time Estimates

| Task | Minimum Time | Typical Time | Notes |
|------|--------------|--------------|-------|
| Install dependencies | 5 min | 15 min | May have issues on Windows |
| Setup GPU (if available) | 5 min | 15 min | Requires CUDA installed |
| Download public dataset | 10 min | 30 min | CORD, SROIE, etc. |
| Convert dataset format | 30 min | 2 hours | Depends on dataset |
| Collect custom invoices | 1 hour | 1 day | Varies greatly |
| Run OCR | 5 min | 1 hour | Depends on doc count |
| Annotate (minimal) | 2 hours | 1 day | 50-100 docs, key fields |
| Annotate (full) | 1 week | 3 weeks | 500+ docs, all fields |
| Split dataset | 1 min | 5 min | Automated |
| **Total (public dataset)** | **1 hour** | **3 hours** | Fastest path |
| **Total (minimal custom)** | **1 day** | **2 days** | Basic training |
| **Total (full custom)** | **1 week** | **4 weeks** | Production quality |

---

## 🎯 Recommended Immediate Actions

**To start training TODAY** (with public data):

1. **Install dependencies** (15 min)
   ```bash
   pip install transformers datasets torch torchvision pytesseract seqeval tensorboard tqdm
   ```

2. **Download sample dataset** (30 min)
   - Use CORD or create 10 sample annotations manually

3. **Place in correct folders** (5 min)
   - `data/processed/train/` - 7 files
   - `data/processed/val/` - 2 files  
   - `data/processed/test/` - 1 file

4. **Configure labels** (10 min)
   - Match config to your data

5. **Start training** ✅
   ```bash
   python training/trainer.py --config configs/training_config.yaml
   ```

**Total time to first training run**: ~1 hour

---

## 🐛 Common Issues & Solutions

### Issue: Dependencies Won't Install
**Solution**: Install in stages, skip problematic packages like detectron2

### Issue: No GPU Detected
**Solution**: Install CUDA version of PyTorch or train on CPU (slower)

### Issue: OCR Quality Poor
**Solution**: Increase image DPI, use better OCR engine (Azure, Google)

### Issue: Out of Memory
**Solution**: Reduce batch_size to 2, increase gradient_accumulation_steps

---

## 📊 Current Workspace Status

```
✅ Code: Complete (100%)
✅ Configs: Complete (100%)
✅ Documentation: Complete (100%)
❌ Dependencies: Not installed (0%)
❌ Data: No data available (0%)
⚠️  GPU: CPU only (performance concern)
```

**Overall Readiness**: 40% (Code ready, but missing data and dependencies)

**Estimated Time to Training-Ready**: 
- With public dataset: 1-3 hours
- With custom minimal dataset: 1-2 days
- With custom full dataset: 2-4 weeks

---

## 🎓 Conclusion

**What you have**: A complete, professional LayoutLMv3 fine-tuning framework

**What you need**:
1. ✅ Install Python packages (15 minutes)
2. ✅ Get training data (1 hour - 2 weeks depending on approach)
3. ✅ Optionally setup GPU for faster training

**Recommended next step**: Follow "Quick Start Path 1" to use a public dataset and be training within 2-3 hours.
