"""
Quick Setup Script
Installs dependencies and prepares for training with minimal data.
"""

import subprocess
import sys
from pathlib import Path

print("="*80)
print("LayoutLMv3 Quick Setup - Preparing for Training")
print("="*80)

# Step 1: Install core dependencies
print("\n[1/4] Installing core dependencies...")
print("This may take 5-15 minutes...")

core_packages = [
    "transformers>=4.35.0",
    "datasets>=2.14.0",
    "pytesseract>=0.3.10",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "opencv-python>=4.8.0",
    "albumentations>=1.3.1",
    "seqeval>=1.2.2",
    "tensorboard>=2.14.0",
    "tqdm>=4.66.0",
    "pyyaml>=6.0",
    "layoutparser>=0.3.4"
]

try:
    print("Installing packages...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install"] + core_packages,
        check=True
    )
    print("✅ Core dependencies installed successfully!")
except subprocess.CalledProcessError:
    print("❌ Failed to install some packages")
    print("Try manually: pip install -r requirements.txt")
    sys.exit(1)

# Step 2: Check directory structure
print("\n[2/4] Checking directory structure...")
required_dirs = [
    "data/raw/invoices",
    "data/processed/train",
    "data/processed/val",
    "data/processed/test",
    "data/annotations",
    "models/checkpoints",
    "logs"
]

for dir_path in required_dirs:
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✅ {dir_path}/")

# Step 3: Verify installations
print("\n[3/4] Verifying installations...")

try:
    import transformers
    print(f"✅ transformers {transformers.__version__}")
except ImportError:
    print("❌ transformers not installed")

try:
    import torch
    print(f"✅ torch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CPU only (training will be slower)")
except ImportError:
    print("❌ torch not installed")

try:
    import albumentations
    print(f"✅ albumentations (for augmentation)")
except ImportError:
    print("⚠️  albumentations not installed (augmentation disabled)")

# Step 4: Next steps
print("\n[4/4] Setup Summary")
print("="*80)

data_count = len(list(Path("data/raw/invoices").glob("*.png"))) + \
             len(list(Path("data/raw/invoices").glob("*.jpg")))

if data_count == 0:
    print("\n⚠️  NO TRAINING DATA FOUND")
    print("\nYou need to add invoice images before training.")
    print("\nQuick start options:")
    print("\n1. Use public dataset (FASTEST - 1 hour):")
    print("   - Download CORD dataset: https://github.com/clovaai/cord")
    print("   - Or SROIE: https://rrc.cvc.uab.es/?ch=13")
    print("   - Convert to required format")
    
    print("\n2. Create minimal test dataset (2-3 hours):")
    print("   - Find 10-20 invoice PDFs or images")
    print("   - Place in data/raw/invoices/")
    print("   - Run: python preprocessing/ocr_processor.py --input data/raw/invoices --output data/ocr_results")
    print("   - Create simple annotations manually")
    print("   - Split: See docs/DATA_PREPARATION.md")
    
    print("\n3. Full production dataset (1-4 weeks):")
    print("   - See docs/DATA_PREPARATION.md for complete guide")
else:
    print(f"\n✅ Found {data_count} images in data/raw/invoices/")
    print("\nNext: Process with OCR and create annotations")

print("\n" + "="*80)
print("Dependencies Installation Complete!")
print("="*80)

print("\n📚 Next Steps:")
print("\n1. Add training data:")
print("   - Place invoices in data/raw/invoices/")
print("   - Or download public dataset")

print("\n2. Process data:")
print("   python preprocessing/ocr_processor.py --input data/raw/invoices --output data/ocr_results")

print("\n3. Create annotations:")
print("   - See docs/DATA_PREPARATION.md for annotation guide")
print("   - Use LabelStudio or manual JSON files")

print("\n4. Split dataset:")
print("   - See example in docs/EXAMPLE_WORKFLOW.md")

print("\n5. Start training:")
print("   python training/trainer.py --config configs/training_config.yaml")

print("\n📖 Full analysis: See PRE_TRAINING_ANALYSIS.md")
print("\n" + "="*80)
