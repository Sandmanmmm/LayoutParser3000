"""
Quick Start Script
Validates environment and creates a sample training run.
"""

import sys
from pathlib import Path
import subprocess

print("="*80)
print("LayoutLMv3 Invoice/PO Fine-tuning - Environment Check")
print("="*80)

# Check Python version
print("\n1. Checking Python version...")
if sys.version_info < (3, 8):
    print("❌ Python 3.8+ required. Current:", sys.version)
    sys.exit(1)
print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

# Check PyTorch
print("\n2. Checking PyTorch...")
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  CUDA not available (CPU training will be slow)")
except ImportError:
    print("❌ PyTorch not installed")
    print("   Install: pip install torch torchvision")
    sys.exit(1)

# Check Transformers
print("\n3. Checking Transformers...")
try:
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
except ImportError:
    print("❌ Transformers not installed")
    print("   Install: pip install transformers")
    sys.exit(1)

# Check other key dependencies
print("\n4. Checking other dependencies...")
dependencies = {
    'PIL': 'Pillow',
    'cv2': 'opencv-python',
    'yaml': 'pyyaml',
    'albumentations': 'albumentations',
    'numpy': 'numpy',
    'pandas': 'pandas',
}

missing = []
for module, package in dependencies.items():
    try:
        __import__(module)
        print(f"✅ {package}")
    except ImportError:
        print(f"❌ {package} not installed")
        missing.append(package)

if missing:
    print(f"\nInstall missing packages:")
    print(f"pip install {' '.join(missing)}")
    sys.exit(1)

# Check Tesseract
print("\n5. Checking Tesseract OCR...")
try:
    result = subprocess.run(
        ['tesseract', '--version'],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        version = result.stdout.split('\n')[0]
        print(f"✅ {version}")
    else:
        print("⚠️  Tesseract found but version check failed")
except FileNotFoundError:
    print("❌ Tesseract not installed")
    print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
    print("   macOS: brew install tesseract")
    print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")

# Check directory structure
print("\n6. Checking directory structure...")
required_dirs = [
    'configs',
    'data',
    'models',
    'preprocessing',
    'training',
    'evaluation',
    'utils',
    'scripts',
    'logs'
]

for dir_name in required_dirs:
    dir_path = Path(dir_name)
    if dir_path.exists():
        print(f"✅ {dir_name}/")
    else:
        print(f"⚠️  {dir_name}/ not found (will be created)")

# Check config files
print("\n7. Checking configuration files...")
config_files = [
    'configs/training_config.yaml',
    'configs/data_config.yaml'
]

for config_file in config_files:
    if Path(config_file).exists():
        print(f"✅ {config_file}")
    else:
        print(f"❌ {config_file} not found")

print("\n" + "="*80)
print("Environment Check Complete!")
print("="*80)

print("\n📚 Next Steps:")
print("\n1. Prepare your data:")
print("   - Place documents in data/raw/")
print("   - Run OCR: python preprocessing/ocr_processor.py")
print("   - Create annotations")
print("   - Split dataset: see docs/DATA_PREPARATION.md")

print("\n2. Configure training:")
print("   - Edit configs/training_config.yaml")
print("   - Adjust hyperparameters as needed")

print("\n3. Start training:")
print("   python training/trainer.py --config configs/training_config.yaml")

print("\n4. Monitor training:")
print("   tensorboard --logdir logs/tensorboard")

print("\n5. Evaluate model:")
print("   python evaluation/evaluate.py --model models/checkpoints/best_model.pt")

print("\n📖 Documentation:")
print("   - README.md: Overview and quick start")
print("   - docs/DATA_PREPARATION.md: Data preparation guide")
print("   - docs/TRAINING_GUIDE.md: Training guide")

print("\n💡 Tips:")
print("   - Start with a small dataset to validate pipeline")
print("   - Use TensorBoard to monitor training progress")
print("   - Enable W&B for advanced experiment tracking")
print("   - Check logs/ directory for detailed logs")

print("\n" + "="*80)
print("Happy Training! 🚀")
print("="*80 + "\n")
