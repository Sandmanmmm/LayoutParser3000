# LayoutLMv3 Fine-tuning Project - Copilot Instructions

## Project Overview
This workspace is designed for fine-tuning LayoutLMv3 variants on invoice and purchase order documents with token classification and table structure detection capabilities.

## Key Technologies
- **Model**: LayoutLMv3 (Microsoft's multimodal document understanding transformer)
- **Framework**: PyTorch, Hugging Face Transformers
- **Tasks**: Token Classification (NER), Table Structure Detection
- **Training**: Mixed precision (fp16), gradient accumulation, learning rate scheduling, CRF layers

## Code Style Guidelines
- Use type hints for all function parameters and returns
- Follow PEP 8 style guidelines
- Document all classes and functions with docstrings
- Keep functions focused and modular
- Use pathlib for file path operations

## Project-Specific Patterns
- Config files use YAML format for hyperparameters
- Data augmentation pipelines use albumentations
- Model checkpoints saved with full training state
- Logging via tensorboard and wandb integration
- Evaluation metrics: Precision, Recall, F1 for NER; IoU for tables

## Directory Structure
- `data/`: Raw and processed datasets, annotations
- `models/`: Model architectures and checkpoints
- `configs/`: Training and model configuration files
- `training/`: Training loops and schedulers
- `evaluation/`: Evaluation scripts and metrics
- `preprocessing/`: OCR, augmentation, data prep scripts
- `utils/`: Helper functions and utilities
- `scripts/`: End-to-end pipeline scripts

## Development Workflow
1. Prepare data with OCR and augmentation
2. Configure training parameters in YAML
3. Train with token classification + table heads
4. Evaluate on diverse holdout sets
5. Iterate based on performance metrics
