#!/usr/bin/env python3
"""
Training Startup Script
=======================
Simplified script to start model training with proper configuration.

This script:
1. Loads production configuration
2. Validates data and labels
3. Initializes model and datasets
4. Starts training with monitoring

Usage:
    python scripts/start_training.py [--config CONFIG_PATH] [--resume CHECKPOINT]
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict
import yaml
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.layoutlmv3_model import LayoutLMv3ForMultiTask
from preprocessing import create_dataloaders
from training.trainer import Trainer


def setup_logging(log_dir: Path):
    """Setup logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_setup(config: Dict, base_path: Path, logger) -> bool:
    """Validate that everything is ready for training."""
    logger.info("Validating training setup...")
    
    issues = []
    
    # Check label file
    label_file = base_path / config['data']['label_list_path']
    if not label_file.exists():
        issues.append(f"Label file not found: {label_file}")
    else:
        with open(label_file, 'r') as f:
            labels = [line.strip() for line in f if line.strip()]
        if len(labels) != config['model']['num_labels']:
            issues.append(f"Label count mismatch: {len(labels)} in file vs {config['model']['num_labels']} in config")
    
    # Check data files
    for split in ['train', 'val']:
        data_file = base_path / config['data'][f'{split}_json']
        if not data_file.exists():
            # Try with different extension
            alt_file = data_file.with_suffix('.json')
            if not alt_file.exists():
                issues.append(f"Data file not found: {data_file}")
            else:
                logger.info(f"Using {alt_file} instead of {data_file}")
                config['data'][f'{split}_json'] = str(alt_file.relative_to(base_path))
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("⚠️  CUDA not available, training will use CPU (very slow)")
        if config['training'].get('fp16', False):
            logger.warning("   Disabling FP16 training (requires CUDA)")
            config['training']['fp16'] = False
    
    if issues:
        logger.error("❌ Validation failed with issues:")
        for issue in issues:
            logger.error(f"   - {issue}")
        return False
    
    logger.info("✅ Validation passed")
    return True


def initialize_model(config: Dict, device: str, logger) -> LayoutLMv3ForMultiTask:
    """Initialize the LayoutLMv3 model."""
    logger.info("Initializing model...")
    
    model_config = config['model']
    
    try:
        model = LayoutLMv3ForMultiTask(
            model_name_or_path=model_config['model_name_or_path'],
            num_labels=model_config['num_labels'],
            use_crf=model_config.get('use_crf', True),
            table_num_cols=model_config.get('table_num_cols', 16)
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"✅ Model initialized")
        logger.info(f"   Total parameters: {total_params:,}")
        logger.info(f"   Trainable parameters: {trainable_params:,}")
        logger.info(f"   Model size: {total_params * 4 / 1e6:.2f} MB")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize model: {e}")
        raise


def prepare_dataloaders(config: Dict, base_path: Path, logger):
    """Prepare training and validation dataloaders."""
    logger.info("Preparing dataloaders...")
    
    try:
        train_loader, val_loader = create_dataloaders(
            train_json=str(base_path / config['data']['train_json']),
            val_json=str(base_path / config['data']['val_json']),
            label_list_path=str(base_path / config['data']['label_list_path']),
            batch_size=config['training']['per_device_train_batch_size'],
            num_workers=config['training'].get('dataloader_num_workers', 4),
            apply_augmentation=config['preprocessing']['image_augmentations']['enabled']
        )
        
        logger.info(f"✅ Dataloaders ready")
        logger.info(f"   Training batches: {len(train_loader)}")
        logger.info(f"   Validation batches: {len(val_loader)}")
        
        return train_loader, val_loader
        
    except Exception as e:
        logger.error(f"❌ Failed to create dataloaders: {e}")
        raise


def print_training_info(config: Dict, logger):
    """Print training configuration information."""
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    
    training_cfg = config['training']
    
    logger.info(f"Experiment: {config.get('experiment_name', 'unnamed')}")
    logger.info(f"Model: {config['model']['model_name_or_path']}")
    logger.info(f"Labels: {config['model']['num_labels']}")
    logger.info(f"CRF: {config['model'].get('use_crf', False)}")
    logger.info("")
    
    logger.info(f"Epochs: {training_cfg['num_epochs']}")
    logger.info(f"Batch size: {training_cfg['per_device_train_batch_size']}")
    logger.info(f"Gradient accumulation: {training_cfg['gradient_accumulation_steps']}")
    effective_batch = training_cfg['per_device_train_batch_size'] * training_cfg['gradient_accumulation_steps']
    logger.info(f"Effective batch size: {effective_batch}")
    logger.info(f"Learning rate: {training_cfg['learning_rate']}")
    logger.info(f"Warmup ratio: {training_cfg.get('warmup_ratio', 0.1)}")
    logger.info(f"LR scheduler: {training_cfg.get('lr_scheduler_type', 'cosine')}")
    logger.info(f"Mixed precision (FP16): {training_cfg.get('fp16', False)}")
    logger.info("")
    
    loss_weights = config.get('loss_weights', {})
    logger.info(f"Loss weights:")
    logger.info(f"  NER: {loss_weights.get('ner_loss_weight', 1.0)}")
    logger.info(f"  Cell: {loss_weights.get('cell_loss_weight', 1.0)}")
    logger.info(f"  Column: {loss_weights.get('col_loss_weight', 0.5)}")
    logger.info("")
    
    logger.info(f"Output directory: {training_cfg['output_dir']}")
    logger.info(f"Best model metric: {training_cfg.get('metric_for_best_model', 'eval_loss')}")
    logger.info("=" * 80)


def main():
    """Main training execution."""
    parser = argparse.ArgumentParser(description='Start model training')
    parser.add_argument('--config', type=str,
                       default='configs/training_config_production.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to train on')
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = Path(__file__).parent.parent
    config_path = base_path / args.config
    
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup logging
    log_dir = base_path / 'logs'
    logger = setup_logging(log_dir)
    
    logger.info("=" * 80)
    logger.info("LAYOUTPARSER3000 - MODEL TRAINING")
    logger.info("=" * 80)
    logger.info("")
    
    # Validate setup
    if not validate_setup(config, base_path, logger):
        logger.error("❌ Setup validation failed. Please fix issues and try again.")
        sys.exit(1)
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info("")
    
    # Print training info
    print_training_info(config, logger)
    
    try:
        # Initialize model
        model = initialize_model(config, device, logger)
        
        # Prepare dataloaders
        train_loader, val_loader = prepare_dataloaders(config, base_path, logger)
        
        # Initialize trainer
        logger.info("Initializing trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        logger.info("✅ Trainer initialized")
        logger.info("")
        
        # Start training
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)
        logger.info("")
        
        trainer.train()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Best model saved to: {config['training']['output_dir']}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
