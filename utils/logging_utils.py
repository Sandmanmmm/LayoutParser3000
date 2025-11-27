"""
Logging Utilities
Setup and configure logging for training and evaluation.
"""

import logging
from pathlib import Path
from datetime import datetime
import sys
from typing import Optional


def setup_logging(
    log_dir: str = "./logs",
    log_level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger
    """
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"training_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to {log_file}")
    
    return logger


def log_model_info(model, logger: Optional[logging.Logger] = None):
    """
    Log model architecture and parameter count.
    
    Args:
        model: PyTorch model
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("="*80)
    logger.info("MODEL INFORMATION")
    logger.info("="*80)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    logger.info("="*80)


def log_config(config: dict, logger: Optional[logging.Logger] = None):
    """
    Log configuration dictionary.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("="*80)
    logger.info("CONFIGURATION")
    logger.info("="*80)
    
    def log_dict(d, prefix=""):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info(f"{prefix}{key}:")
                log_dict(value, prefix + "  ")
            else:
                logger.info(f"{prefix}{key}: {value}")
    
    log_dict(config)
    logger.info("="*80)
