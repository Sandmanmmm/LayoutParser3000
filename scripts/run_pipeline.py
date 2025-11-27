"""
End-to-End Training Pipeline
Run complete training pipeline from data preparation to evaluation.
"""

import argparse
import logging
from pathlib import Path
import yaml
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from preprocessing.ocr_processor import batch_process_documents
from utils.data_utils import split_dataset, calculate_dataset_statistics, print_dataset_statistics
from training.trainer import train_model
from evaluation.evaluate import evaluate_model
from utils.logging_utils import setup_logging


def run_pipeline(
    raw_data_dir: str,
    output_dir: str,
    config_path: str = "./configs/training_config.yaml",
    skip_ocr: bool = False,
    skip_split: bool = False,
    skip_train: bool = False,
    skip_eval: bool = False
):
    """
    Run end-to-end training pipeline.
    
    Args:
        raw_data_dir: Directory containing raw documents
        output_dir: Output directory for all artifacts
        config_path: Path to training configuration
        skip_ocr: Skip OCR processing
        skip_split: Skip data splitting
        skip_train: Skip training
        skip_eval: Skip evaluation
    """
    # Setup logging
    logger = setup_logging(log_dir=f"{output_dir}/logs")
    logger.info("="*80)
    logger.info("STARTING END-TO-END TRAINING PIPELINE")
    logger.info("="*80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: OCR Processing
    if not skip_ocr:
        logger.info("\n[Step 1/5] Running OCR processing...")
        ocr_output = output_path / "ocr_results"
        batch_process_documents(
            Path(raw_data_dir),
            ocr_output,
            config_path
        )
        logger.info(f"OCR processing complete. Results saved to {ocr_output}")
    else:
        logger.info("\n[Step 1/5] Skipping OCR processing")
    
    # Step 2: Data Splitting
    if not skip_split:
        logger.info("\n[Step 2/5] Splitting dataset...")
        annotations_dir = Path(config['data']['annotations_path'])
        processed_dir = Path(config['data']['train_data_path']).parent
        
        split_dataset(
            annotations_dir,
            processed_dir,
            train_ratio=config['data_split'].get('train_ratio', 0.7),
            val_ratio=config['data_split'].get('val_ratio', 0.15),
            test_ratio=config['data_split'].get('test_ratio', 0.15),
            stratify_key=config['data_split'].get('stratify_by'),
            seed=config.get('seed', 42)
        )
        
        # Calculate statistics
        for split in ['train', 'val', 'test']:
            split_dir = processed_dir / split
            if split_dir.exists():
                stats = calculate_dataset_statistics(split_dir)
                logger.info(f"\n{split.upper()} SET STATISTICS:")
                print_dataset_statistics(stats)
        
        logger.info("Data splitting complete")
    else:
        logger.info("\n[Step 2/5] Skipping data splitting")
    
    # Step 3: Training
    best_model_path = None
    if not skip_train:
        logger.info("\n[Step 3/5] Starting training...")
        train_model(config_path)
        best_model_path = Path(config['training']['output_dir']) / "best_model.pt"
        logger.info(f"Training complete. Best model saved to {best_model_path}")
    else:
        logger.info("\n[Step 3/5] Skipping training")
        # Try to find existing model
        checkpoint_dir = Path(config['training']['output_dir'])
        if (checkpoint_dir / "best_model.pt").exists():
            best_model_path = checkpoint_dir / "best_model.pt"
    
    # Step 4: Evaluation
    if not skip_eval:
        if best_model_path and best_model_path.exists():
            logger.info("\n[Step 4/5] Running evaluation...")
            eval_output = output_path / "logs" / "evaluation_results.json"
            evaluate_model(
                str(best_model_path),
                config_path,
                str(eval_output)
            )
            logger.info(f"Evaluation complete. Results saved to {eval_output}")
        else:
            logger.warning("No model found for evaluation. Skipping.")
    else:
        logger.info("\n[Step 4/5] Skipping evaluation")
    
    # Step 5: Summary
    logger.info("\n[Step 5/5] Pipeline Summary")
    logger.info("="*80)
    logger.info("Pipeline execution complete!")
    logger.info(f"All outputs saved to: {output_path}")
    logger.info("\nNext steps:")
    logger.info("1. Review training logs in logs/")
    logger.info("2. Check evaluation metrics")
    logger.info("3. Visualize predictions on test samples")
    logger.info("4. Fine-tune hyperparameters if needed")
    logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end LayoutLMv3 training pipeline"
    )
    
    parser.add_argument(
        "--raw-data",
        type=str,
        required=True,
        help="Directory containing raw documents"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory for all artifacts"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/training_config.yaml",
        help="Path to training configuration"
    )
    
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR processing step"
    )
    
    parser.add_argument(
        "--skip-split",
        action="store_true",
        help="Skip data splitting step"
    )
    
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training step"
    )
    
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation step"
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        args.raw_data,
        args.output,
        args.config,
        args.skip_ocr,
        args.skip_split,
        args.skip_train,
        args.skip_eval
    )


if __name__ == "__main__":
    main()
