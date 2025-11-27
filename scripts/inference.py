"""
Inference Script
Run inference on new documents using trained model.
"""

import argparse
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))

from evaluation.evaluate import Evaluator
from preprocessing.ocr_processor import OCRProcessor
from utils.visualization import visualize_predictions
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_inference(
    model_path: str,
    image_path: str,
    config_path: str = "./configs/training_config.yaml",
    output_path: str = None,
    visualize: bool = True
):
    """
    Run inference on a single document.
    
    Args:
        model_path: Path to trained model
        image_path: Path to document image
        config_path: Path to configuration
        output_path: Path to save predictions JSON
        visualize: Whether to create visualization
    """
    logger.info(f"Loading model from {model_path}")
    
    # Initialize evaluator
    evaluator = Evaluator(
        Path(model_path),
        Path(config_path),
        device="cuda"
    )
    
    # Run OCR
    logger.info(f"Processing document: {image_path}")
    ocr_processor = OCRProcessor(config_path)
    ocr_results = ocr_processor.process_document(Path(image_path))
    
    if not ocr_results:
        logger.error("OCR processing failed")
        return
    
    logger.info(f"Extracted {len(ocr_results['words'])} words")
    
    # Run inference
    logger.info("Running inference...")
    predictions = evaluator.predict_single_document(
        Path(image_path),
        ocr_results
    )
    
    # Log entity counts
    from collections import Counter
    entity_counts = Counter()
    for pred in predictions['predictions']:
        label = pred['label']
        if label != 'O':
            entity_type = label.split('-')[1] if '-' in label else label
            entity_counts[entity_type] += 1
    
    logger.info("\nExtracted Entities:")
    for entity, count in entity_counts.most_common():
        logger.info(f"  {entity}: {count}")
    
    # Save predictions
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nPredictions saved to {output_file}")
    
    # Visualize
    if visualize:
        viz_path = Path(image_path).parent / f"{Path(image_path).stem}_predictions.png"
        visualize_predictions(
            Path(image_path),
            predictions['predictions'],
            save_path=viz_path,
            show=False
        )
        logger.info(f"Visualization saved to {viz_path}")
    
    return predictions


def batch_inference(
    model_path: str,
    images_dir: str,
    output_dir: str,
    config_path: str = "./configs/training_config.yaml"
):
    """
    Run inference on multiple documents.
    
    Args:
        model_path: Path to trained model
        images_dir: Directory containing images
        output_dir: Directory to save results
        config_path: Path to configuration
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
    images = []
    for ext in image_extensions:
        images.extend(images_dir.glob(f'*{ext}'))
    
    logger.info(f"Found {len(images)} images to process")
    
    # Process each image
    for i, image_path in enumerate(images, 1):
        logger.info(f"\n[{i}/{len(images)}] Processing {image_path.name}")
        
        output_file = output_dir / f"{image_path.stem}_predictions.json"
        
        try:
            run_inference(
                model_path,
                str(image_path),
                config_path,
                str(output_file),
                visualize=True
            )
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            continue
    
    logger.info(f"\nBatch inference complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on documents")
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to single image (for single inference)"
    )
    
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Directory of images (for batch inference)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./predictions",
        help="Output directory or file path"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/training_config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization"
    )
    
    args = parser.parse_args()
    
    if args.image:
        # Single inference
        run_inference(
            args.model,
            args.image,
            args.config,
            args.output,
            visualize=not args.no_viz
        )
    elif args.images_dir:
        # Batch inference
        batch_inference(
            args.model,
            args.images_dir,
            args.output,
            args.config
        )
    else:
        parser.error("Must provide either --image or --images-dir")


if __name__ == "__main__":
    main()
