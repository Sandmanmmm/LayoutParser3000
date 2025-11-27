"""
OCR Processing Module
Handles OCR extraction from invoice and purchase order documents with quality enhancement.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import cv2
import numpy as np
import pytesseract
from PIL import Image
import yaml
import logging

logger = logging.getLogger(__name__)


class OCRProcessor:
    """High-quality OCR processing for document images."""
    
    def __init__(self, config_path: str = "./configs/data_config.yaml"):
        """Initialize OCR processor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['ocr']
        
        self.engine = self.config['engine']
        self.tesseract_config = self.config.get('tesseract', {})
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image quality before OCR.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        preprocessing = self.config.get('preprocessing', {})
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Deskew if enabled
        if preprocessing.get('deskew', True):
            gray = self._deskew(gray)
        
        # Denoise if enabled
        if preprocessing.get('denoise', True):
            gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Binarization
        binarization_method = preprocessing.get('binarization', 'adaptive')
        if binarization_method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
        elif binarization_method == 'otsu':
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            binary = gray
        
        return binary
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct skew in document image."""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Only deskew if angle is significant
        if abs(angle) < 0.5:
            return image
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def extract_text_tesseract(
        self, 
        image: np.ndarray
    ) -> Dict[str, any]:
        """
        Extract text using Tesseract OCR.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with text, bounding boxes, and confidence scores
        """
        # Preprocess image
        processed = self.preprocess_image(image)
        
        # Get detailed OCR data
        ocr_config = self.tesseract_config.get('config', '--oem 3 --psm 6')
        data = pytesseract.image_to_data(
            processed,
            lang=self.tesseract_config.get('lang', 'eng'),
            config=ocr_config,
            output_type=pytesseract.Output.DICT
        )
        
        # Parse results
        words = []
        quality_threshold = self.config.get('quality_threshold', 0.7)
        min_conf = self.config.get('min_word_confidence', 0.5)
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            if text and conf > min_conf * 100:
                word_data = {
                    'text': text,
                    'bbox': [
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    ],
                    'confidence': conf / 100.0,
                    'block_num': data['block_num'][i],
                    'line_num': data['line_num'][i],
                    'word_num': data['word_num'][i]
                }
                words.append(word_data)
        
        # Calculate overall quality
        avg_confidence = np.mean([w['confidence'] for w in words]) if words else 0.0
        
        return {
            'words': words,
            'full_text': ' '.join([w['text'] for w in words]),
            'average_confidence': avg_confidence,
            'quality_pass': avg_confidence >= quality_threshold
        }
    
    def process_document(
        self, 
        image_path: Path
    ) -> Optional[Dict[str, any]]:
        """
        Process a single document image.
        
        Args:
            image_path: Path to document image
            
        Returns:
            OCR results or None if processing fails
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Run OCR
            if self.engine == 'tesseract':
                result = self.extract_text_tesseract(image)
            else:
                raise NotImplementedError(f"OCR engine '{self.engine}' not implemented")
            
            # Add metadata
            result['image_path'] = str(image_path)
            result['image_size'] = image.shape[:2]
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None


def batch_process_documents(
    input_dir: Path,
    output_dir: Path,
    config_path: str = "./configs/data_config.yaml"
) -> None:
    """
    Process multiple documents in batch.
    
    Args:
        input_dir: Directory containing document images
        output_dir: Directory to save OCR results
        config_path: Path to configuration file
    """
    import json
    
    processor = OCRProcessor(config_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported image formats
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'**/*{ext}'))
    
    logger.info(f"Found {len(image_files)} images to process")
    
    for img_path in image_files:
        logger.info(f"Processing {img_path.name}")
        result = processor.process_document(img_path)
        
        if result and result.get('quality_pass', False):
            # Save OCR result
            output_file = output_dir / f"{img_path.stem}_ocr.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved result to {output_file}")
        else:
            logger.warning(f"Low quality OCR for {img_path.name}, skipping")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process documents with OCR")
    parser.add_argument("--input", type=str, required=True, help="Input directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--config", type=str, default="./configs/data_config.yaml")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    batch_process_documents(
        Path(args.input),
        Path(args.output),
        args.config
    )
