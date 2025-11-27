"""
Data Augmentation Module
Implements image and text augmentations for invoice/PO documents.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import yaml
import logging

logger = logging.getLogger(__name__)


class DocumentAugmentation:
    """Augmentation pipeline for document images."""
    
    def __init__(self, config_path: str = "./configs/training_config.yaml"):
        """Initialize augmentation pipeline."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.aug_config = config.get('augmentation', {})
        self.enabled = self.aug_config.get('enabled', True)
        
        if self.enabled:
            self.image_transform = self._build_image_pipeline()
            self.text_config = self.aug_config.get('text_transforms', {})
    
    def _build_image_pipeline(self) -> A.Compose:
        """Build albumentations augmentation pipeline."""
        transforms = []
        
        for transform_config in self.aug_config.get('image_transforms', []):
            transform_name = transform_config['name']
            transform_params = {k: v for k, v in transform_config.items() if k != 'name'}
            
            # Map transform names to albumentations classes
            transform_map = {
                'RandomBrightnessContrast': A.RandomBrightnessContrast,
                'GaussianBlur': A.GaussianBlur,
                'GaussNoise': A.GaussNoise,
                'Rotate': A.Rotate,
                'RandomScale': A.RandomScale,
                'ShiftScaleRotate': A.ShiftScaleRotate,
                'ElasticTransform': A.ElasticTransform,
                'GridDistortion': A.GridDistortion,
            }
            
            if transform_name in transform_map:
                transforms.append(transform_map[transform_name](**transform_params))
        
        # Compose all transforms with bbox support
        return A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels'],
                min_visibility=0.3
            )
        )
    
    def augment_image(
        self,
        image: np.ndarray,
        bboxes: List[List[float]],
        labels: List[int]
    ) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Apply image augmentations.
        
        Args:
            image: Input image (H, W, C)
            bboxes: List of bounding boxes in pascal_voc format [x_min, y_min, x_max, y_max]
            labels: List of label indices for each bbox
            
        Returns:
            Augmented image, bboxes, and labels
        """
        if not self.enabled:
            return image, bboxes, labels
        
        try:
            transformed = self.image_transform(
                image=image,
                bboxes=bboxes,
                labels=labels
            )
            
            return (
                transformed['image'],
                transformed['bboxes'],
                transformed['labels']
            )
        except Exception as e:
            logger.warning(f"Augmentation failed: {str(e)}, returning original")
            return image, bboxes, labels
    
    def augment_text(self, text: str, label: str = "O") -> str:
        """
        Apply text-level augmentations.
        
        Args:
            text: Input text
            label: Entity label (skip augmentation for certain labels)
            
        Returns:
            Augmented text
        """
        if not self.enabled or label == "O":
            return text
        
        words = text.split()
        if len(words) == 0:
            return text
        
        # Synonym replacement
        if random.random() < self.text_config.get('synonym_replacement', 0):
            words = self._synonym_replacement(words)
        
        # Random deletion
        if random.random() < self.text_config.get('random_deletion', 0):
            words = self._random_deletion(words)
        
        # OCR error simulation
        if random.random() < self.text_config.get('ocr_error_simulation', 0):
            words = self._simulate_ocr_errors(words)
        
        return ' '.join(words)
    
    def _synonym_replacement(
        self, 
        words: List[str],
        n: int = 1
    ) -> List[str]:
        """Replace n words with synonyms."""
        # Simple synonym mapping for common invoice terms
        synonym_map = {
            'invoice': ['bill', 'receipt'],
            'total': ['amount', 'sum'],
            'date': ['dated'],
            'quantity': ['qty', 'count'],
            'price': ['cost', 'rate'],
        }
        
        new_words = words.copy()
        for _ in range(n):
            if len(new_words) == 0:
                break
            idx = random.randint(0, len(new_words) - 1)
            word_lower = new_words[idx].lower()
            
            if word_lower in synonym_map:
                synonym = random.choice(synonym_map[word_lower])
                # Preserve case
                if new_words[idx].isupper():
                    new_words[idx] = synonym.upper()
                elif new_words[idx][0].isupper():
                    new_words[idx] = synonym.capitalize()
                else:
                    new_words[idx] = synonym
        
        return new_words
    
    def _random_deletion(
        self, 
        words: List[str],
        p: float = 0.1
    ) -> List[str]:
        """Randomly delete words with probability p."""
        if len(words) == 1:
            return words
        
        new_words = [w for w in words if random.random() > p]
        
        # Ensure at least one word remains
        if len(new_words) == 0:
            return [random.choice(words)]
        
        return new_words
    
    def _simulate_ocr_errors(self, words: List[str]) -> List[str]:
        """Simulate common OCR errors."""
        # Common OCR character substitutions
        ocr_errors = {
            'o': ['0', 'O'],
            'O': ['0', 'o'],
            '0': ['o', 'O'],
            'l': ['1', 'I'],
            'I': ['1', 'l'],
            '1': ['l', 'I'],
            'S': ['5'],
            '5': ['S'],
            'B': ['8'],
            '8': ['B'],
        }
        
        new_words = []
        for word in words:
            if random.random() < 0.3 and len(word) > 2:  # 30% chance per word
                # Pick a random character to corrupt
                idx = random.randint(0, len(word) - 1)
                char = word[idx]
                
                if char in ocr_errors:
                    replacement = random.choice(ocr_errors[char])
                    word = word[:idx] + replacement + word[idx+1:]
            
            new_words.append(word)
        
        return new_words


class MixupAugmentation:
    """Mixup augmentation for document images."""
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize Mixup.
        
        Args:
            alpha: Beta distribution parameter for mixing ratio
        """
        self.alpha = alpha
    
    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        labels1: np.ndarray,
        labels2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mixup augmentation.
        
        Args:
            image1, image2: Input images
            labels1, labels2: Corresponding labels
            
        Returns:
            Mixed image and labels
        """
        # Sample mixing ratio
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2
        
        # Mix labels (for soft labels)
        mixed_labels = lam * labels1 + (1 - lam) * labels2
        
        return mixed_image.astype(np.float32), mixed_labels


def get_train_transforms(config_path: str = "./configs/training_config.yaml") -> A.Compose:
    """Get training augmentation pipeline."""
    aug = DocumentAugmentation(config_path)
    return aug.image_transform if aug.enabled else A.Compose([])


def get_val_transforms() -> A.Compose:
    """Get validation transforms (no augmentation)."""
    return A.Compose([])


if __name__ == "__main__":
    # Test augmentation pipeline
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy image and bboxes
    image = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
    bboxes = [[100, 100, 200, 150], [300, 300, 400, 400]]
    labels = [1, 2]
    
    aug = DocumentAugmentation()
    
    for i in range(5):
        aug_image, aug_bboxes, aug_labels = aug.augment_image(image, bboxes, labels)
        logger.info(f"Iteration {i+1}: {len(aug_bboxes)} bboxes retained")
    
    # Test text augmentation
    text = "Invoice number 12345 dated January 1, 2024"
    for i in range(5):
        aug_text = aug.augment_text(text, "B-INVOICE_NUMBER")
        logger.info(f"Original: {text}")
        logger.info(f"Augmented: {aug_text}")
