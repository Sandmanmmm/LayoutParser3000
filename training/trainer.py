"""
Training Module - Production Multi-Task
Training loop for LayoutLMv3 with multi-task learning:
- NER (115 labels)
- Cell detection (binary)
- Column classification (16 classes)

Features:
- Separate loss component logging
- Per-task metrics tracking
- Composite metric for checkpointing
- Mixed precision (FP16)
- Gradient accumulation
"""

from typing import Dict
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    get_scheduler,
    LayoutLMv3Processor
)
from tqdm import tqdm
import yaml
import logging

from models.layoutlmv3_model import LayoutLMv3ForMultiTask
from preprocessing import create_dataloaders, DocumentAugmentation

logger = logging.getLogger(__name__)


class Trainer:
    """
    Production trainer for LayoutLMv3 multi-task learning.
    
    Handles:
    - Multi-task loss (NER + Cell + Column)
    - Individual loss component logging
    - Per-task metrics computation
    - Composite metric for best model selection
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cuda"
    ):
        """
        Initialize trainer.
        
        Args:
            model: LayoutLMv3ForMultiTask model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Training config
        training_cfg = config['training']
        self.num_epochs = training_cfg['num_epochs']
        self.gradient_accumulation_steps = training_cfg.get(
            'gradient_accumulation_steps', 1
        )
        self.max_grad_norm = training_cfg.get('max_grad_norm', 1.0)
        
        # Multi-task loss tracking
        self.loss_weights = config.get('loss_weights', {
            'ner_loss_weight': 1.0,
            'cell_loss_weight': 1.0,
            'col_loss_weight': 0.5
        })
        self.track_loss_components = True
        
        # Mixed precision training
        self.use_fp16 = training_cfg.get('fp16', True) and device == "cuda"
        self.scaler = GradScaler() if self.use_fp16 else None
        
        # Optimizer
        self.optimizer = self._setup_optimizer(training_cfg)
        
        # Learning rate scheduler
        num_training_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        num_warmup_steps = int(num_training_steps * training_cfg.get('warmup_ratio', 0.1))
        
        self.scheduler = get_scheduler(
            name=training_cfg.get('lr_scheduler_type', 'cosine'),
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Checkpointing
        self.output_dir = Path(training_cfg['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float('-inf')
        self.patience_counter = 0
        self.early_stopping_patience = training_cfg.get('early_stopping_patience', 5)
        
        # Logging
        self.logging_steps = training_cfg.get('logging_steps', 50)
        self._setup_logging()
        
        # Gradient checkpointing
        if training_cfg.get('gradient_checkpointing', False):
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
    
    def _setup_optimizer(self, training_cfg: Dict) -> torch.optim.Optimizer:
        """Setup optimizer with weight decay."""
        # Separate parameters with/without weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                'weight_decay': training_cfg.get('weight_decay', 0.01)
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=training_cfg.get('learning_rate', 5e-5),
            eps=training_cfg.get('adam_epsilon', 1e-8),
            betas=(
                training_cfg.get('adam_beta1', 0.9),
                training_cfg.get('adam_beta2', 0.999)
            )
        )
        
        return optimizer
    
    def _setup_logging(self) -> None:
        """Setup tensorboard and wandb logging."""
        logging_cfg = self.config.get('logging', {})
        
        # Tensorboard
        if logging_cfg.get('use_tensorboard', True):
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = Path(logging_cfg.get('tensorboard_dir', './logs/tensorboard'))
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(tb_dir)
        else:
            self.tb_writer = None
        
        # Weights & Biases
        if logging_cfg.get('use_wandb', False):
            try:
                import wandb
                wandb.init(
                    project=logging_cfg.get('wandb_project', 'layoutlmv3-training'),
                    entity=logging_cfg.get('wandb_entity'),
                    config=self.config
                )
                self.use_wandb = True
            except ImportError:
                logger.warning("wandb not installed, skipping W&B logging")
                self.use_wandb = False
        else:
            self.use_wandb = False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with multi-task loss tracking.
        
        Returns:
            Dictionary with total_loss and component losses
        """
        self.model.train()
        total_loss = 0
        ner_loss_sum = 0
        cell_loss_sum = 0
        col_loss_sum = 0
        steps = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.num_epochs}"
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Forward pass with mixed precision
            with autocast(enabled=self.use_fp16):
                outputs = self.model(**batch)
                
                # Handle dict output from multi-task model
                if isinstance(outputs, dict):
                    loss = outputs['loss']
                    ner_loss = outputs.get('ner_loss')
                    cell_loss = outputs.get('cell_loss')
                    col_loss = outputs.get('col_loss')
                else:
                    # Fallback for single-task models
                    loss = outputs.loss
                    ner_loss = cell_loss = col_loss = None
                
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_fp16:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                
                # Optimizer step
                if self.use_fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                steps += 1
            
            # Accumulate losses
            total_loss += loss.item() * self.gradient_accumulation_steps
            if ner_loss is not None:
                ner_loss_sum += ner_loss.item()
            if cell_loss is not None:
                cell_loss_sum += cell_loss.item()
            if col_loss is not None:
                col_loss_sum += col_loss.item()
            
            # Logging
            if steps % self.logging_steps == 0 and steps > 0:
                avg_loss = total_loss / steps
                avg_ner = ner_loss_sum / steps if ner_loss_sum > 0 else 0
                avg_cell = cell_loss_sum / steps if cell_loss_sum > 0 else 0
                avg_col = col_loss_sum / steps if col_loss_sum > 0 else 0
                lr = self.scheduler.get_last_lr()[0]
                
                progress_bar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ner': f'{avg_ner:.4f}',
                    'cell': f'{avg_cell:.4f}',
                    'col': f'{avg_col:.4f}',
                    'lr': f'{lr:.2e}'
                })
                
                # TensorBoard logging
                if self.tb_writer:
                    global_step = epoch * len(self.train_loader) + batch_idx
                    self.tb_writer.add_scalar(
                        'train/total_loss', avg_loss, global_step
                    )
                    if avg_ner > 0:
                        self.tb_writer.add_scalar(
                            'train/ner_loss', avg_ner, global_step
                        )
                    if avg_cell > 0:
                        self.tb_writer.add_scalar(
                            'train/cell_loss', avg_cell, global_step
                        )
                    if avg_col > 0:
                        self.tb_writer.add_scalar(
                            'train/col_loss', avg_col, global_step
                        )
                    self.tb_writer.add_scalar('train/lr', lr, global_step)
                
                # Weights & Biases logging
                if self.use_wandb:
                    import wandb
                    log_dict = {
                        'train/total_loss': avg_loss,
                        'train/lr': lr,
                        'epoch': epoch
                    }
                    if avg_ner > 0:
                        log_dict['train/ner_loss'] = avg_ner
                    if avg_cell > 0:
                        log_dict['train/cell_loss'] = avg_cell
                    if avg_col > 0:
                        log_dict['train/col_loss'] = avg_col
                    wandb.log(log_dict)
        
        # Return loss components
        num_steps = len(self.train_loader)
        return {
            'total_loss': total_loss / num_steps,
            'ner_loss': ner_loss_sum / num_steps if ner_loss_sum > 0 else 0,
            'cell_loss': cell_loss_sum / num_steps if cell_loss_sum > 0 else 0,
            'col_loss': col_loss_sum / num_steps if col_loss_sum > 0 else 0
        }
    
    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict[str, float]:
        """
        Evaluate on validation set with multi-task metrics.
        
        Returns:
            Dictionary with loss components and task-specific metrics
        """
        self.model.eval()
        total_loss = 0
        ner_loss_sum = 0
        cell_loss_sum = 0
        col_loss_sum = 0
        
        # Predictions for each task
        all_ner_preds = []
        all_ner_labels = []
        all_cell_preds = []
        all_cell_labels = []
        all_col_preds = []
        all_col_labels = []
        
        progress_bar = tqdm(self.val_loader, desc="Evaluating")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            outputs = self.model(**batch)
            
            # Handle dict output
            if isinstance(outputs, dict):
                if outputs['loss'] is not None:
                    total_loss += outputs['loss'].item()
                if outputs.get('ner_loss') is not None:
                    ner_loss_sum += outputs['ner_loss'].item()
                if outputs.get('cell_loss') is not None:
                    cell_loss_sum += outputs['cell_loss'].item()
                if outputs.get('col_loss') is not None:
                    col_loss_sum += outputs['col_loss'].item()
                
                # Get predictions for each task
                ner_preds = torch.argmax(
                    outputs['ner_logits'], dim=-1
                ).cpu().numpy()
                cell_preds = torch.argmax(
                    outputs['cell_logits'], dim=-1
                ).cpu().numpy()
                col_preds = torch.argmax(
                    outputs['col_logits'], dim=-1
                ).cpu().numpy()
            else:
                # Fallback for single-task
                if outputs.loss is not None:
                    total_loss += outputs.loss.item()
                ner_preds = torch.argmax(
                    outputs.logits, dim=-1
                ).cpu().numpy()
                cell_preds = None
                col_preds = None
            
            # Store predictions and labels (exclude -100)
            if 'labels' in batch:
                ner_labels_np = batch['labels'].cpu().numpy()
                for pred, label in zip(ner_preds, ner_labels_np):
                    mask = label != -100
                    all_ner_preds.extend(pred[mask].tolist())
                    all_ner_labels.extend(label[mask].tolist())
            
            if cell_preds is not None and 'cell_labels' in batch:
                cell_labels_np = batch['cell_labels'].cpu().numpy()
                for pred, label in zip(cell_preds, cell_labels_np):
                    mask = label != -100
                    all_cell_preds.extend(pred[mask].tolist())
                    all_cell_labels.extend(label[mask].tolist())
            
            if col_preds is not None and 'col_labels' in batch:
                col_labels_np = batch['col_labels'].cpu().numpy()
                for pred, label in zip(col_preds, col_labels_np):
                    mask = label != -100
                    all_col_preds.extend(pred[mask].tolist())
                    all_col_labels.extend(label[mask].tolist())
        
        # Average losses
        num_batches = len(self.val_loader)
        metrics = {
            'eval_loss': total_loss / num_batches,
            'eval_ner_loss': ner_loss_sum / num_batches,
            'eval_cell_loss': cell_loss_sum / num_batches,
            'eval_col_loss': col_loss_sum / num_batches
        }
        
        # Compute task-specific metrics
        if all_ner_labels:
            ner_metrics = self._compute_classification_metrics(
                all_ner_preds, all_ner_labels, 'ner'
            )
            metrics.update(ner_metrics)
        
        if all_cell_labels:
            cell_metrics = self._compute_classification_metrics(
                all_cell_preds, all_cell_labels, 'cell'
            )
            metrics.update(cell_metrics)
        
        if all_col_labels:
            col_metrics = self._compute_classification_metrics(
                all_col_preds, all_col_labels, 'col'
            )
            metrics.update(col_metrics)
        
        # Composite metric (weighted average of task F1s)
        composite_f1 = 0
        weights_sum = 0
        if 'eval_ner_f1' in metrics:
            composite_f1 += metrics['eval_ner_f1'] * 1.0
            weights_sum += 1.0
        if 'eval_cell_f1' in metrics:
            composite_f1 += metrics['eval_cell_f1'] * 0.5
            weights_sum += 0.5
        if 'eval_col_f1' in metrics:
            composite_f1 += metrics['eval_col_f1'] * 0.3
            weights_sum += 0.3
        
        if weights_sum > 0:
            metrics['eval_composite_f1'] = composite_f1 / weights_sum
        
        # Log metrics
        logger.info(f"Epoch {epoch+1} Validation Metrics:")
        for k, v in sorted(metrics.items()):
            logger.info(f"  {k}: {v:.4f}")
        
        if self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(f'eval/{k}', v, epoch)
        
        if self.use_wandb:
            import wandb
            wandb.log({f'eval/{k}': v for k, v in metrics.items()})
        
        return metrics
    
    def _compute_classification_metrics(
        self,
        predictions: list,
        labels: list,
        task_name: str
    ) -> Dict[str, float]:
        """
        Compute precision, recall, F1 for a classification task.
        
        Args:
            predictions: List of predicted labels
            labels: List of ground truth labels
            task_name: Name prefix for metrics (e.g., 'ner', 'cell')
            
        Returns:
            Dictionary with precision, recall, F1
        """
        from sklearn.metrics import (
            precision_recall_fscore_support,
            accuracy_score
        )
        
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted', zero_division=0
            )
            accuracy = accuracy_score(labels, predictions)
            
            return {
                f'eval_{task_name}_precision': precision,
                f'eval_{task_name}_recall': recall,
                f'eval_{task_name}_f1': f1,
                f'eval_{task_name}_accuracy': accuracy
            }
        except Exception as e:
            logger.warning(
                f"Error computing metrics for {task_name}: {e}"
            )
            return {}
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ) -> None:
        """
        Save model checkpoint with multi-task metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary with all task metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'loss_weights': self.loss_weights
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
            
            # Also save model for HuggingFace
            self.model.save_pretrained(self.output_dir / "best_model_hf")
    
    def train(self) -> None:
        """Main training loop with multi-task tracking."""
        logger.info("=" * 70)
        logger.info("Starting Multi-Task Training")
        logger.info("=" * 70)
        logger.info(f"Epochs: {self.num_epochs}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mixed Precision (FP16): {self.use_fp16}")
        logger.info(
            f"Gradient Accumulation: {self.gradient_accumulation_steps}"
        )
        logger.info(f"Loss Weights: {self.loss_weights}")
        logger.info("=" * 70)
        
        for epoch in range(self.num_epochs):
            # Train
            train_losses = self.train_epoch(epoch)
            logger.info(f"\nEpoch {epoch+1} - Training Losses:")
            logger.info(f"  Total: {train_losses['total_loss']:.4f}")
            if train_losses['ner_loss'] > 0:
                logger.info(f"  NER: {train_losses['ner_loss']:.4f}")
            if train_losses['cell_loss'] > 0:
                logger.info(f"  Cell: {train_losses['cell_loss']:.4f}")
            if train_losses['col_loss'] > 0:
                logger.info(f"  Column: {train_losses['col_loss']:.4f}")
            
            # Evaluate
            metrics = self.evaluate(epoch)
            
            # Check for improvement using composite metric
            metric_for_best = self.config['training'].get(
                'metric_for_best_model', 'eval_composite_f1'
            )
            current_metric = metrics.get(metric_for_best, 0.0)
            
            is_best = current_metric > self.best_metric
            if is_best:
                logger.info(
                    f"\n✨ New best {metric_for_best}: "
                    f"{current_metric:.4f} (prev: {self.best_metric:.4f})"
                )
                self.best_metric = current_metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement. Patience: "
                    f"{self.patience_counter}/{self.early_stopping_patience}"
                )
            
            # Save checkpoint
            self.save_checkpoint(epoch, metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs "
                    f"(patience: {self.early_stopping_patience})"
                )
                break
        
        logger.info("\n" + "=" * 70)
        logger.info("Training Completed!")
        logger.info("=" * 70)
        logger.info(f"Best {metric_for_best}: {self.best_metric:.4f}")
        logger.info(f"Best model saved to: {self.output_dir / 'best_model.pt'}")
        logger.info("=" * 70)
        
        if self.tb_writer:
            self.tb_writer.close()


def train_model(config_path: str = "./configs/training_config.yaml") -> None:
    """
    Train LayoutLMv3 model.
    
    Args:
        config_path: Path to training configuration file
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(
        config['hardware']['device'] 
        if torch.cuda.is_available() 
        else "cpu"
    )
    logger.info(f"Using device: {device}")
    
    # Load processor and create label mapping
    processor = LayoutLMv3Processor.from_pretrained(
        config['model']['name']
    )
    
    labels = config['labels']['token_classification']
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # Create dataloaders
    augmentation = DocumentAugmentation(config_path) if config['augmentation']['enabled'] else None
    
    train_loader, val_loader = create_dataloaders(
        Path(config['data']['train_data_path']),
        Path(config['data']['val_data_path']),
        processor,
        label2id,
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        train_augmentation=augmentation
    )
    
    # Initialize model
    from transformers import LayoutLMv3Config
    model_config = LayoutLMv3Config.from_pretrained(
        config['model']['name'],
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )
    
    # Add custom config attributes for multi-task learning
    model_config.use_crf = config['model'].get('use_crf', False)
    model_config.use_table_head = config['model']['table_structure']['enabled']
    if model_config.use_table_head:
        model_config.num_row_labels = config['model']['table_structure'][
            'num_row_labels'
        ]
        model_config.num_col_labels = config['model']['table_structure'][
            'num_col_labels'
        ]
    
    # Use multi-task model
    model = LayoutLMv3ForMultiTask.from_pretrained(
        config['model']['name'],
        config=model_config
    )
    
    # Create trainer and train
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        config,
        device=str(device)
    )
    
    trainer.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train LayoutLMv3 model")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/training_config.yaml",
        help="Path to training config"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    train_model(args.config)
