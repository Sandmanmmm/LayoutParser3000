"""
LayoutLMv3 Model with Multi-Task Heads - Production Grade
Extended model for token classification (115 labels) + table structure
detection.

Architecture:
  - NER Head: 115 labels (57 entity types × 2 + O) with optional CRF
  - Cell Head: Binary classification (in-table vs not-in-table)
  - Column Head: 16-way classification (column index 0-15)

Multi-task loss: α·NER + β·Cell + γ·Column
"""

from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
from transformers import LayoutLMv3Model, LayoutLMv3PreTrainedModel
import logging

logger = logging.getLogger(__name__)

# Import pytorch-crf for production CRF layer
try:
    from torchcrf import CRF
    CRF_AVAILABLE = True
except ImportError:
    CRF_AVAILABLE = False
    logger.warning("pytorch-crf not installed. CRF layer will be disabled.")


class CRFLayer(nn.Module):
    """Conditional Random Field layer for sequence labeling."""
    
    def __init__(self, num_labels: int):
        """
        Initialize CRF layer.
        
        Args:
            num_labels: Number of label classes
        """
        super().__init__()
        self.num_labels = num_labels
        
        # Transition scores between labels
        self.transitions = nn.Parameter(
            torch.randn(num_labels, num_labels)
        )
        
        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_labels))
        self.end_transitions = nn.Parameter(torch.randn(num_labels))
    
    def forward(
        self,
        emissions: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through CRF.
        
        Args:
            emissions: Emission scores (batch_size, seq_len, num_labels)
            labels: True labels (batch_size, seq_len)
            mask: Attention mask (batch_size, seq_len)
            
        Returns:
            loss and predictions
        """
        if mask is None:
            mask = torch.ones_like(emissions[:, :, 0], dtype=torch.bool)
        
        if labels is not None:
            # Compute loss during training
            loss = -self._compute_log_likelihood(emissions, labels, mask)
            return loss, self.decode(emissions, mask)
        else:
            # Only decode during inference
            return None, self.decode(emissions, mask)
    
    def _compute_log_likelihood(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute log likelihood of the labels."""
        batch_size, seq_len = labels.shape
        
        # Compute score of the gold sequence
        gold_score = self._compute_score(emissions, labels, mask)
        
        # Compute partition function (sum over all possible sequences)
        forward_score = self._forward_algorithm(emissions, mask)
        
        # Log likelihood = gold_score - log(Z)
        log_likelihood = gold_score - forward_score
        
        return log_likelihood.sum()
    
    def _compute_score(
        self,
        emissions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute score of a given sequence."""
        batch_size, seq_len = labels.shape
        
        score = self.start_transitions[labels[:, 0]]
        score += emissions[:, 0].gather(
            1, labels[:, 0].unsqueeze(1)
        ).squeeze(1)
        
        for i in range(1, seq_len):
            mask_i = mask[:, i]
            
            # Transition score
            trans_score = self.transitions[labels[:, i-1], labels[:, i]]
            # Emission score
            emit_score = emissions[:, i].gather(
                1, labels[:, i].unsqueeze(1)
            ).squeeze(1)
            
            score += (trans_score + emit_score) * mask_i
        
        # Add end transition
        last_labels = labels.gather(
            1, mask.sum(1).long().unsqueeze(1) - 1
        ).squeeze(1)
        score += self.end_transitions[last_labels]
        
        return score
    
    def _forward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward algorithm to compute partition function."""
        batch_size, seq_len, num_labels = emissions.shape
        
        # Initialize with start transitions
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        
        for i in range(1, seq_len):
            # Broadcast for all possible transitions
            emit_score = emissions[:, i].unsqueeze(1)
            trans_score = self.transitions.unsqueeze(0)
            
            # Current step score
            next_alpha = alpha.unsqueeze(2) + emit_score + trans_score
            next_alpha = torch.logsumexp(next_alpha, dim=1)
            
            # Update alpha with mask
            alpha = (
                next_alpha * mask[:, i].unsqueeze(1) +
                alpha * (~mask[:, i]).unsqueeze(1)
            )
        
        # Add end transitions
        alpha = alpha + self.end_transitions.unsqueeze(0)
        
        return torch.logsumexp(alpha, dim=1)
    
    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Viterbi decoding to find best sequence."""
        batch_size, seq_len, num_labels = emissions.shape
        
        # Initialize
        viterbi = self.start_transitions.unsqueeze(0) + emissions[:, 0]
        backpointers = []
        
        for i in range(1, seq_len):
            # Broadcast for all transitions
            broadcast_viterbi = viterbi.unsqueeze(2)
            broadcast_transitions = self.transitions.unsqueeze(0)
            
            # Compute scores
            next_viterbi = broadcast_viterbi + broadcast_transitions
            next_viterbi, indices = next_viterbi.max(dim=1)
            
            # Add emissions
            next_viterbi = next_viterbi + emissions[:, i]
            
            backpointers.append(indices)
            viterbi = (
                next_viterbi * mask[:, i].unsqueeze(1) +
                viterbi * (~mask[:, i]).unsqueeze(1)
            )
        
        # Add end transitions
        viterbi = viterbi + self.end_transitions.unsqueeze(0)
        
        # Backtrack
        best_paths = []
        _, best_last_labels = viterbi.max(dim=1)
        
        for batch_idx in range(batch_size):
            path = [best_last_labels[batch_idx].item()]
            
            for backpointer in reversed(backpointers):
                path.append(backpointer[batch_idx, path[-1]].item())
            
            path.reverse()
            best_paths.append(path)
        
        return torch.tensor(best_paths, device=emissions.device)


class LayoutLMv3ForMultiTask(LayoutLMv3PreTrainedModel):
    """LayoutLMv3 with multi-task heads.
    
    Heads: NER (115 labels) + Cell Detection + Column Classification
    
    Production-grade implementation with:
    - 115-label NER (57 entity types)
    - Binary cell detection (in-table vs not)
    - 16-way column classification
    - CRF layer for NER (optional)
    - Weighted multi-task loss
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # Base model
        self.layoutlmv3 = LayoutLMv3Model(config)
        
        # Dropout
        self.dropout = nn.Dropout(
            config.hidden_dropout_prob
            if hasattr(config, 'hidden_dropout_prob')
            else 0.1
        )
        
        # NER classification head (115 labels)
        self.ner_classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Optional CRF layer for NER
        self.use_crf = getattr(config, 'use_crf', False)
        if self.use_crf:
            if CRF_AVAILABLE:
                self.crf = CRF(config.num_labels, batch_first=True)
                logger.info(
                    f"Initialized CRF layer with {config.num_labels} labels"
                )
            else:
                logger.warning(
                    "CRF requested but pytorch-crf not available. "
                    "Using softmax instead."
                )
                self.use_crf = False
        
        # Multi-task heads for table structure
        # Cell classifier: binary (in-table vs not-in-table)
        self.cell_classifier = nn.Linear(config.hidden_size, 2)
        
        # Column classifier: 16-way (column index 0-15)
        self.col_classifier = nn.Linear(config.hidden_size, 16)
        
        # Loss weights from config
        self.ner_loss_weight = getattr(config, 'ner_loss_weight', 1.0)
        self.cell_loss_weight = getattr(config, 'cell_loss_weight', 1.0)
        self.col_loss_weight = getattr(config, 'col_loss_weight', 0.5)
        
        logger.info("Initialized LayoutLMv3ForMultiTask:")
        logger.info(f"  - NER labels: {config.num_labels}")
        logger.info("  - Cell detection: Binary")
        logger.info("  - Column classification: 16 classes")
        logger.info(f"  - CRF enabled: {self.use_crf}")
        logger.info(
            f"  - Loss weights: NER={self.ner_loss_weight}, "
            f"Cell={self.cell_loss_weight}, Col={self.col_loss_weight}"
        )
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        cell_labels: Optional[torch.Tensor] = None,
        col_labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Dict[str, Any]]:
        """
        Forward pass with multi-task outputs.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            bbox: Bounding boxes (batch_size, seq_len, 4)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: NER labels (batch_size, seq_len) - use -100
            cell_labels: Cell detection labels (batch_size, seq_len)
            col_labels: Column labels (batch_size, seq_len) - 0-15 or -100
            pixel_values: Document images for visual features
            
        Returns:
            Dictionary with:
            - loss: Total weighted loss (scalar)
            - ner_loss: NER loss component (scalar)
            - cell_loss: Cell detection loss (scalar)
            - col_loss: Column classification loss (scalar)
            - ner_logits: NER predictions (batch_size, seq_len, 115)
            - cell_logits: Cell predictions (batch_size, seq_len, 2)
            - col_logits: Column predictions (batch_size, seq_len, 16)
        """
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )
        
        # Get base model outputs
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        
        # === Multi-task heads ===
        ner_logits = self.ner_classifier(sequence_output)
        cell_logits = self.cell_classifier(sequence_output)
        col_logits = self.col_classifier(sequence_output)
        
        # === Loss computation ===
        total_loss = None
        ner_loss = None
        cell_loss = None
        col_loss = None
        losses_dict = {}
        
        if labels is not None:
            # --- NER Loss ---
            if self.use_crf:
                # CRF loss with mask
                # Note: LayoutLMv3 may add special tokens
                # Adjust labels to match logits sequence length
                batch_size, seq_len, _ = ner_logits.shape
                
                # Pad or trim labels to match logits sequence length
                if labels.shape[1] < seq_len:
                    # Pad labels with -100
                    pad_size = seq_len - labels.shape[1]
                    labels_padded = torch.cat([
                        labels,
                        torch.full(
                            (batch_size, pad_size),
                            -100,
                            dtype=labels.dtype,
                            device=labels.device
                        )
                    ], dim=1)
                elif labels.shape[1] > seq_len:
                    # Trim labels (shouldn't happen normally)
                    labels_padded = labels[:, :seq_len]
                else:
                    labels_padded = labels
                
                # Create mask: valid tokens (not padding and not -100)
                mask = (labels_padded != -100)
                if attention_mask is not None:
                    if attention_mask.shape[1] == seq_len:
                        mask = mask & (attention_mask == 1)
                
                # Replace -100 with 0 for CRF (will be masked anyway)
                labels_for_crf = labels_padded.clone()
                labels_for_crf[labels_padded == -100] = 0
                
                ner_loss = -self.crf(
                    ner_logits, labels_for_crf, mask=mask, reduction='mean'
                )
                losses_dict['ner_loss'] = ner_loss.item()
            else:
                # Standard cross-entropy with ignore_index=-100
                batch_size, seq_len, _ = ner_logits.shape
                
                # Pad or trim labels to match logits
                if labels.shape[1] < seq_len:
                    pad_size = seq_len - labels.shape[1]
                    labels_padded = torch.cat([
                        labels,
                        torch.full(
                            (batch_size, pad_size),
                            -100,
                            dtype=labels.dtype,
                            device=labels.device
                        )
                    ], dim=1)
                elif labels.shape[1] > seq_len:
                    labels_padded = labels[:, :seq_len]
                else:
                    labels_padded = labels
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                ner_loss = loss_fct(
                    ner_logits.view(-1, self.num_labels),
                    labels_padded.view(-1)
                )
                losses_dict['ner_loss'] = ner_loss.item()
            
            # --- Cell Detection Loss ---
            if cell_labels is not None:
                batch_size, seq_len, _ = cell_logits.shape
                
                # Pad or trim cell_labels to match logits
                if cell_labels.shape[1] < seq_len:
                    pad_size = seq_len - cell_labels.shape[1]
                    cell_labels_padded = torch.cat([
                        cell_labels,
                        torch.full(
                            (batch_size, pad_size),
                            -100,
                            dtype=cell_labels.dtype,
                            device=cell_labels.device
                        )
                    ], dim=1)
                elif cell_labels.shape[1] > seq_len:
                    cell_labels_padded = cell_labels[:, :seq_len]
                else:
                    cell_labels_padded = cell_labels
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                cell_loss = loss_fct(
                    cell_logits.view(-1, 2),
                    cell_labels_padded.view(-1)
                )
                losses_dict['cell_loss'] = cell_loss.item()
            
            # --- Column Classification Loss ---
            if col_labels is not None:
                batch_size, seq_len, _ = col_logits.shape
                
                # Pad or trim col_labels to match logits
                if col_labels.shape[1] < seq_len:
                    pad_size = seq_len - col_labels.shape[1]
                    col_labels_padded = torch.cat([
                        col_labels,
                        torch.full(
                            (batch_size, pad_size),
                            -100,
                            dtype=col_labels.dtype,
                            device=col_labels.device
                        )
                    ], dim=1)
                elif col_labels.shape[1] > seq_len:
                    col_labels_padded = col_labels[:, :seq_len]
                else:
                    col_labels_padded = col_labels
                
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                col_loss = loss_fct(
                    col_logits.view(-1, 16),
                    col_labels_padded.view(-1)
                )
                losses_dict['col_loss'] = col_loss.item()
            
            # === Weighted Multi-Task Loss ===
            total_loss = 0.0
            if ner_loss is not None:
                total_loss += self.ner_loss_weight * ner_loss
            if cell_loss is not None:
                total_loss += self.cell_loss_weight * cell_loss
            if col_loss is not None:
                total_loss += self.col_loss_weight * col_loss
            
            if isinstance(total_loss, torch.Tensor):
                losses_dict['total_loss'] = total_loss.item()
            else:
                losses_dict['total_loss'] = float(total_loss)
        
        # === Return outputs ===
        if not return_dict:
            output = (ner_logits, cell_logits, col_logits) + outputs[1:]
            if total_loss is not None:
                return ((total_loss,) + output)
            return output
        
        return {
            'loss': total_loss,
            'ner_loss': ner_loss,
            'cell_loss': cell_loss,
            'col_loss': col_loss,
            'losses': losses_dict,
            'ner_logits': ner_logits,
            'cell_logits': cell_logits,
            'col_logits': col_logits,
            'hidden_states': (
                outputs.hidden_states if output_hidden_states else None
            ),
            'attentions': (
                outputs.attentions if output_attentions else None
            ),
        }
    
    def predict(self, **kwargs) -> Dict[str, Any]:
        """
        Inference-only forward pass (no loss computation).
        
        Returns:
            Dictionary with logits for all tasks.
        """
        return self.forward(**kwargs)


# Backward compatibility alias
LayoutLMv3ForTokenClassification = LayoutLMv3ForMultiTask
