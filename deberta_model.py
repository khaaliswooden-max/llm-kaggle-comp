"""
DeBERTa model for LLM preference classification.

Architecture:
- DeBERTa-v3-large backbone (304M params)
- Mean pooling or CLS pooling
- Classification head (3 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    PreTrainedModel
)
from typing import Optional, Tuple


class MeanPooling(nn.Module):
    """
    Mean pooling over token embeddings with attention mask.
    """
    
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            last_hidden_state: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
        
        Returns:
            pooled: (batch, hidden_dim)
        """
        # Expand attention mask to match hidden state dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        
        # Sum embeddings where mask is 1
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
        
        # Sum of mask (number of valid tokens)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        # Mean
        return sum_embeddings / sum_mask


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over token embeddings.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            last_hidden_state: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
        
        Returns:
            pooled: (batch, hidden_dim)
        """
        # Compute attention weights
        weights = self.attention(last_hidden_state)  # (batch, seq_len, 1)
        
        # Mask padding tokens
        mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        weights = weights.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(weights, dim=1)
        
        # Weighted sum
        pooled = torch.sum(last_hidden_state * weights, dim=1)
        
        return pooled


class PreferenceClassifier(nn.Module):
    """
    DeBERTa-based classifier for LLM preference prediction.
    
    Supports multiple pooling strategies and multi-sample dropout.
    """
    
    def __init__(
        self,
        model_name: str = 'microsoft/deberta-v3-large',
        num_labels: int = 3,
        pooling: str = 'mean',  # 'cls', 'mean', 'attention'
        dropout: float = 0.1,
        multi_sample_dropout: bool = True,
        num_dropout_samples: int = 5,
        gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update({
            'hidden_dropout_prob': dropout,
            'attention_probs_dropout_prob': dropout
        })
        
        self.backbone = AutoModel.from_pretrained(
            model_name,
            config=self.config
        )
        
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
        
        hidden_size = self.config.hidden_size
        
        # Pooling layer
        self.pooling_type = pooling
        if pooling == 'attention':
            self.pooler = AttentionPooling(hidden_size)
        elif pooling == 'mean':
            self.pooler = MeanPooling()
        # 'cls' uses backbone's first token directly
        
        # Multi-sample dropout
        self.multi_sample_dropout = multi_sample_dropout
        self.num_dropout_samples = num_dropout_samples
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classification head weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            token_type_ids: (batch, seq_len) - optional
            labels: (batch, num_labels) - one-hot or soft labels
        
        Returns:
            logits: (batch, num_labels)
            loss: scalar loss if labels provided
        """
        # Get backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # Pooling
        if self.pooling_type == 'cls':
            pooled = last_hidden_state[:, 0, :]  # CLS token
        else:
            pooled = self.pooler(last_hidden_state, attention_mask)
        
        # Multi-sample dropout during training
        if self.training and self.multi_sample_dropout:
            logits_list = []
            for _ in range(self.num_dropout_samples):
                dropped = self.dropout(pooled)
                logits_list.append(self.classifier(dropped))
            logits = torch.mean(torch.stack(logits_list), dim=0)
        else:
            pooled = self.dropout(pooled)
            logits = self.classifier(pooled)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)
        
        return logits, loss
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Supports both hard labels (class indices) and soft labels (probabilities).
        """
        # Check if labels are one-hot/soft or class indices
        if labels.dim() == 2:
            # Soft labels - use KL divergence or soft CE
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -torch.sum(labels * log_probs, dim=-1).mean()
        else:
            # Hard labels
            loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get class probabilities.
        
        Returns:
            probs: (batch, num_labels)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            probs = F.softmax(logits, dim=-1)
        return probs


def get_model_and_tokenizer(
    model_name: str = 'microsoft/deberta-v3-large',
    num_labels: int = 3,
    pooling: str = 'mean',
    **kwargs
) -> Tuple[PreferenceClassifier, AutoTokenizer]:
    """
    Initialize model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = PreferenceClassifier(
        model_name=model_name,
        num_labels=num_labels,
        pooling=pooling,
        **kwargs
    )
    
    return model, tokenizer


class AWP:
    """
    Adversarial Weight Perturbation for robustness.
    
    Adds small perturbations to model weights during training
    to improve generalization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        adv_param: str = 'weight',
        adv_lr: float = 1e-4,
        adv_eps: float = 1e-2
    ):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}
    
    def attack_step(self):
        """Apply adversarial perturbation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = grad_eps
                    
                param.data.add_(param.grad.sign() * self.adv_lr)
                param.data = torch.clamp(
                    param.data,
                    self.backup[name] - self.backup_eps[name],
                    self.backup[name] + self.backup_eps[name]
                )
    
    def restore(self):
        """Restore original weights."""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}
