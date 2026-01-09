"""
Advanced training utilities for winning Kaggle competitions.

Includes:
- Layer-wise Learning Rate Decay (LLRD)
- Label Smoothing
- Focal Loss
- R-Drop Regularization
- Exponential Moving Average (EMA)
- FGM Adversarial Training
- Stochastic Weight Averaging (SWA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, List, Tuple, Callable
import copy
import math


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Label smoothing helps prevent overconfidence and improves generalization.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) raw model output
            targets: (batch, num_classes) one-hot or soft labels
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Apply label smoothing to targets
        if targets.dim() == 2:
            # Soft labels - apply smoothing
            smoothed_targets = targets * (1 - self.smoothing) + self.smoothing / num_classes
        else:
            # Hard labels - convert to one-hot first
            one_hot = torch.zeros_like(logits).scatter_(1, targets.unsqueeze(1), 1)
            smoothed_targets = one_hot * (1 - self.smoothing) + self.smoothing / num_classes

        loss = -torch.sum(smoothed_targets * log_probs, dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focuses learning on hard examples by down-weighting easy examples.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        alpha: Optional[List[float]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch, num_classes) one-hot or (batch,) class indices
        """
        probs = F.softmax(logits, dim=-1)

        if targets.dim() == 2:
            # One-hot targets
            ce_loss = -torch.sum(targets * torch.log(probs + 1e-10), dim=-1)
            pt = torch.sum(probs * targets, dim=-1)
        else:
            # Class indices
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = torch.tensor(self.alpha, device=logits.device)
            if targets.dim() == 2:
                alpha_weight = torch.sum(alpha_t * targets, dim=-1)
            else:
                alpha_weight = alpha_t[targets]
            loss = alpha_weight * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class RDropLoss(nn.Module):
    """
    R-Drop: Regularized Dropout for Neural Networks.

    Forces the model to produce consistent outputs across multiple
    forward passes with different dropout masks.
    """

    def __init__(self, alpha: float = 0.3, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits1: First forward pass output (batch, num_classes)
            logits2: Second forward pass output (batch, num_classes)
            targets: (batch, num_classes) or (batch,)

        Returns:
            total_loss: Combined CE + KL loss
            kl_loss: KL divergence component (for logging)
        """
        # Cross entropy loss (average of both passes)
        if targets.dim() == 2:
            ce1 = -torch.sum(targets * F.log_softmax(logits1, dim=-1), dim=-1)
            ce2 = -torch.sum(targets * F.log_softmax(logits2, dim=-1), dim=-1)
        else:
            ce1 = self.ce_loss(logits1, targets)
            ce2 = self.ce_loss(logits2, targets)

        ce_loss = (ce1 + ce2) / 2

        # KL divergence between the two outputs (symmetric)
        p1 = F.log_softmax(logits1, dim=-1)
        p2 = F.log_softmax(logits2, dim=-1)
        q1 = F.softmax(logits1, dim=-1)
        q2 = F.softmax(logits2, dim=-1)

        kl1 = self.kl_loss(p1, q2).sum(dim=-1)
        kl2 = self.kl_loss(p2, q1).sum(dim=-1)
        kl_loss = (kl1 + kl2) / 2

        total_loss = ce_loss + self.alpha * kl_loss

        if self.reduction == 'mean':
            return total_loss.mean(), kl_loss.mean()
        return total_loss, kl_loss


# =============================================================================
# LAYER-WISE LEARNING RATE DECAY (LLRD)
# =============================================================================

def get_optimizer_with_llrd(
    model: nn.Module,
    base_lr: float = 2e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 0.01,
    llrd_factor: float = 0.9,
    no_decay_keywords: List[str] = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
) -> AdamW:
    """
    Create optimizer with layer-wise learning rate decay.

    Earlier layers get lower learning rates, later layers get higher.
    Classification head gets the highest learning rate.

    Args:
        model: The model
        base_lr: Base learning rate for middle layers
        head_lr: Learning rate for classification head
        weight_decay: Weight decay for regularization
        llrd_factor: Decay factor per layer (0.9 = 10% decay per layer)
        no_decay_keywords: Parameters that shouldn't have weight decay
    """
    optimizer_grouped_parameters = []

    # Find all named layers in the backbone
    layers = []
    for name, _ in model.named_parameters():
        if 'backbone' in name and 'layer' in name:
            # Extract layer number
            parts = name.split('.')
            for part in parts:
                if part.startswith('layer'):
                    layer_num = int(part.replace('layer', '').replace('s', ''))
                    if layer_num not in layers:
                        layers.append(layer_num)

    num_layers = max(layers) if layers else 12

    # Group parameters by layer
    param_groups = {
        'embeddings': {'params': [], 'params_no_decay': []},
        'head': {'params': [], 'params_no_decay': []},
    }
    for i in range(num_layers + 1):
        param_groups[f'layer_{i}'] = {'params': [], 'params_no_decay': []}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine which group this parameter belongs to
        group_name = 'head'  # Default

        if 'backbone' in name:
            if 'embeddings' in name:
                group_name = 'embeddings'
            else:
                for i in range(num_layers + 1):
                    if f'layer.{i}.' in name or f'layers.{i}.' in name:
                        group_name = f'layer_{i}'
                        break

        # Check if parameter should have weight decay
        has_decay = not any(nd in name for nd in no_decay_keywords)

        if has_decay:
            param_groups[group_name]['params'].append(param)
        else:
            param_groups[group_name]['params_no_decay'].append(param)

    # Create optimizer groups with LLRD
    # Embeddings get lowest LR
    if param_groups['embeddings']['params']:
        optimizer_grouped_parameters.append({
            'params': param_groups['embeddings']['params'],
            'lr': base_lr * (llrd_factor ** num_layers),
            'weight_decay': weight_decay
        })
    if param_groups['embeddings']['params_no_decay']:
        optimizer_grouped_parameters.append({
            'params': param_groups['embeddings']['params_no_decay'],
            'lr': base_lr * (llrd_factor ** num_layers),
            'weight_decay': 0.0
        })

    # Layers get progressively higher LR
    for i in range(num_layers + 1):
        layer_lr = base_lr * (llrd_factor ** (num_layers - i))

        if param_groups[f'layer_{i}']['params']:
            optimizer_grouped_parameters.append({
                'params': param_groups[f'layer_{i}']['params'],
                'lr': layer_lr,
                'weight_decay': weight_decay
            })
        if param_groups[f'layer_{i}']['params_no_decay']:
            optimizer_grouped_parameters.append({
                'params': param_groups[f'layer_{i}']['params_no_decay'],
                'lr': layer_lr,
                'weight_decay': 0.0
            })

    # Head gets highest LR
    if param_groups['head']['params']:
        optimizer_grouped_parameters.append({
            'params': param_groups['head']['params'],
            'lr': head_lr,
            'weight_decay': weight_decay
        })
    if param_groups['head']['params_no_decay']:
        optimizer_grouped_parameters.append({
            'params': param_groups['head']['params_no_decay'],
            'lr': head_lr,
            'weight_decay': 0.0
        })

    return AdamW(optimizer_grouped_parameters, eps=1e-8)


# =============================================================================
# EXPONENTIAL MOVING AVERAGE (EMA)
# =============================================================================

class EMA:
    """
    Exponential Moving Average for model weights.

    Maintains a moving average of model parameters which often
    generalizes better than the final checkpoint.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow weights with current model weights."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply shadow weights to model (for inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (after inference)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# FGM ADVERSARIAL TRAINING
# =============================================================================

class FGM:
    """
    Fast Gradient Method for adversarial training.

    Adds perturbations to embeddings to improve robustness.
    """

    def __init__(self, model: nn.Module, epsilon: float = 1.0, emb_name: str = 'word_embeddings'):
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}

    def attack(self):
        """Apply adversarial perturbation to embeddings."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        """Restore original embeddings."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


# =============================================================================
# LEARNING RATE SCHEDULERS
# =============================================================================

def get_cosine_schedule_with_warmup_and_hard_restarts(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: int = 1,
    min_lr_ratio: float = 0.0
) -> LambdaLR:
    """
    Cosine learning rate schedule with warmup and optional hard restarts.

    Args:
        optimizer: The optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        num_cycles: Number of cosine cycles (restarts)
        min_lr_ratio: Minimum LR as fraction of initial LR
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay with restarts
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

        if num_cycles > 1:
            progress = progress * num_cycles
            progress = progress - int(progress)  # Reset for each cycle

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# STOCHASTIC WEIGHT AVERAGING
# =============================================================================

class SWA:
    """
    Stochastic Weight Averaging.

    Averages model weights across training to find flatter minima
    that generalize better.
    """

    def __init__(self, model: nn.Module, swa_start: int = 0, swa_freq: int = 1):
        self.model = model
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_n = 0
        self.swa_state = {}

        # Initialize SWA weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.swa_state[name] = torch.zeros_like(param.data)

    def update(self, step: int):
        """Update SWA weights if we're past the start point."""
        if step >= self.swa_start and (step - self.swa_start) % self.swa_freq == 0:
            self.swa_n += 1
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.swa_state[name] += (param.data - self.swa_state[name]) / self.swa_n

    def apply_swa(self):
        """Apply SWA weights to model."""
        if self.swa_n > 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data = self.swa_state[name].clone()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def mixup_data(
    x: Dict[str, torch.Tensor],
    y: torch.Tensor,
    alpha: float = 0.2
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, float]:
    """
    Mixup data augmentation for text classification.

    Note: For transformer models, this mixes the embeddings or labels.

    Args:
        x: Input dict with 'input_ids', 'attention_mask'
        y: Labels
        alpha: Mixup interpolation strength

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = y.size(0)
    index = torch.randperm(batch_size, device=y.device)

    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def freeze_layers(model: nn.Module, num_layers_to_freeze: int = 0):
    """
    Freeze the first N layers of a transformer model.

    Useful for gradual unfreezing during training.
    """
    for name, param in model.named_parameters():
        if 'embeddings' in name:
            param.requires_grad = False
        elif 'layer' in name:
            # Extract layer number
            for part in name.split('.'):
                if part.startswith('layer') or part.isdigit():
                    try:
                        layer_num = int(part.replace('layer', '').replace('s', ''))
                        if layer_num < num_layers_to_freeze:
                            param.requires_grad = False
                        break
                    except ValueError:
                        continue


def unfreeze_all(model: nn.Module):
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
