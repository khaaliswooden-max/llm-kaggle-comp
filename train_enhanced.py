"""
Enhanced training script with competition-winning techniques.

Features:
- Layer-wise Learning Rate Decay (LLRD)
- Label Smoothing / Focal Loss
- R-Drop Regularization
- Exponential Moving Average (EMA)
- FGM/AWP Adversarial Training
- Multiple input format options
- Multi-seed ensembling
- Gradient checkpointing
"""

import os
import gc
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import log_loss

from data_loader import (
    load_data,
    create_folds,
    get_dataloaders,
    PreferenceDataset
)
from deberta_model import get_model_and_tokenizer, AWP
from training_utils import (
    LabelSmoothingCrossEntropy,
    FocalLoss,
    RDropLoss,
    get_optimizer_with_llrd,
    get_cosine_schedule_with_warmup_and_hard_restarts,
    EMA,
    FGM,
    SWA,
    freeze_layers,
    unfreeze_all
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loss_fn(config: dict, device: torch.device) -> nn.Module:
    """Get loss function based on config."""
    loss_type = config.get('loss', {}).get('type', 'cross_entropy')

    if loss_type == 'label_smoothing':
        smoothing = config.get('loss', {}).get('smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)

    elif loss_type == 'focal':
        gamma = config.get('loss', {}).get('gamma', 2.0)
        alpha = config.get('loss', {}).get('alpha', None)
        return FocalLoss(alpha=alpha, gamma=gamma)

    else:
        # Default cross entropy that handles soft labels
        return LabelSmoothingCrossEntropy(smoothing=0.0)


def train_epoch_enhanced(
    model: nn.Module,
    train_loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    config: dict,
    device: torch.device,
    loss_fn: nn.Module,
    ema: Optional[EMA] = None,
    fgm: Optional[FGM] = None,
    awp: Optional[AWP] = None,
    use_rdrop: bool = False,
    rdrop_alpha: float = 0.3
) -> float:
    """
    Enhanced training epoch with all techniques.
    """
    model.train()
    total_loss = 0
    num_batches = 0

    grad_accum_steps = config['training']['gradient_accumulation_steps']
    max_grad_norm = config['training']['max_grad_norm']
    use_fp16 = config['training']['fp16']

    rdrop_loss_fn = RDropLoss(alpha=rdrop_alpha) if use_rdrop else None

    progress_bar = tqdm(train_loader, desc='Training')
    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        token_type_ids = None
        if 'token_type_ids' in batch:
            token_type_ids = batch['token_type_ids'].to(device)

        # Forward pass with mixed precision
        with autocast(enabled=use_fp16):
            if use_rdrop and model.training:
                # R-Drop: two forward passes with different dropout
                logits1, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                logits2, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                loss, kl_loss = rdrop_loss_fn(logits1, logits2, labels)
            else:
                logits, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                loss = loss_fn(logits, labels)

            loss = loss / grad_accum_steps

        # Backward pass
        scaler.scale(loss).backward()

        # FGM adversarial training
        if fgm is not None:
            fgm.attack()
            with autocast(enabled=use_fp16):
                logits_adv, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                loss_adv = loss_fn(logits_adv, labels) / grad_accum_steps
            scaler.scale(loss_adv).backward()
            fgm.restore()

        # AWP adversarial training
        if awp is not None and (step + 1) % grad_accum_steps == 0:
            awp.attack_step()
            with autocast(enabled=use_fp16):
                logits_adv, _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                loss_adv = loss_fn(logits_adv, labels) / grad_accum_steps
            scaler.scale(loss_adv).backward()
            awp.restore()

        # Gradient accumulation step
        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            # Update EMA
            if ema is not None:
                ema.update()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        progress_bar.set_postfix({
            'loss': total_loss / num_batches,
            'lr': scheduler.get_last_lr()[0]
        })

    return total_loss / num_batches


@torch.no_grad()
def validate_enhanced(
    model: nn.Module,
    val_loader,
    device: torch.device,
    ema: Optional[EMA] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Validate model, optionally using EMA weights.
    """
    # Apply EMA weights for validation
    if ema is not None:
        ema.apply_shadow()

    model.eval()

    all_preds = []
    all_targets = []

    for batch in tqdm(val_loader, desc='Validating'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        token_type_ids = None
        if 'token_type_ids' in batch:
            token_type_ids = batch['token_type_ids'].to(device)

        probs = model.predict_proba(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        all_preds.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    # Restore original weights
    if ema is not None:
        ema.restore()

    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    # Calculate log loss
    val_loss = log_loss(targets, predictions)

    return val_loss, predictions, targets


def train_fold_enhanced(
    fold: int,
    train_df: pd.DataFrame,
    config: dict,
    device: torch.device,
    output_dir: Path,
    seed: int = 42
) -> Tuple[float, np.ndarray]:
    """
    Train a single fold with enhanced techniques.
    """
    print(f"\n{'='*50}")
    print(f"Training Fold {fold + 1} (Seed: {seed})")
    print(f"{'='*50}")

    set_seed(seed)

    # Split data
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)

    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Get model and tokenizer
    model, tokenizer = get_model_and_tokenizer(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels'],
        pooling=config['model']['pooling'],
        gradient_checkpointing=True
    )
    model = model.to(device)

    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        train_df=train_data,
        val_df=val_data,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['model']['max_length'],
        num_workers=4
    )

    # Enhanced optimizer with LLRD
    use_llrd = config.get('optimizer', {}).get('use_llrd', True)
    if use_llrd:
        optimizer = get_optimizer_with_llrd(
            model,
            base_lr=config['training']['learning_rate'],
            head_lr=config['training']['learning_rate'] * 5,  # Higher LR for head
            weight_decay=config['training']['weight_decay'],
            llrd_factor=config.get('optimizer', {}).get('llrd_factor', 0.9)
        )
    else:
        from train_deberta import get_optimizer
        optimizer = get_optimizer(model, config)

    # Scheduler
    num_training_steps = (
        len(train_loader) // config['training']['gradient_accumulation_steps']
    ) * config['training']['epochs']

    scheduler = get_cosine_schedule_with_warmup_and_hard_restarts(
        optimizer,
        num_warmup_steps=int(num_training_steps * config['training']['warmup_ratio']),
        num_training_steps=num_training_steps,
        num_cycles=config.get('scheduler', {}).get('num_cycles', 1),
        min_lr_ratio=config.get('scheduler', {}).get('min_lr_ratio', 0.0)
    )

    scaler = GradScaler()

    # Loss function
    loss_fn = get_loss_fn(config, device)

    # Initialize training enhancements
    ema = None
    if config.get('ema', {}).get('enabled', False):
        ema = EMA(model, decay=config['ema'].get('decay', 0.999))
        print("Using EMA")

    fgm = None
    if config.get('fgm', {}).get('enabled', False):
        fgm = FGM(model, epsilon=config['fgm'].get('epsilon', 1.0))
        print("Using FGM adversarial training")

    awp = None
    if config.get('awp', {}).get('enabled', False):
        awp = AWP(
            model, optimizer,
            adv_lr=config['awp'].get('adv_lr', 1e-4),
            adv_eps=config['awp'].get('adv_eps', 1e-2)
        )
        print("Using AWP adversarial training")

    use_rdrop = config.get('rdrop', {}).get('enabled', False)
    rdrop_alpha = config.get('rdrop', {}).get('alpha', 0.3)
    if use_rdrop:
        print(f"Using R-Drop with alpha={rdrop_alpha}")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    oof_preds = None

    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # Train
        train_loss = train_epoch_enhanced(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            device=device,
            loss_fn=loss_fn,
            ema=ema,
            fgm=fgm,
            awp=awp,
            use_rdrop=use_rdrop,
            rdrop_alpha=rdrop_alpha
        )

        # Validate
        val_loss, predictions, targets = validate_enhanced(
            model, val_loader, device, ema=ema
        )

        print(f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            oof_preds = predictions

            # Save checkpoint (with EMA weights if available)
            if ema is not None:
                ema.apply_shadow()

            checkpoint_path = output_dir / f'model_fold{fold}_seed{seed}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

            if ema is not None:
                ema.restore()
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping']['patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Cleanup
    del model, optimizer, scheduler, scaler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_loss, oof_preds


def train_multi_seed(config: dict):
    """
    Train with multiple seeds for ensemble.
    """
    seeds = config.get('ensemble', {}).get('seeds', [42])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare data
    print("Loading data...")
    train_df, test_df = load_data(
        train_path=config['data']['train_path'],
        test_path=config['data']['test_path']
    )

    print(f"Train shape: {train_df.shape}")

    # Create folds
    train_df = create_folds(
        train_df,
        n_folds=config['cv']['n_folds'],
        seed=42,  # Use fixed seed for folds
        stratified=config['cv']['stratified']
    )

    # Save fold assignments
    train_df.to_csv(output_dir / 'train_folds.csv', index=False)

    # Track all results
    all_results = []
    all_oof_preds = {}

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# Training with Seed: {seed}")
        print(f"{'#'*60}")

        seed_oof_preds = np.zeros((len(train_df), config['model']['num_labels']))
        fold_scores = []

        for fold in range(config['cv']['n_folds']):
            fold_loss, oof_preds = train_fold_enhanced(
                fold=fold,
                train_df=train_df,
                config=config,
                device=device,
                output_dir=output_dir,
                seed=seed
            )
            fold_scores.append(fold_loss)

            # Store OOF predictions
            val_idx = train_df[train_df['fold'] == fold].index
            seed_oof_preds[val_idx] = oof_preds

        # Calculate seed CV score
        target_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
        seed_cv_score = log_loss(train_df[target_cols].values, seed_oof_preds)

        print(f"\nSeed {seed} Results:")
        for fold, score in enumerate(fold_scores):
            print(f"  Fold {fold + 1}: {score:.5f}")
        print(f"  CV Score: {seed_cv_score:.5f}")

        all_results.append({
            'seed': seed,
            'cv_score': seed_cv_score,
            'fold_scores': fold_scores
        })
        all_oof_preds[seed] = seed_oof_preds

        # Save seed OOF predictions
        oof_df = train_df[['id']].copy()
        oof_df['winner_model_a'] = seed_oof_preds[:, 0]
        oof_df['winner_model_b'] = seed_oof_preds[:, 1]
        oof_df['winner_tie'] = seed_oof_preds[:, 2]
        oof_df.to_csv(output_dir / f'oof_predictions_seed{seed}.csv', index=False)

    # Calculate ensemble OOF score
    print(f"\n{'='*60}")
    print("Multi-Seed Ensemble Results")
    print(f"{'='*60}")

    # Average OOF predictions across seeds
    ensemble_oof = np.mean([all_oof_preds[s] for s in seeds], axis=0)
    ensemble_cv_score = log_loss(train_df[target_cols].values, ensemble_oof)

    for result in all_results:
        print(f"Seed {result['seed']}: CV = {result['cv_score']:.5f}")
    print(f"\nEnsemble CV Score: {ensemble_cv_score:.5f}")

    # Save ensemble OOF
    oof_df = train_df[['id']].copy()
    oof_df['winner_model_a'] = ensemble_oof[:, 0]
    oof_df['winner_model_b'] = ensemble_oof[:, 1]
    oof_df['winner_tie'] = ensemble_oof[:, 2]
    oof_df.to_csv(output_dir / 'oof_predictions_ensemble.csv', index=False)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    return ensemble_cv_score, all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Enhanced DeBERTa training')
    parser.add_argument('--config', type=str, default='configs/config_enhanced.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Train
    cv_score, results = train_multi_seed(config)
