"""
Training script for DeBERTa preference classifier.

Supports:
- Multi-fold cross-validation
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Early stopping
- Model checkpointing
"""

import os
import gc
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    get_linear_schedule_with_warmup
)
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import log_loss

from data_loader import (
    load_data,
    create_folds,
    get_dataloaders,
    PreferenceDataset
)
from deberta_model import get_model_and_tokenizer, AWP


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    """
    Get optimizer with layer-wise learning rate decay.
    """
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    
    # Group parameters
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': config['training']['weight_decay'],
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            'weight_decay': 0.0,
        }
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config['training']['learning_rate'],
        eps=1e-8
    )
    
    return optimizer


def get_scheduler(optimizer, config: dict, num_training_steps: int):
    """Get learning rate scheduler."""
    warmup_steps = int(num_training_steps * config['training']['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return scheduler


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    config: dict,
    device: torch.device,
    awp: Optional[AWP] = None
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    grad_accum_steps = config['training']['gradient_accumulation_steps']
    max_grad_norm = config['training']['max_grad_norm']
    use_fp16 = config['training']['fp16']
    
    progress_bar = tqdm(train_loader, desc='Training')
    
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
            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            loss = loss / grad_accum_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # AWP attack (optional)
        if awp is not None and step % grad_accum_steps == 0:
            awp.attack_step()
            with autocast(enabled=use_fp16):
                _, adv_loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                adv_loss = adv_loss / grad_accum_steps
            scaler.scale(adv_loss).backward()
            awp.restore()
        
        # Gradient accumulation
        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
        
        progress_bar.set_postfix({
            'loss': total_loss / num_batches,
            'lr': scheduler.get_last_lr()[0]
        })
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Validate model.
    
    Returns:
        val_loss: Validation log loss
        predictions: Predicted probabilities
        targets: Ground truth labels
    """
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
    
    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    
    # Calculate log loss
    val_loss = log_loss(targets, predictions)
    
    return val_loss, predictions, targets


def train_fold(
    fold: int,
    train_df: pd.DataFrame,
    config: dict,
    device: torch.device,
    output_dir: Path
) -> Tuple[float, np.ndarray]:
    """
    Train a single fold.
    
    Returns:
        best_val_loss: Best validation loss for this fold
        oof_preds: Out-of-fold predictions
    """
    print(f"\n{'='*50}")
    print(f"Training Fold {fold + 1}")
    print(f"{'='*50}")
    
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
    
    # Optimizer and scheduler
    num_training_steps = (
        len(train_loader) // config['training']['gradient_accumulation_steps']
    ) * config['training']['epochs']
    
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, num_training_steps)
    scaler = GradScaler()
    
    # AWP (optional)
    awp = None
    # awp = AWP(model, optimizer)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    oof_preds = None
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            config=config,
            device=device,
            awp=awp
        )
        
        # Validate
        val_loss, predictions, targets = validate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            oof_preds = predictions
            
            # Save checkpoint
            checkpoint_path = output_dir / f'model_fold{fold}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
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


def train(config: dict):
    """
    Main training function.
    
    Runs k-fold cross-validation and saves OOF predictions.
    """
    # Set seed
    set_seed(config['training']['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output directory
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
        seed=config['training']['seed'],
        stratified=config['cv']['stratified']
    )
    
    # Save fold assignments
    train_df.to_csv(output_dir / 'train_folds.csv', index=False)
    
    # Train each fold
    fold_scores = []
    all_oof_preds = np.zeros((len(train_df), config['model']['num_labels']))
    
    for fold in range(config['cv']['n_folds']):
        fold_loss, oof_preds = train_fold(
            fold=fold,
            train_df=train_df,
            config=config,
            device=device,
            output_dir=output_dir
        )
        fold_scores.append(fold_loss)
        
        # Store OOF predictions
        val_idx = train_df[train_df['fold'] == fold].index
        all_oof_preds[val_idx] = oof_preds
    
    # Calculate overall CV score
    target_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
    cv_score = log_loss(train_df[target_cols].values, all_oof_preds)
    
    print(f"\n{'='*50}")
    print(f"Cross-Validation Results")
    print(f"{'='*50}")
    for fold, score in enumerate(fold_scores):
        print(f"Fold {fold + 1}: {score:.5f}")
    print(f"Mean: {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
    print(f"Overall CV Log Loss: {cv_score:.5f}")
    
    # Save OOF predictions
    oof_df = train_df[['id']].copy()
    oof_df['winner_model_a'] = all_oof_preds[:, 0]
    oof_df['winner_model_b'] = all_oof_preds[:, 1]
    oof_df['winner_tie'] = all_oof_preds[:, 2]
    oof_df.to_csv(output_dir / 'oof_predictions.csv', index=False)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    return cv_score, fold_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeBERTa classifier')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Train
    cv_score, fold_scores = train(config)
