"""
Training script for QLoRA-based LLM classifiers.

Supports finetuning large decoder models (Llama, Mistral, etc.)
with 4-bit quantization and LoRA adapters.

Hardware requirements:
- 7B models: ~16GB VRAM (single GPU)
- 13B models: ~24GB VRAM or 2x 16GB GPUs
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
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import log_loss
from transformers import get_linear_schedule_with_warmup

from data_loader import load_data, create_folds
from qlora_model import (
    get_qlora_model_and_tokenizer,
    get_qlora_dataloaders,
    QLoRAClassifier
)
from training_utils import (
    LabelSmoothingCrossEntropy,
    EMA,
    get_cosine_schedule_with_warmup_and_hard_restarts
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch_qlora(
    model: QLoRAClassifier,
    train_loader,
    optimizer,
    scheduler,
    config: dict,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    grad_accum_steps = config['training']['gradient_accumulation_steps']
    max_grad_norm = config['training']['max_grad_norm']
    use_fp16 = config['training'].get('fp16', False)  # Usually False for QLoRA

    progress_bar = tqdm(train_loader, desc='Training')
    optimizer.zero_grad()

    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        if use_fp16 and scaler is not None:
            with autocast():
                logits, loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = loss / grad_accum_steps
            loss.backward()

        # Gradient accumulation step
        if (step + 1) % grad_accum_steps == 0:
            if use_fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

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
def validate_qlora(
    model: QLoRAClassifier,
    val_loader,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Validate model."""
    model.eval()

    all_preds = []
    all_targets = []

    for batch in tqdm(val_loader, desc='Validating'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        probs = model.predict_proba(input_ids, attention_mask)

        all_preds.append(probs.cpu().numpy())
        all_targets.append(labels.cpu().numpy())

    predictions = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    val_loss = log_loss(targets, predictions)

    return val_loss, predictions, targets


def train_fold_qlora(
    fold: int,
    train_df: pd.DataFrame,
    config: dict,
    device: torch.device,
    output_dir: Path,
    seed: int = 42
) -> Tuple[float, np.ndarray]:
    """Train a single fold with QLoRA."""
    print(f"\n{'='*50}")
    print(f"Training Fold {fold + 1} (Seed: {seed})")
    print(f"{'='*50}")

    set_seed(seed)

    # Split data
    train_data = train_df[train_df['fold'] != fold].reset_index(drop=True)
    val_data = train_df[train_df['fold'] == fold].reset_index(drop=True)

    print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

    # Get model and tokenizer
    model, tokenizer = get_qlora_model_and_tokenizer(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels'],
        lora_r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        use_4bit=config['quantization']['use_4bit'],
        gradient_checkpointing=config.get('gradient_checkpointing', True)
    )

    # Create dataloaders
    train_loader, val_loader = get_qlora_dataloaders(
        train_df=train_data,
        val_df=val_data,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['model']['max_length'],
        num_workers=2,  # Reduced for large models
        input_format=config['model'].get('input_format', 'instruct')
    )

    # Optimizer - only train LoRA parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    num_training_steps = (
        len(train_loader) // config['training']['gradient_accumulation_steps']
    ) * config['training']['epochs']

    scheduler = get_cosine_schedule_with_warmup_and_hard_restarts(
        optimizer,
        num_warmup_steps=int(num_training_steps * config['training']['warmup_ratio']),
        num_training_steps=num_training_steps
    )

    scaler = GradScaler() if config['training'].get('fp16', False) else None

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    oof_preds = None

    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        train_loss = train_epoch_qlora(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            device=device,
            scaler=scaler
        )

        val_loss, predictions, targets = validate_qlora(model, val_loader, device)

        print(f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            oof_preds = predictions

            # Save LoRA adapters
            checkpoint_dir = output_dir / f'qlora_fold{fold}_seed{seed}'
            model.save_pretrained(str(checkpoint_dir))

            # Save config for loading
            with open(checkpoint_dir / 'training_config.yaml', 'w') as f:
                yaml.dump(config, f)

            print(f"Saved best model to {checkpoint_dir}")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping']['patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    return best_val_loss, oof_preds


def train_qlora(config: dict):
    """Main QLoRA training function."""
    seeds = config.get('ensemble', {}).get('seeds', [42])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
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
        seed=42,
        stratified=config['cv']['stratified']
    )

    train_df.to_csv(output_dir / 'train_folds.csv', index=False)

    # Train
    all_results = []
    all_oof_preds = {}

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# Training with Seed: {seed}")
        print(f"{'#'*60}")

        seed_oof_preds = np.zeros((len(train_df), config['model']['num_labels']))
        fold_scores = []

        for fold in range(config['cv']['n_folds']):
            fold_loss, oof_preds = train_fold_qlora(
                fold=fold,
                train_df=train_df,
                config=config,
                device=device,
                output_dir=output_dir,
                seed=seed
            )
            fold_scores.append(fold_loss)

            val_idx = train_df[train_df['fold'] == fold].index
            seed_oof_preds[val_idx] = oof_preds

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

        # Save OOF predictions
        oof_df = train_df[['id']].copy()
        oof_df['winner_model_a'] = seed_oof_preds[:, 0]
        oof_df['winner_model_b'] = seed_oof_preds[:, 1]
        oof_df['winner_tie'] = seed_oof_preds[:, 2]
        oof_df.to_csv(output_dir / f'oof_qlora_seed{seed}.csv', index=False)

    # Ensemble
    print(f"\n{'='*60}")
    print("QLoRA Multi-Seed Ensemble Results")
    print(f"{'='*60}")

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
    oof_df.to_csv(output_dir / 'oof_qlora_ensemble.csv', index=False)

    with open(output_dir / 'config_qlora.yaml', 'w') as f:
        yaml.dump(config, f)

    return ensemble_cv_score, all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train QLoRA classifier')
    parser.add_argument('--config', type=str, default='configs/config_qlora.yaml',
                        help='Path to config file')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    cv_score, results = train_qlora(config)
