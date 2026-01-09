#!/usr/bin/env python
"""
Run DeBERTa training pipeline.

Usage:
    python scripts/run_training.py
    python scripts/run_training.py --config configs/config.yaml
    python scripts/run_training.py --config configs/config.yaml --seed 1337
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from train_deberta import train, set_seed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train DeBERTa classifier')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--folds', type=int, default=None,
                        help='Override number of folds')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model name')
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        # Use default config
        print(f"Config not found at {config_path}, using defaults")
        config = {
            'data': {
                'train_path': 'data/train.csv',
                'test_path': 'data/test.csv',
                'output_dir': 'models',
                'submission_path': 'submission.csv'
            },
            'model': {
                'name': 'microsoft/deberta-v3-large',
                'num_labels': 3,
                'max_length': 1024,
                'pooling': 'mean'
            },
            'training': {
                'seed': 42,
                'epochs': 3,
                'batch_size': 4,
                'gradient_accumulation_steps': 4,
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'warmup_ratio': 0.1,
                'max_grad_norm': 1.0,
                'fp16': True
            },
            'cv': {
                'n_folds': 5,
                'stratified': True
            },
            'early_stopping': {
                'patience': 3,
                'min_delta': 0.001
            },
            'inference': {
                'batch_size': 16
            }
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Apply overrides
    if args.seed is not None:
        config['training']['seed'] = args.seed
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.lr is not None:
        config['training']['learning_rate'] = args.lr
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.folds is not None:
        config['cv']['n_folds'] = args.folds
    if args.model is not None:
        config['model']['name'] = args.model
    
    print("Configuration:")
    print("-" * 40)
    for section, values in config.items():
        print(f"{section}:")
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"  {k}: {v}")
        else:
            print(f"  {values}")
    print("-" * 40)
    
    # Run training
    cv_score, fold_scores = train(config)
    
    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"CV Score: {cv_score:.5f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
