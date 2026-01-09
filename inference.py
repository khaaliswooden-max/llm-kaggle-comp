"""
Inference script for generating Kaggle submissions.

Supports:
- Single model inference
- Multi-fold ensemble
- Multi-seed ensemble
- Test-time augmentation (swap responses)
"""

import os
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from data_loader import get_test_dataloader, PreferenceDataset
from deberta_model import PreferenceClassifier


def load_model(
    checkpoint_path: str,
    device: torch.device
) -> PreferenceClassifier:
    """
    Load model from checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = PreferenceClassifier(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels'],
        pooling=config['model']['pooling'],
        gradient_checkpointing=False  # Disable for inference
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


@torch.no_grad()
def predict(
    model: PreferenceClassifier,
    test_loader,
    device: torch.device
) -> np.ndarray:
    """
    Generate predictions for test data.
    
    Returns:
        predictions: (n_samples, n_classes) probability array
    """
    model.eval()
    all_preds = []
    
    for batch in tqdm(test_loader, desc='Predicting'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        token_type_ids = None
        if 'token_type_ids' in batch:
            token_type_ids = batch['token_type_ids'].to(device)
        
        probs = model.predict_proba(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        all_preds.append(probs.cpu().numpy())
    
    return np.vstack(all_preds)


def predict_with_tta(
    model: PreferenceClassifier,
    test_df: pd.DataFrame,
    tokenizer,
    config: dict,
    device: torch.device
) -> np.ndarray:
    """
    Predict with test-time augmentation (swap responses).
    
    Averages predictions from original and swapped versions.
    """
    # Original predictions
    test_loader = get_test_dataloader(
        test_df=test_df,
        tokenizer=tokenizer,
        batch_size=config['inference']['batch_size'],
        max_length=config['model']['max_length']
    )
    preds_original = predict(model, test_loader, device)
    
    # Swapped predictions
    test_df_swapped = test_df.copy()
    test_df_swapped['response_a'], test_df_swapped['response_b'] = (
        test_df['response_b'].values.copy(),
        test_df['response_a'].values.copy()
    )
    
    test_loader_swapped = get_test_dataloader(
        test_df=test_df_swapped,
        tokenizer=tokenizer,
        batch_size=config['inference']['batch_size'],
        max_length=config['model']['max_length']
    )
    preds_swapped = predict(model, test_loader_swapped, device)
    
    # Swap the predictions back (a <-> b)
    preds_swapped_corrected = preds_swapped[:, [1, 0, 2]]
    
    # Average
    preds_final = (preds_original + preds_swapped_corrected) / 2
    
    return preds_final


def ensemble_predictions(
    predictions_list: List[np.ndarray],
    method: str = 'arithmetic'
) -> np.ndarray:
    """
    Ensemble multiple predictions.
    
    Args:
        predictions_list: List of prediction arrays
        method: 'arithmetic' or 'geometric' averaging
    
    Returns:
        Ensembled predictions
    """
    preds_stack = np.stack(predictions_list, axis=0)
    
    if method == 'arithmetic':
        ensemble_preds = np.mean(preds_stack, axis=0)
    elif method == 'geometric':
        ensemble_preds = np.exp(np.mean(np.log(preds_stack + 1e-10), axis=0))
        # Normalize
        ensemble_preds = ensemble_preds / ensemble_preds.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    return ensemble_preds


def run_inference(
    model_dir: str,
    test_path: str,
    output_path: str,
    use_tta: bool = False,
    ensemble_method: str = 'arithmetic'
) -> pd.DataFrame:
    """
    Run inference and generate submission.
    
    Args:
        model_dir: Directory containing model checkpoints
        test_path: Path to test data
        output_path: Path to save submission
        use_tta: Whether to use test-time augmentation
        ensemble_method: How to ensemble fold predictions
    
    Returns:
        submission: Submission dataframe
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_dir = Path(model_dir)
    
    # Load config
    config_path = model_dir / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    print(f"Test shape: {test_df.shape}")
    
    # Find all model checkpoints
    checkpoint_paths = sorted(model_dir.glob('model_fold*.pt'))
    print(f"Found {len(checkpoint_paths)} fold checkpoints")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Generate predictions for each fold
    all_predictions = []
    
    for checkpoint_path in checkpoint_paths:
        print(f"\nLoading {checkpoint_path.name}...")
        model, _ = load_model(str(checkpoint_path), device)
        
        if use_tta:
            preds = predict_with_tta(model, test_df, tokenizer, config, device)
        else:
            test_loader = get_test_dataloader(
                test_df=test_df,
                tokenizer=tokenizer,
                batch_size=config['inference']['batch_size'],
                max_length=config['model']['max_length']
            )
            preds = predict(model, test_loader, device)
        
        all_predictions.append(preds)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Ensemble predictions
    print(f"\nEnsembling {len(all_predictions)} predictions using {ensemble_method} averaging...")
    final_preds = ensemble_predictions(all_predictions, method=ensemble_method)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'winner_model_a': final_preds[:, 0],
        'winner_model_b': final_preds[:, 1],
        'winner_tie': final_preds[:, 2]
    })
    
    # Validate probabilities sum to 1
    prob_sums = submission[['winner_model_a', 'winner_model_b', 'winner_tie']].sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probabilities don't sum to 1!"
    
    # Save
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(submission.head(10))
    
    # Print statistics
    print("\nPrediction Statistics:")
    print(f"Mean winner_model_a: {final_preds[:, 0].mean():.4f}")
    print(f"Mean winner_model_b: {final_preds[:, 1].mean():.4f}")
    print(f"Mean winner_tie: {final_preds[:, 2].mean():.4f}")
    
    return submission


def run_single_model_inference(
    model_path: str,
    test_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Run inference with a single model checkpoint.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, config = load_model(model_path, device)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    print(f"Test shape: {test_df.shape}")
    
    # Create dataloader
    test_loader = get_test_dataloader(
        test_df=test_df,
        tokenizer=tokenizer,
        batch_size=config['inference']['batch_size'],
        max_length=config['model']['max_length']
    )
    
    # Generate predictions
    print("Generating predictions...")
    predictions = predict(model, test_loader, device)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'winner_model_a': predictions[:, 0],
        'winner_model_b': predictions[:, 1],
        'winner_tie': predictions[:, 2]
    })
    
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")
    print(submission.head())
    
    return submission


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate predictions')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory containing model checkpoints')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to single model checkpoint (overrides model-dir)')
    parser.add_argument('--test', type=str, default='data/test.csv',
                        help='Path to test data')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output submission path')
    parser.add_argument('--tta', action='store_true',
                        help='Use test-time augmentation')
    parser.add_argument('--ensemble-method', type=str, default='arithmetic',
                        choices=['arithmetic', 'geometric'],
                        help='Ensemble method')
    
    args = parser.parse_args()
    
    if args.model_path:
        # Single model inference
        submission = run_single_model_inference(
            model_path=args.model_path,
            test_path=args.test,
            output_path=args.output
        )
    else:
        # Multi-fold ensemble inference
        submission = run_inference(
            model_dir=args.model_dir,
            test_path=args.test,
            output_path=args.output,
            use_tta=args.tta,
            ensemble_method=args.ensemble_method
        )
