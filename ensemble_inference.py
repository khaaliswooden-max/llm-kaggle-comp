"""
Ensemble inference script for combining multiple models.

Supports:
- Multi-fold DeBERTa models
- Multi-fold QLoRA models
- Weighted averaging
- Test-time augmentation
- Optimized ensemble weights via CV
"""

import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.metrics import log_loss

import torch
from transformers import AutoTokenizer

from data_loader import get_test_dataloader
from deberta_model import PreferenceClassifier
from inference import predict, predict_with_tta, load_model


def ensemble_predictions(
    predictions_list: List[np.ndarray],
    weights: Optional[List[float]] = None,
    method: str = 'arithmetic'
) -> np.ndarray:
    """
    Ensemble multiple predictions.

    Args:
        predictions_list: List of (n_samples, n_classes) arrays
        weights: Optional weights for each prediction
        method: 'arithmetic' or 'geometric'

    Returns:
        Ensembled predictions
    """
    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)

    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize

    preds_stack = np.stack(predictions_list, axis=0)

    if method == 'arithmetic':
        ensemble_preds = np.average(preds_stack, axis=0, weights=weights)
    elif method == 'geometric':
        # Weighted geometric mean
        log_preds = np.log(preds_stack + 1e-10)
        weighted_log = np.average(log_preds, axis=0, weights=weights)
        ensemble_preds = np.exp(weighted_log)
        # Normalize to sum to 1
        ensemble_preds = ensemble_preds / ensemble_preds.sum(axis=1, keepdims=True)
    elif method == 'rank':
        # Rank averaging
        from scipy.stats import rankdata
        rank_preds = []
        for preds in preds_stack:
            ranked = np.apply_along_axis(lambda x: rankdata(x) / len(x), 1, preds)
            rank_preds.append(ranked)
        rank_stack = np.stack(rank_preds, axis=0)
        ensemble_preds = np.average(rank_stack, axis=0, weights=weights)
        # Normalize
        ensemble_preds = ensemble_preds / ensemble_preds.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown method: {method}")

    return ensemble_preds


def optimize_ensemble_weights(
    oof_predictions: List[np.ndarray],
    targets: np.ndarray,
    method: str = 'arithmetic'
) -> Tuple[List[float], float]:
    """
    Find optimal ensemble weights using OOF predictions.

    Uses scipy minimize to find weights that minimize log loss.

    Args:
        oof_predictions: List of OOF prediction arrays
        targets: Ground truth labels (n_samples, n_classes)
        method: Ensemble method

    Returns:
        optimal_weights: List of weights
        best_score: Best log loss achieved
    """
    n_models = len(oof_predictions)

    def objective(weights):
        weights = np.abs(weights)  # Ensure positive
        weights = weights / weights.sum()  # Normalize
        ensemble = ensemble_predictions(oof_predictions, weights.tolist(), method)
        return log_loss(targets, ensemble)

    # Initial weights (equal)
    x0 = np.ones(n_models) / n_models

    # Optimize
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 1000}
    )

    optimal_weights = np.abs(result.x)
    optimal_weights = optimal_weights / optimal_weights.sum()

    return optimal_weights.tolist(), result.fun


def load_oof_predictions(model_dirs: List[str]) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """
    Load OOF predictions from model directories.

    Args:
        model_dirs: List of directories containing OOF predictions

    Returns:
        oof_predictions: List of OOF prediction arrays
        targets: Ground truth targets (from first valid file)
    """
    oof_predictions = []
    targets = None

    for model_dir in model_dirs:
        model_dir = Path(model_dir)

        # Try different OOF file patterns
        oof_files = list(model_dir.glob('oof_*.csv')) + list(model_dir.glob('*oof*.csv'))

        if not oof_files:
            print(f"Warning: No OOF files found in {model_dir}")
            continue

        # Use ensemble OOF if available, otherwise first seed
        ensemble_file = model_dir / 'oof_predictions_ensemble.csv'
        if ensemble_file.exists():
            oof_file = ensemble_file
        else:
            oof_file = sorted(oof_files)[0]

        print(f"Loading OOF from: {oof_file}")
        oof_df = pd.read_csv(oof_file)

        pred_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
        oof_predictions.append(oof_df[pred_cols].values)

        # Load targets if not already loaded
        if targets is None:
            train_folds_file = model_dir / 'train_folds.csv'
            if train_folds_file.exists():
                train_df = pd.read_csv(train_folds_file)
                targets = train_df[pred_cols].values

    return oof_predictions, targets


class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models.
    """

    def __init__(
        self,
        deberta_dirs: List[str] = None,
        qlora_dirs: List[str] = None,
        weights: Optional[Dict[str, float]] = None,
        ensemble_method: str = 'arithmetic'
    ):
        self.deberta_dirs = deberta_dirs or []
        self.qlora_dirs = qlora_dirs or []
        self.weights = weights
        self.ensemble_method = ensemble_method
        self.models = []

    def load_deberta_models(self, device: torch.device):
        """Load all DeBERTa models."""
        for model_dir in self.deberta_dirs:
            model_dir = Path(model_dir)
            checkpoint_paths = sorted(model_dir.glob('model_fold*.pt'))

            for checkpoint_path in checkpoint_paths:
                print(f"Loading {checkpoint_path.name}...")
                model, config = load_model(str(checkpoint_path), device)
                tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])

                self.models.append({
                    'type': 'deberta',
                    'model': model,
                    'tokenizer': tokenizer,
                    'config': config
                })

    def predict(
        self,
        test_df: pd.DataFrame,
        device: torch.device,
        use_tta: bool = True
    ) -> np.ndarray:
        """
        Generate ensemble predictions.

        Args:
            test_df: Test dataframe
            device: torch device
            use_tta: Whether to use test-time augmentation

        Returns:
            predictions: (n_samples, 3) probability array
        """
        all_predictions = []

        for model_info in tqdm(self.models, desc='Generating predictions'):
            if model_info['type'] == 'deberta':
                model = model_info['model']
                tokenizer = model_info['tokenizer']
                config = model_info['config']

                if use_tta:
                    preds = predict_with_tta(
                        model, test_df, tokenizer, config, device
                    )
                else:
                    test_loader = get_test_dataloader(
                        test_df=test_df,
                        tokenizer=tokenizer,
                        batch_size=config['inference']['batch_size'],
                        max_length=config['model']['max_length']
                    )
                    preds = predict(model, test_loader, device)

                all_predictions.append(preds)

            # Add QLoRA prediction logic here if needed

        # Ensemble
        if self.weights:
            weights = list(self.weights.values())
        else:
            weights = None

        final_preds = ensemble_predictions(
            all_predictions,
            weights=weights,
            method=self.ensemble_method
        )

        return final_preds


def create_submission(
    predictions: np.ndarray,
    test_df: pd.DataFrame,
    output_path: str
) -> pd.DataFrame:
    """Create and save submission file."""
    submission = pd.DataFrame({
        'id': test_df['id'],
        'winner_model_a': predictions[:, 0],
        'winner_model_b': predictions[:, 1],
        'winner_tie': predictions[:, 2]
    })

    # Validate
    prob_sums = submission[['winner_model_a', 'winner_model_b', 'winner_tie']].sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probabilities don't sum to 1!"

    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to: {output_path}")

    # Statistics
    print("\nPrediction Statistics:")
    print(f"  Mean winner_model_a: {predictions[:, 0].mean():.4f}")
    print(f"  Mean winner_model_b: {predictions[:, 1].mean():.4f}")
    print(f"  Mean winner_tie: {predictions[:, 2].mean():.4f}")

    return submission


def main():
    parser = argparse.ArgumentParser(description='Ensemble inference')
    parser.add_argument('--deberta-dirs', nargs='+', default=['models'],
                        help='DeBERTa model directories')
    parser.add_argument('--qlora-dirs', nargs='+', default=[],
                        help='QLoRA model directories')
    parser.add_argument('--test', type=str, default='data/test.csv',
                        help='Test data path')
    parser.add_argument('--output', type=str, default='submission_ensemble.csv',
                        help='Output path')
    parser.add_argument('--method', type=str, default='arithmetic',
                        choices=['arithmetic', 'geometric', 'rank'],
                        help='Ensemble method')
    parser.add_argument('--optimize-weights', action='store_true',
                        help='Optimize ensemble weights using OOF')
    parser.add_argument('--no-tta', action='store_true',
                        help='Disable test-time augmentation')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Optimize weights if requested
    weights = None
    if args.optimize_weights:
        print("\nOptimizing ensemble weights using OOF predictions...")
        all_dirs = args.deberta_dirs + args.qlora_dirs
        oof_predictions, targets = load_oof_predictions(all_dirs)

        if targets is not None and len(oof_predictions) > 1:
            weights, best_score = optimize_ensemble_weights(
                oof_predictions, targets, args.method
            )
            print(f"Optimal weights: {weights}")
            print(f"Best OOF score: {best_score:.5f}")
        else:
            print("Could not optimize weights - using equal weights")

    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv(args.test)
    print(f"Test shape: {test_df.shape}")

    # Create ensemble predictor
    predictor = EnsemblePredictor(
        deberta_dirs=args.deberta_dirs,
        qlora_dirs=args.qlora_dirs,
        weights=dict(enumerate(weights)) if weights else None,
        ensemble_method=args.method
    )

    # Load models
    print("\nLoading models...")
    predictor.load_deberta_models(device)

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = predictor.predict(
        test_df=test_df,
        device=device,
        use_tta=not args.no_tta
    )

    # Create submission
    create_submission(predictions, test_df, args.output)


if __name__ == '__main__':
    main()
