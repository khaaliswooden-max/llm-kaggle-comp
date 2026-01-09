"""
TF-IDF + Logistic Regression baseline for LLM Classification.

This establishes a performance floor and validates the data pipeline.
Expected score: ~1.10 log loss
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import log_loss
from scipy.special import softmax
import joblib

from preprocessing import prepare_baseline_features


class TFIDFBaseline:
    """
    TF-IDF + Logistic Regression baseline classifier.
    
    Uses multi-class logistic regression with softmax outputs.
    """
    
    def __init__(
        self,
        tfidf_max_features: int = 5000,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: Optional[str] = 'balanced',
        n_jobs: int = -1
    ):
        self.tfidf_max_features = tfidf_max_features
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            multi_class='multinomial',
            solver='lbfgs',
            n_jobs=n_jobs,
            random_state=42
        )
        self.is_fitted = False
        
    def fit(
        self,
        train_df: pd.DataFrame,
        target_cols: list = ['winner_model_a', 'winner_model_b', 'winner_tie']
    ) -> 'TFIDFBaseline':
        """
        Fit the baseline model.
        """
        # Prepare features
        self.train_features, _, self.feature_names = prepare_baseline_features(
            train_df,
            test_df=None,
            tfidf_max_features=self.tfidf_max_features
        )
        
        # Get labels (argmax of one-hot)
        y = train_df[target_cols].values.argmax(axis=1)
        
        # Fit model
        self.model.fit(self.train_features, y)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Returns:
            Array of shape (n_samples, 3) with probabilities for
            [winner_model_a, winner_model_b, winner_tie]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features using same vectorizer
        # Note: In practice, we need to save the vectorizer
        # For now, re-fit on combined data
        _, test_features, _ = prepare_baseline_features(
            pd.DataFrame(),  # Empty train
            test_df,
            tfidf_max_features=self.tfidf_max_features
        )
        
        return self.model.predict_proba(test_features)
    
    def cross_validate(
        self,
        train_df: pd.DataFrame,
        n_folds: int = 5,
        target_cols: list = ['winner_model_a', 'winner_model_b', 'winner_tie']
    ) -> Tuple[float, np.ndarray]:
        """
        Perform cross-validation and return OOF predictions.
        
        Returns:
            cv_score: Average log loss across folds
            oof_preds: Out-of-fold predictions
        """
        # Prepare features
        train_features, _, _ = prepare_baseline_features(
            train_df,
            test_df=None,
            tfidf_max_features=self.tfidf_max_features
        )
        
        # Get labels
        y_onehot = train_df[target_cols].values
        y = y_onehot.argmax(axis=1)
        
        # Cross-validation
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        oof_preds = np.zeros((len(train_df), 3))
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_features, y)):
            X_train = train_features[train_idx]
            X_val = train_features[val_idx]
            y_train = y[train_idx]
            y_val_onehot = y_onehot[val_idx]
            
            # Fit model
            fold_model = LogisticRegression(
                C=self.model.C,
                max_iter=self.model.max_iter,
                class_weight=self.model.class_weight,
                multi_class='multinomial',
                solver='lbfgs',
                n_jobs=self.model.n_jobs,
                random_state=42
            )
            fold_model.fit(X_train, y_train)
            
            # Predict
            val_preds = fold_model.predict_proba(X_val)
            oof_preds[val_idx] = val_preds
            
            # Calculate fold score
            fold_loss = log_loss(y_val_onehot, val_preds)
            fold_scores.append(fold_loss)
            
            print(f"Fold {fold + 1}: Log Loss = {fold_loss:.5f}")
        
        cv_score = np.mean(fold_scores)
        print(f"\nCV Mean Log Loss: {cv_score:.5f} (+/- {np.std(fold_scores):.5f})")
        
        return cv_score, oof_preds
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: str) -> 'TFIDFBaseline':
        """Load model from disk."""
        return joblib.load(path)


def run_baseline(
    train_path: str,
    test_path: Optional[str] = None,
    output_path: str = 'submission_baseline.csv',
    tfidf_max_features: int = 5000
) -> Tuple[float, pd.DataFrame]:
    """
    Run the complete TF-IDF baseline pipeline.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data (optional)
        output_path: Path to save submission
        tfidf_max_features: Number of TF-IDF features
    
    Returns:
        cv_score: Cross-validation log loss
        submission: Submission dataframe
    """
    print("=" * 50)
    print("TF-IDF Baseline")
    print("=" * 50)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(train_path)
    print(f"Train shape: {train_df.shape}")
    
    # Check for target columns
    target_cols = ['winner_model_a', 'winner_model_b', 'winner_tie']
    if target_cols[0] not in train_df.columns:
        # Try to infer from 'winner' column
        if 'winner' in train_df.columns:
            train_df['winner_model_a'] = (train_df['winner'] == 'model_a').astype(int)
            train_df['winner_model_b'] = (train_df['winner'] == 'model_b').astype(int)
            train_df['winner_tie'] = (train_df['winner'] == 'tie').astype(int)
        else:
            raise ValueError("Could not find target columns")
    
    # Initialize model
    model = TFIDFBaseline(tfidf_max_features=tfidf_max_features)
    
    # Cross-validation
    print("\nRunning cross-validation...")
    cv_score, oof_preds = model.cross_validate(train_df)
    
    # Generate submission if test data provided
    submission = None
    if test_path and Path(test_path).exists():
        print("\nGenerating test predictions...")
        test_df = pd.read_csv(test_path)
        print(f"Test shape: {test_df.shape}")
        
        # Fit on full training data
        model.fit(train_df)
        
        # Predict
        # Note: Need to handle the feature preparation properly
        # For this baseline, we'll re-prepare features
        train_features, test_features, _ = prepare_baseline_features(
            train_df,
            test_df,
            tfidf_max_features=tfidf_max_features
        )
        
        # Refit model on all training data
        y = train_df[target_cols].values.argmax(axis=1)
        model.model.fit(train_features, y)
        
        # Predict probabilities
        test_preds = model.model.predict_proba(test_features)
        
        # Create submission
        submission = pd.DataFrame({
            'id': test_df['id'],
            'winner_model_a': test_preds[:, 0],
            'winner_model_b': test_preds[:, 1],
            'winner_tie': test_preds[:, 2]
        })
        
        submission.to_csv(output_path, index=False)
        print(f"\nSubmission saved to: {output_path}")
        print(submission.head())
    
    return cv_score, submission


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run TF-IDF baseline')
    parser.add_argument('--train', type=str, default='data/train.csv',
                        help='Path to training data')
    parser.add_argument('--test', type=str, default='data/test.csv',
                        help='Path to test data')
    parser.add_argument('--output', type=str, default='submission_baseline.csv',
                        help='Output submission path')
    parser.add_argument('--tfidf-features', type=int, default=5000,
                        help='Number of TF-IDF features')
    
    args = parser.parse_args()
    
    cv_score, submission = run_baseline(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        tfidf_max_features=args.tfidf_features
    )
