#!/usr/bin/env python
"""
Run TF-IDF baseline model.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --train data/train.csv --test data/test.csv
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from baseline_tfidf import run_baseline


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
        test_path=args.test if Path(args.test).exists() else None,
        output_path=args.output,
        tfidf_max_features=args.tfidf_features
    )
    
    print(f"\nFinal CV Score: {cv_score:.5f}")
