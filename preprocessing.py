"""
Text preprocessing utilities for LLM Classification.
"""

import re
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    """
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # Normalize unicode
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Remove excessive punctuation
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    return text.strip()


def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract statistical text features for baseline models.
    """
    df = df.copy()
    
    # Length features
    df['prompt_len'] = df['prompt'].fillna('').apply(len)
    df['response_a_len'] = df['response_a'].fillna('').apply(len)
    df['response_b_len'] = df['response_b'].fillna('').apply(len)
    
    # Word counts
    df['prompt_words'] = df['prompt'].fillna('').apply(lambda x: len(str(x).split()))
    df['response_a_words'] = df['response_a'].fillna('').apply(lambda x: len(str(x).split()))
    df['response_b_words'] = df['response_b'].fillna('').apply(lambda x: len(str(x).split()))
    
    # Length ratios
    df['len_ratio_ab'] = df['response_a_len'] / (df['response_b_len'] + 1)
    df['word_ratio_ab'] = df['response_a_words'] / (df['response_b_words'] + 1)
    
    # Length differences
    df['len_diff_ab'] = df['response_a_len'] - df['response_b_len']
    df['word_diff_ab'] = df['response_a_words'] - df['response_b_words']
    
    # Code detection (rough heuristic)
    df['response_a_has_code'] = df['response_a'].fillna('').apply(
        lambda x: int('```' in str(x) or 'def ' in str(x) or 'import ' in str(x))
    )
    df['response_b_has_code'] = df['response_b'].fillna('').apply(
        lambda x: int('```' in str(x) or 'def ' in str(x) or 'import ' in str(x))
    )
    
    # List detection
    df['response_a_has_list'] = df['response_a'].fillna('').apply(
        lambda x: int(bool(re.search(r'^\s*[-*\d+\.]\s', str(x), re.MULTILINE)))
    )
    df['response_b_has_list'] = df['response_b'].fillna('').apply(
        lambda x: int(bool(re.search(r'^\s*[-*\d+\.]\s', str(x), re.MULTILINE)))
    )
    
    return df


def create_tfidf_features(
    train_texts: List[str],
    test_texts: Optional[List[str]] = None,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 3,
    max_df: float = 0.95
) -> Tuple[csr_matrix, Optional[csr_matrix], TfidfVectorizer]:
    """
    Create TF-IDF features from text.
    
    Returns:
        train_tfidf: TF-IDF matrix for training
        test_tfidf: TF-IDF matrix for test (if provided)
        vectorizer: Fitted TF-IDF vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        strip_accents='unicode',
        lowercase=True,
        analyzer='word',
        token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b',  # Words with 2+ chars
        sublinear_tf=True
    )
    
    train_tfidf = vectorizer.fit_transform(train_texts)
    
    test_tfidf = None
    if test_texts is not None:
        test_tfidf = vectorizer.transform(test_texts)
    
    return train_tfidf, test_tfidf, vectorizer


def prepare_baseline_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    tfidf_max_features: int = 5000
) -> Tuple[csr_matrix, Optional[csr_matrix], List[str]]:
    """
    Prepare all features for baseline model.
    
    Combines:
    - TF-IDF on prompt
    - TF-IDF on response_a
    - TF-IDF on response_b
    - TF-IDF on concatenated text
    - Statistical features
    """
    # Extract statistical features
    train_df = extract_text_features(train_df)
    if test_df is not None:
        test_df = extract_text_features(test_df)
    
    # Statistical feature columns
    stat_cols = [
        'prompt_len', 'response_a_len', 'response_b_len',
        'prompt_words', 'response_a_words', 'response_b_words',
        'len_ratio_ab', 'word_ratio_ab', 'len_diff_ab', 'word_diff_ab',
        'response_a_has_code', 'response_b_has_code',
        'response_a_has_list', 'response_b_has_list'
    ]
    
    # Get statistical features as matrices
    train_stats = csr_matrix(train_df[stat_cols].fillna(0).values)
    test_stats = None
    if test_df is not None:
        test_stats = csr_matrix(test_df[stat_cols].fillna(0).values)
    
    # Create combined text for TF-IDF
    train_combined = (
        train_df['prompt'].fillna('') + ' [SEP] ' +
        train_df['response_a'].fillna('') + ' [SEP] ' +
        train_df['response_b'].fillna('')
    ).tolist()
    
    test_combined = None
    if test_df is not None:
        test_combined = (
            test_df['prompt'].fillna('') + ' [SEP] ' +
            test_df['response_a'].fillna('') + ' [SEP] ' +
            test_df['response_b'].fillna('')
        ).tolist()
    
    # Create TF-IDF features
    train_tfidf, test_tfidf, _ = create_tfidf_features(
        train_combined,
        test_combined,
        max_features=tfidf_max_features
    )
    
    # Combine all features
    train_features = hstack([train_stats, train_tfidf])
    test_features = None
    if test_df is not None and test_tfidf is not None:
        test_features = hstack([test_stats, test_tfidf])
    
    feature_names = stat_cols + [f'tfidf_{i}' for i in range(tfidf_max_features)]
    
    return train_features, test_features, feature_names


def augment_by_swap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data augmentation by swapping response_a and response_b.
    
    For training: swaps responses AND flips labels accordingly.
    """
    df_aug = df.copy()
    
    # Swap responses
    df_aug['response_a'], df_aug['response_b'] = (
        df['response_b'].values.copy(),
        df['response_a'].values.copy()
    )
    
    # Swap labels if present
    if 'winner_model_a' in df_aug.columns:
        df_aug['winner_model_a'], df_aug['winner_model_b'] = (
            df['winner_model_b'].values.copy(),
            df['winner_model_a'].values.copy()
        )
        # winner_tie stays the same
    
    # Mark as augmented
    df_aug['is_augmented'] = True
    df['is_augmented'] = False
    
    # Combine
    combined = pd.concat([df, df_aug], ignore_index=True)
    
    return combined
