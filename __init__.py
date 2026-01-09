"""
LLM Classification Finetuning - Source Module
"""

from .data_loader import (
    PreferenceDataset,
    load_data,
    create_folds,
    get_dataloaders,
    get_test_dataloader
)

from .preprocessing import (
    clean_text,
    extract_text_features,
    create_tfidf_features,
    prepare_baseline_features,
    augment_by_swap
)

from .baseline_tfidf import TFIDFBaseline, run_baseline

from .deberta_model import (
    PreferenceClassifier,
    get_model_and_tokenizer,
    MeanPooling,
    AttentionPooling,
    AWP
)

from .inference import (
    load_model,
    predict,
    predict_with_tta,
    ensemble_predictions,
    run_inference,
    run_single_model_inference
)

__all__ = [
    # Data
    'PreferenceDataset',
    'load_data',
    'create_folds',
    'get_dataloaders',
    'get_test_dataloader',
    
    # Preprocessing
    'clean_text',
    'extract_text_features',
    'create_tfidf_features',
    'prepare_baseline_features',
    'augment_by_swap',
    
    # Baseline
    'TFIDFBaseline',
    'run_baseline',
    
    # Model
    'PreferenceClassifier',
    'get_model_and_tokenizer',
    'MeanPooling',
    'AttentionPooling',
    'AWP',
    
    # Inference
    'load_model',
    'predict',
    'predict_with_tta',
    'ensemble_predictions',
    'run_inference',
    'run_single_model_inference',
]
