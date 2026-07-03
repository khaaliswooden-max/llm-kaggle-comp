# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project documentation suite: `LICENSE` (MIT), `CONTRIBUTING.md`, and
  `CHANGELOG.md`.
- Expanded `README.md` with an accurate flat-layout project map, data-format
  reference, and configuration overview.

## [0.1.0] — 2026-07-03

### Added
- TF-IDF + Logistic Regression baseline (`baseline_tfidf.py`, `run_baseline.py`).
- DeBERTa-v3 classification pipeline (`deberta_model.py`, `train_deberta.py`,
  `run_training.py`) with cross-validation, class weighting, and AWP.
- Data loading utilities with multi-turn conversation parsing (`data_loader.py`).
- Text preprocessing and feature helpers (`preprocessing.py`).
- Inference utilities with test-time augmentation and ensembling (`inference.py`).
- `config.yaml` for centralized hyperparameter management.
- Exploration notebook (`deberta_training.ipynb`).
- Sample data fixtures (`train_sample.csv`, `test.csv`, `sample_submission.csv`).

[Unreleased]: https://github.com/khaaliswooden-max/llm-kaggle-comp/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/khaaliswooden-max/llm-kaggle-comp/releases/tag/v0.1.0
