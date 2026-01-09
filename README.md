# LLM Classification Finetuning - Kaggle Competition

Predict which chatbot response humans prefer in head-to-head LLM battles.

## Competition Overview

- **Task**: 3-class classification (winner_model_a, winner_model_b, tie)
- **Metric**: Log Loss
- **Data**: Chatbot Arena conversations with human preference labels

## Project Structure

```
llm-kaggle-comp/
├── configs/
│   └── config.yaml          # Training hyperparameters
├── data/
│   ├── train.csv            # Download from Kaggle
│   └── test.csv             # Download from Kaggle
├── models/                  # Saved model checkpoints
├── notebooks/               # Exploration notebooks
├── scripts/
│   ├── run_baseline.py      # Run TF-IDF baseline
│   └── run_training.py      # Run DeBERTa training
├── src/
│   ├── data_loader.py       # Data loading utilities
│   ├── preprocessing.py     # Text preprocessing
│   ├── baseline_tfidf.py    # TF-IDF + LogReg baseline
│   ├── deberta_model.py     # DeBERTa model architecture
│   ├── train_deberta.py     # Training loop
│   └── inference.py         # Generate submissions
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

### 2. Download Data

```bash
kaggle competitions download -c llm-classification-finetuning
unzip llm-classification-finetuning.zip -d data/
```

### 3. Run TF-IDF Baseline

```bash
python scripts/run_baseline.py
```

Expected score: ~1.10 (establishes floor)

### 4. Train DeBERTa

```bash
python scripts/run_training.py --config configs/config.yaml
```

Expected score: ~1.02-1.05

### 5. Generate Submission

```bash
python src/inference.py --model_path models/best_model.pt --output submission.csv
```

## Approach Hierarchy

| Phase | Method | Expected LB |
|-------|--------|-------------|
| 1 | TF-IDF + LogReg | ~1.10 |
| 2 | DeBERTa-v3-large | ~1.03 |
| 3 | Multi-seed ensemble | ~1.00 |
| 4 | + QLoRA decoder models | <1.00 |

## Key Insights

1. **Input Format**: Concatenate `[CLS] prompt [SEP] response_a [SEP] response_b [SEP]`
2. **Max Length**: 1024-1536 tokens captures most examples
3. **Class Weights**: Ties are underrepresented; consider weighted loss
4. **Ensembling**: All top solutions use model ensembles

## Hardware Requirements

- **Baseline**: CPU only, <5 min
- **DeBERTa**: 1x GPU (16GB+), ~2-4 hours
- **7B Models**: 1x A100 or 2x T4 with QLoRA

## License

MIT
