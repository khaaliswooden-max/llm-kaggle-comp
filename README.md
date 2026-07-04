# LLM Classification Finetuning — Kaggle Competition

Predict which chatbot response humans prefer in head-to-head LLM battles from the
[Chatbot Arena](https://lmarena.ai/) dataset.

## Competition Overview

- **Task**: 3-class classification — `winner_model_a`, `winner_model_b`, `winner_tie`
- **Metric**: Multi-class log loss (lower is better)
- **Data**: Chatbot Arena conversations with human preference labels
- **Submission**: A CSV of per-row probabilities for the three outcomes

## Project Structure

All modules live at the repository root:

```
llm-kaggle-comp/
├── config.yaml            # Training hyperparameters
├── data_loader.py         # Dataset, folds, and dataloaders
├── preprocessing.py       # Text cleaning + TF-IDF feature helpers
├── baseline_tfidf.py      # TF-IDF + LogReg baseline
├── deberta_model.py       # DeBERTa architecture (pooling, AWP)
├── train_deberta.py       # Cross-validation training loop
├── inference.py           # Prediction, TTA, ensembling
├── run_baseline.py        # Entry point: run the TF-IDF baseline
├── run_training.py        # Entry point: run DeBERTa training
├── deberta_training.ipynb # Exploration / Kaggle notebook
├── __init__.py            # Package exports
├── requirements.txt
├── train_sample.csv       # Small sample of the training data
├── test.csv               # Test set
└── sample_submission.csv  # Submission format reference
```

> **Note:** The scripts default to `data/train.csv`, `data/test.csv`, and
> `configs/config.yaml` paths. Either pass explicit paths via the CLI flags shown
> below, or place your Kaggle downloads under a `data/` directory.

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

A `train_sample.csv` and `test.csv` are included so you can smoke-test the
pipeline before downloading the full dataset.

### 3. Run the TF-IDF Baseline

```bash
python run_baseline.py --train train_sample.csv --test test.csv
```

Expected score: ~1.10 (establishes the floor).

### 4. Train DeBERTa

```bash
python run_training.py --config config.yaml
```

Expected score: ~1.02–1.05. Override any hyperparameter from the CLI:

```bash
python run_training.py --config config.yaml --seed 1337 --epochs 2 --lr 1e-5
```

### 5. Generate a Submission

```bash
python inference.py --model_path models/best_model.pt --output submission.csv
```

## Data Format

Each row contains a shared `prompt` and two competing responses:

| Column | Description |
|--------|-------------|
| `id` | Unique row identifier |
| `prompt` | User prompt (may be a JSON array of multi-turn messages) |
| `response_a` | Response from model A |
| `response_b` | Response from model B |
| `winner_model_a` / `winner_model_b` / `winner_tie` | One-hot labels (train only) |

Prompts and responses may be stored as JSON arrays for multi-turn conversations;
`data_loader.parse_conversation` flattens these into a single string joined by
`[TURN]` markers.

## Approach Hierarchy

| Phase | Method | Expected LB |
|-------|--------|-------------|
| 1 | TF-IDF + LogReg | ~1.10 |
| 2 | DeBERTa-v3-large | ~1.03 |
| 3 | Multi-seed ensemble | ~1.00 |
| 4 | + QLoRA decoder models | <1.00 |

## Key Insights

1. **Input Format**: Concatenate prompt + both responses with segment markers
   (`[PROMPT] … [RESPONSE A] … [RESPONSE B] …`).
2. **Max Length**: 1024–1536 tokens captures most examples.
3. **Class Weights**: Ties are underrepresented; a weighted loss helps
   (see `class_weights` in `config.yaml`).
4. **Ensembling**: All strong solutions average across multiple seeds/models
   (geometric averaging by default).

## Hardware Requirements

| Stage | Requirement | Runtime |
|-------|-------------|---------|
| Baseline | CPU only | <5 min |
| DeBERTa | 1× GPU (16 GB+) | ~2–4 hours |
| 7B models | 1× A100 or 2× T4 with QLoRA | Varies |

## Configuration

All training behavior is controlled by `config.yaml` — model name, max sequence
length, learning rate, cross-validation folds, class weights, and the ensemble
seed list. See the file for the full annotated set of options.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Released under the [MIT License](LICENSE).
