# LLM Classification Finetuning - Kaggle Competition

Predict which chatbot response humans prefer in head-to-head LLM battles.

## Competition Overview

| Aspect | Details |
|--------|---------|
| **Task** | 3-class classification (winner_model_a, winner_model_b, winner_tie) |
| **Metric** | Log Loss (lower is better) |
| **Data** | Chatbot Arena conversations with human preference labels |

## Project Structure

```
llm-kaggle-comp/
├── configs/
│   ├── config.yaml           # Basic DeBERTa config
│   ├── config_enhanced.yaml  # Enhanced config with all techniques
│   └── config_qlora.yaml     # QLoRA config for large models
├── data/
│   ├── train.csv             # Download from Kaggle
│   ├── test.csv              # Download from Kaggle
│   └── create_sample_data.py # Generate sample data for testing
├── models/                   # Saved DeBERTa checkpoints
├── models_qlora/             # Saved QLoRA adapters
├──
│── data_loader.py            # Data loading with multiple formats
│── preprocessing.py          # Text preprocessing & TF-IDF
│── baseline_tfidf.py         # TF-IDF + LogReg baseline
│── deberta_model.py          # DeBERTa model architecture
│── train_deberta.py          # Basic DeBERTa training
│── train_enhanced.py         # Enhanced training with all techniques
│── training_utils.py         # LLRD, EMA, FGM, loss functions
│── qlora_model.py            # QLoRA model for large LLMs
│── train_qlora.py            # QLoRA training script
│── inference.py              # Single model inference
│── ensemble_inference.py     # Multi-model ensemble inference
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

Or create sample data for testing:
```bash
python data/create_sample_data.py
```

### 3. Run TF-IDF Baseline

```bash
python baseline_tfidf.py --train data/train.csv --test data/test.csv
```

Expected score: ~1.10

### 4. Train DeBERTa (Basic)

```bash
python train_deberta.py --config configs/config.yaml
```

Expected score: ~1.02-1.05

### 5. Train DeBERTa (Enhanced with All Techniques)

```bash
python train_enhanced.py --config configs/config_enhanced.yaml
```

Expected score: ~0.98-1.00

### 6. Train QLoRA (Large Models)

```bash
python train_qlora.py --config configs/config_qlora.yaml
```

Expected score: ~0.95-0.98

### 7. Generate Ensemble Submission

```bash
python ensemble_inference.py \
    --deberta-dirs models \
    --test data/test.csv \
    --output submission_ensemble.csv \
    --optimize-weights
```

## Winning Techniques Implemented

### 1. Layer-wise Learning Rate Decay (LLRD)
Earlier transformer layers get lower learning rates, later layers get higher.
```yaml
optimizer:
  use_llrd: true
  llrd_factor: 0.9  # 10% decay per layer
```

### 2. Label Smoothing
Prevents overconfidence and improves generalization.
```yaml
loss:
  type: "label_smoothing"
  smoothing: 0.05
```

### 3. Focal Loss
Handles class imbalance by focusing on hard examples.
```yaml
loss:
  type: "focal"
  gamma: 2.0
  alpha: [1.0, 1.0, 1.5]  # Upweight ties
```

### 4. Exponential Moving Average (EMA)
Maintains moving average of weights for better generalization.
```yaml
ema:
  enabled: true
  decay: 0.999
```

### 5. FGM Adversarial Training
Adds perturbations to embeddings for robustness.
```yaml
fgm:
  enabled: true
  epsilon: 0.5
```

### 6. AWP Adversarial Training
Adversarial weight perturbation for extra robustness.
```yaml
awp:
  enabled: true
  adv_lr: 1.0e-4
  adv_eps: 1.0e-2
```

### 7. R-Drop Regularization
Forces consistent outputs across dropout masks.
```yaml
rdrop:
  enabled: true
  alpha: 0.3
```

### 8. Multi-Seed Ensemble
Train with multiple seeds and average predictions.
```yaml
ensemble:
  seeds: [42, 1337, 2024]
  averaging: "geometric"
```

### 9. Test-Time Augmentation (TTA)
Swap responses A/B and average predictions.
```yaml
inference:
  tta: true
```

### 10. Multiple Input Formats
Experiment with different prompt formats.
- `default`: [PROMPT] ... [RESPONSE A] ... [RESPONSE B] ...
- `markdown`: ### Prompt\n...\n### Response A\n...
- `comparison`: Compare these responses...
- `simple`: prompt [SEP] response_a [SEP] response_b

## Performance Roadmap

| Phase | Method | Expected LB |
|-------|--------|-------------|
| 1 | TF-IDF + LogReg | ~1.10 |
| 2 | DeBERTa-v3-large | ~1.03 |
| 3 | + LLRD + Label Smoothing | ~1.01 |
| 4 | + EMA + FGM | ~0.99 |
| 5 | + Multi-seed ensemble | ~0.97 |
| 6 | + QLoRA (Gemma-7B) | ~0.95 |
| 7 | + DeBERTa + QLoRA ensemble | <0.93 |

## Supported Models

### Encoder Models (DeBERTa)
- `microsoft/deberta-v3-large` (recommended)
- `microsoft/deberta-v3-base`
- `microsoft/deberta-v2-xlarge`

### Decoder Models (QLoRA)
- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Meta-Llama-3-8B`
- `mistralai/Mistral-7B-v0.1`
- `google/gemma-2b` / `google/gemma-7b`
- `Qwen/Qwen1.5-1.8B` / `Qwen/Qwen1.5-7B`
- `microsoft/phi-2`

## Hardware Requirements

| Model | VRAM | Training Time |
|-------|------|---------------|
| TF-IDF Baseline | CPU only | ~5 min |
| DeBERTa-v3-large | 16GB+ | ~2-4 hours |
| Gemma-2B (QLoRA) | 8GB+ | ~3-5 hours |
| Gemma-7B (QLoRA) | 16GB+ | ~6-10 hours |
| Llama-2-7B (QLoRA) | 16GB+ | ~6-10 hours |

## Key Insights

1. **Input Format Matters**: Experiment with different formats
2. **Max Length**: 1536 tokens captures most examples
3. **Class Weights**: Ties are underrepresented; upweight them
4. **Ensembling**: All top solutions use model ensembles
5. **Diversity**: Mix encoder (DeBERTa) and decoder (Llama) models

## Files Overview

| File | Description |
|------|-------------|
| `data_loader.py` | Dataset class with multiple input formats |
| `deberta_model.py` | DeBERTa with mean/attention pooling, multi-sample dropout |
| `training_utils.py` | LLRD, EMA, FGM, label smoothing, focal loss, R-Drop |
| `train_enhanced.py` | Training with all winning techniques |
| `qlora_model.py` | QLoRA wrapper for decoder models |
| `train_qlora.py` | QLoRA training with 4-bit quantization |
| `ensemble_inference.py` | Ensemble predictions with weight optimization |

## License

MIT
