# Contributing

Thanks for your interest in improving this project! This repository holds an
experimentation pipeline for the **LLM Classification Finetuning** Kaggle
competition. Contributions that improve model performance, reproducibility, or
code clarity are welcome.

## Getting Started

1. Fork and clone the repository.
2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Create a feature branch:

   ```bash
   git checkout -b feature/my-improvement
   ```

## Development Guidelines

- **Style**: Follow [PEP 8](https://peps.python.org/pep-0008/). Keep functions
  focused and add type hints where practical — the existing modules use them.
- **Docstrings**: Every public function and class should carry a short docstring
  describing its purpose, matching the style already in the codebase.
- **Config over constants**: New hyperparameters belong in `config.yaml`, not
  hard-coded in the training scripts.
- **Reproducibility**: Respect the `seed` setting and avoid introducing
  nondeterminism into the training/inference paths.

## Making Changes

- Keep pull requests focused on a single concern.
- Update the [README](README.md) and [CHANGELOG](CHANGELOG.md) when behavior or
  the public API changes.
- Verify the baseline still runs end-to-end before submitting:

  ```bash
  python run_baseline.py --train train_sample.csv --test test.csv
  ```

## Reporting Issues

When opening an issue, please include:

- What you expected to happen versus what actually happened.
- Steps to reproduce, including the command and any config overrides.
- Environment details (OS, Python version, GPU, key package versions).

## Data Notice

Do **not** commit the full competition dataset or model checkpoints. Only the
small `train_sample.csv` / `test.csv` fixtures belong in version control; keep
large downloads under an ignored `data/` directory.
