# TransFuzz

## Overview

## Environment Setup

```bash
python -m venv transfuzz-env
source transfuzz-env/bin/activate
pip install -r requirements.txt
```

## TransFuzz Interface
### Arguments

`transfuzz.py` exposes the following command-line arguments to control the fuzzing process:

| Argument | Description |
|--------|-------------|
| `--model` | Target model to fuzz (required) |
| `--seed-dataset` | Dataset to use for seeds |
| `--seed-count`   | Number of seed inputs from the dataset to use (-1 for all) |
| `--split`  | Dataset split to use for seed inputs |
| `--coverage-metric` | Coverage signal used for feedback (default: NLC) |
| `--target-label` | Enables targeted adversarial testing |
| `--random-mutation` | Disables gradient-guided mutation |
| `--time-budget` | Maximum fuzzing time in seconds |
| `--N` | Number of seed inputs per perturbation |
| `--seed` | Random seed for reproducibility |


## Quick Start
### 


## Reproducing Key Results



README.md
├── Overview
├── Environment Setup
├── Quick Start (5-minute run)
├── Reproducing Key Results
│   ├── Coverage-guided fuzzing (RQ1)
│   ├── Fault quality evaluation (RQ2)
│   └── Targeted fuzzing (RQ3)
├── Configuration Files
├── Output and Metrics
├── Notes on Runtime and Resources
├── Adapted Components and Attribution
