# TransFuzz

## Installation

```bash
python -m venv transfuzz-env
source transfuzz-env/bin/activate
pip install -r requirements.txt
```

## TransFuzz Interface

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
| `--N` | Number of seed inputs associated with a single perturbation |
| `--seed` | Random seed for reproducibility |


## Getting Started

### Fuzz pretrained ResNet-50 image classification model

```bash
#  Download ImageNet seed inputs. This will store the seed inputs in `seeds` directory
python download_imagenet.py

# Run `TransFuzz`
python transfuzz.py --model resnet50 --seed-dataset ImageNet  --split val --time-budget 300 --N 24
```

### Fuzz pretrained AST keyword spotting model with (target label = 24, "off")

```bash
python transfuzz.py --model mitast --seed-dataset speech_commands  --split test --time-budget 300 --N 24 --target-label 24
```

## Experiments

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
