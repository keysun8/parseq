# PARSeq_Indic_HTR

Experimenting with **PARSeq** (Scene Text Recognition with Permuted Autoregressive Sequence Models, ECCV 2022) for **Indic Handwritten/Printed Text Recognition (HTR)**.

This repo is a fork of [baudm/parseq](https://github.com/baudm/parseq), adapted for training and evaluating PARSeq on Indic-language datasets (e.g. Hindi), using [Hydra](https://hydra.cc) for configuration and [PyTorch Lightning](https://www.pytorchlightning.ai/) for training.

## Table of Contents

- [Setup](#setup)
- [Preparing Data and Charsets](#preparing-data-and-charsets)
- [Training (`train.py`)](#training-trainpy)
- [Evaluation (`test.py`)](#evaluation-testpy)
- [Predicting on Custom Images (`test_word_image.py`)](#predicting-on-custom-images-test_word_imagepy)
- [Reading Single Images (`read.py`)](#reading-single-images-readpy)
- [Slurm Cluster Scripts](#slurm-cluster-scripts)
- [Repo Structure](#repo-structure)

## Setup

**Requirements:** Python 3.9+, a CUDA-capable GPU is strongly recommended for training.

```bash
git clone https://github.com/keysun8/parseq.git
cd parseq

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies (pins torch==1.13.1+cu117, pytorch-lightning==1.9.5, hydra-core==1.3.2, etc.)
pip install -r requirements.txt

# Install the project itself (needed so the `strhub` package is importable)
pip install -e .
```

> If you need a different CUDA/CPU build of PyTorch, edit the `--extra-index-url` / torch version pins at the top of `requirements.txt` before installing.

## Preparing Data and Charsets

1. Prepare your dataset in the LMDB format expected by PARSeq. Follow the original project's data guide: [baudm/parseq/Datasets.md](https://github.com/baudm/parseq/blob/main/Datasets.md).
2. Place the dataset(s) in the paths referenced by your data config (see `configs/` / `config/`), and update `data.root_dir` (or the relevant config) to match where you placed them.
3. Add your character set (charset) file to the `configs/` folder. A few charsets are already included — add a new one for your language if needed.
4. Set which charset to use, and any other hyperparameters, in `configs/main.yaml` (or via command-line overrides — see below).
5. Double-check the paths where training outputs/checkpoints will be saved and adjust them if necessary.

## Training (`train.py`)

Training is driven by Hydra, so you can override any config value from the command line. See the default configuration with:

```bash
./train.py --help
```

### Basic training run

```bash
python3 train.py +experiment=parseq
```

### Common overrides

```bash
# Train for a fixed number of epochs, on GPU, using a given batch size
python3 train.py +experiment=parseq trainer.max_epochs=10 model.batch_size=128 trainer.accelerator=gpu trainer.devices=4

# Pick the character set (see configs/charset/)
python3 train.py charset=94_full

# Point to your dataset root and data loading options
python3 train.py data.root_dir=data data.num_workers=2 data.augment=true

# Finetune from pretrained weights (not all model variants have pretrained weights)
python3 train.py +experiment=parseq-tiny pretrained=parseq-tiny

# Resume from a checkpoint
python3 train.py +experiment=parseq ckpt_path=outputs/parseq/<timestamp>/checkpoints/last.ckpt
```

Any [PyTorch Lightning `Trainer` parameter](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html) can be passed the same way; prefix it with `+` if it isn't already defined in `configs/main.yaml`.

Checkpoints and logs are written under `outputs/<model>/<timestamp>/checkpoints/`.

## Evaluation (`test.py`)

`test.py` evaluates any checkpoint produced by `train.py` (or a released pretrained model) against your test/benchmark datasets.

```bash
./test.py --help

# Evaluate a trained checkpoint (default: lowercase alphanumeric charset)
./test.py outputs/parseq/<timestamp>/checkpoints/last.ckpt

# Or evaluate a released pretrained model
./test.py pretrained=parseq
```

Evaluate with different character sets:

```bash
./test.py outputs/parseq/<timestamp>/checkpoints/last.ckpt                          # lowercase alphanumeric
./test.py outputs/parseq/<timestamp>/checkpoints/last.ckpt --cased                   # mixed-case alphanumeric
./test.py outputs/parseq/<timestamp>/checkpoints/last.ckpt --cased --punctuation     # mixed-case + punctuation
```

Model-specific runtime parameters can be passed as `param:type=value`, e.g. to use PARSeq's non-autoregressive decoding with refinement:

```bash
./test.py outputs/parseq/<timestamp>/checkpoints/last.ckpt refine_iters:int=2 decode_ar:bool=false
```

## Predicting on Custom Images (`test_word_image.py`)

This script (added in this fork) runs a trained checkpoint over a **folder of word images** and writes predictions with confidence scores to a text file — useful for quick qualitative checks without needing LMDB test sets.

```bash
python 0_code/test_word_image.py <path/to/checkpoint.ckpt> \
    --image_dir <path/to/image/folder> \
    --output_file predictions.txt \
    --device cuda
```

Arguments:
- `checkpoint` (positional) — path to the trained `.ckpt` file.
- `--image_dir` (required) — folder containing `.jpg` / `.jpeg` / `.png` / `.bmp` images.
- `--output_file` (optional, default `predictions.txt`) — tab-separated output file with columns `image_name`, `prediction`, `confidence`.
- `--device` (optional, default `cuda`) — set to `cpu` if no GPU is available.

## Reading Single Images (`read.py`)

For quickly reading text from a handful of images using the original PARSeq entry point:

```bash
./read.py outputs/parseq/<timestamp>/checkpoints/last.ckpt --images demo_images/*
# or using a released pretrained weight
./read.py pretrained=parseq --images demo_images/*
```

## Slurm Cluster Scripts

Two helper shell scripts are included for running on a Slurm-managed GPU cluster — edit the `#SBATCH` directives (account, node, GPUs, time) and paths to match your environment before use:

- **`hindi_train.sh`** — submits a training job (`train.py`) with GPU/epoch settings.
- **`hindi_test_image.sh`** — runs `test_word_image.py` against a checkpoint to generate predictions for a folder of images.

Submit a job with:

```bash
sbatch hindi_train.sh
sbatch hindi_test_image.sh
```

## Repo Structure

```
config/, configs/     Hydra configuration files (model, dataset, charset, experiment presets)
strhub/               Core library (models, data modules, tokenizers, etc.)
tools/                Utility scripts
train.py              Training entry point
test.py               Evaluation entry point (LMDB benchmark datasets)
test_word_image.py    Run inference on a folder of images, output predictions + confidence
read.py               Read text from individual images using a trained/pretrained model
hubconf.py            Torch Hub integration
hindi_train.sh         Slurm script for training
hindi_test_image.sh    Slurm script for image-folder inference
requirements.txt       Pinned Python dependencies
predictions.txt        Example/sample output from test_word_image.py
```

## License

Released under the Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for details (portions derived from ABINet/CRNN carry their own BSD/MIT licenses per the original PARSeq project).

## Acknowledgements

Built on top of [baudm/parseq](https://github.com/baudm/parseq) by Darwin Bautista and Rowel Atienza (ECCV 2022). See the original paper: *Scene Text Recognition with Permuted Autoregressive Sequence Models* ([arXiv:2207.06966](https://arxiv.org/abs/2207.06966)).
