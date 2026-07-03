# Swift-SRGAN (fork)

Fork of [**Swift-SRGAN**](https://github.com/Koushik0901/Swift-SRGAN) — *Rethinking Super-Resolution for real-time inference* ([arXiv:2111.14320](https://arxiv.org/abs/2111.14320)).

This fork extends the original training code with **multi-dataset automation**, **resume support**, **2x / 4x / 8x weight naming**, and **disk cleanup** after training.

## What changed in this fork

| Area | Original | This fork |
|------|----------|-----------|
| Upscale factors | 2x, 4x (8x via manual weights) | First-class **2x, 4x, 8x** training |
| Datasets | Single local `dataset/` folder | **general** (DF2K), **personalized**, **compound** |
| Weight files | `original_{scale}x.pth` (upstream) | `{dataset}_{scale}x.pth` (trained) |
| Training entrypoint | `train.py` only | `run_all_training.py` orchestrates all jobs |
| Resume | Not supported | `--resume` with corrupt-checkpoint recovery |
| Artifacts | Kept manually | Auto-cleanup of checkpoints/images after each run |

## Repository layout

```text
Swift-SRGAN/
├── swift-srgan/          # Python package and scripts
│   ├── train.py          # Core SRGAN training loop
│   ├── run_all_training.py
│   ├── prepare_datasets.py
│   ├── download_df2k.py
│   ├── models.py
│   ├── data.py
│   ├── loss.py
│   └── metric.py
├── datasets/
│   ├── general/          # DF2K → train/ valid/
│   ├── personalized/
│   │   └── source/       # place your HR images here
│   └── compound/         # general + personalized merge
├── weights/              # final generator weights for inference
├── runs/                 # per-run checkpoints, logs, validation images
├── requirements.txt
└── LICENSE
```

## Setup

```bash
cd swift-srgan
pip install -r ../requirements.txt
```

Requires **CUDA** for practical training times (`train.py` falls back to CPU if unavailable).

## Quick start: train all models

```bash
cd swift-srgan

# 1. Place personalized images (2K recommended) in:
#    ../datasets/personalized/source/

# 2. Prepare datasets (downloads DF2K on first run)
python prepare_datasets.py

# 3. Train 9 models: 3 datasets × 3 scales
python run_all_training.py
```

### Output weights

Final generators are written to `weights/`:

```text
general_2x.pth       general_4x.pth       general_8x.pth
personalized_2x.pth  personalized_4x.pth  personalized_8x.pth
compound_2x.pth      compound_4x.pth      compound_8x.pth
```

Legacy upstream weights: `original_2x.pth`, `original_4x.pth`. Trained weights: `general_2x.pth`, `personalized_4x.pth`, etc.

## Scripts

### `run_all_training.py` — main orchestrator

Runs dataset preparation (optional) and training for selected datasets and scales.

```bash
python run_all_training.py --datasets general personalized --scales 2 4
python run_all_training.py --skip-prepare          # datasets already built
python run_all_training.py --no-resume             # ignore checkpoints, start fresh
python run_all_training.py --force-retrain         # retrain even if weights exist
python run_all_training.py --keep-run-artifacts    # keep checkpoints/images after training
python run_all_training.py --cleanup-completed    # remove leftovers for finished runs
```

| Flag | Description |
|------|-------------|
| `--datasets` | `general`, `personalized`, `compound` (default: all) |
| `--scales` | `2`, `4`, `8` (default: all) |
| `--num-epochs` | Epochs per run (default: 100) |
| `--batch-size` | Training batch size (default: 16) |
| `--crop-size` | HR crop size (default: 96) |
| `--skip-prepare` | Skip `prepare_datasets.py` |
| `--force-prepare` | Rebuild train/valid splits |
| `--no-resume` | Do not load checkpoints |
| `--force-retrain` | Retrain even when `weights/*.pth` exists |
| `--keep-run-artifacts` | Disable post-training cleanup |
| `--cleanup-completed` | Delete leftover `runs/` artifacts for completed jobs |

**Resume** is enabled by default. Checkpoints live under `runs/{dataset}/{scale}x/checkpoints/`. Only the latest valid `netG` + `netD` pair is used; corrupt files are skipped automatically.

After a successful run, checkpoints and validation images are removed by default (logs and final weights are kept).

---

### `prepare_datasets.py` — build train/valid splits

```bash
python prepare_datasets.py                    # all three datasets
python prepare_datasets.py --only general       # DF2K only
python prepare_datasets.py --only personalized  # split source/ → train/ valid/
python prepare_datasets.py --only compound      # merge general + personalized
python prepare_datasets.py --force              # rebuild even if splits exist
```

Skips work that is already done (existing `train/` and `valid/` folders with images).

---

### `download_df2k.py` — DF2K downloader

Used internally by `prepare_datasets.py`. Can also be run standalone:

```bash
python download_df2k.py --output-dir ../datasets/general
```

Downloads DIV2K + Flickr2K (train) and DIV2K valid. Skips re-download if `train/` and `valid/` already contain images.

---

### `train.py` — single training job

Lower-level entry point; called by `run_all_training.py`.

```bash
python train.py \
  --upscale_factor 4 \
  --train-dir ../datasets/general/train \
  --valid-dir ../datasets/general/valid \
  --weights-out ../weights/general_4x.pth \
  --checkpoint-dir ../runs/general/4x/checkpoints \
  --results-dir ../runs/general/4x/results \
  --logs-dir ../runs/general/4x/logs \
  --run-name general_4x \
  --resume
```

| Flag | Description |
|------|-------------|
| `--upscale_factor` | `2`, `4`, or `8` |
| `--num_epochs` | Default 100 |
| `--resume` | Continue from latest valid checkpoint |
| `--keep-run-artifacts` | Do not delete checkpoints/images when done |

---

### Core modules (not run directly)

| File | Role |
|------|------|
| `models.py` | Generator and Discriminator |
| `data.py` | `TrainDataset`, `ValDataset`, augmentations |
| `loss.py` | Generator GAN + perceptual + content loss |
| `metric.py` | SSIM for validation |

---

## Pre-trained models

The [original repository releases](https://github.com/Koushik0901/Swift-SRGAN/releases) provide 2x and 4x generators. For 8x, train locally or place a weight file in `weights/`.

## License

See [LICENSE](LICENSE) (CC0 1.0 Universal, from upstream).
