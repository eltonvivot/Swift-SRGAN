#!/usr/bin/env python3
"""Train Swift-SRGAN weights for general, personalized, and compound datasets."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from download_df2k import IMAGE_SUFFIXES

SWIFT_SRGAN_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SWIFT_SRGAN_ROOT / "datasets"
WEIGHTS_DIR = SWIFT_SRGAN_ROOT / "weights"
RUNS_DIR = SWIFT_SRGAN_ROOT / "runs"

DATASET_LABELS = ("general", "personalized", "compound")
UPSCALE_FACTORS = (2, 4, 8)

# Edit these defaults before launching long training jobs.
NUM_EPOCHS = 100
BATCH_SIZE = 16
CROP_SIZE = 96
PREPARE_DATASETS = True
KEEP_DF2K_ARCHIVES = False

PREPARE_ORDER = ("general", "personalized", "compound")


def _has_images(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in directory.iterdir()
    )


def _dataset_is_ready(dataset_label: str) -> bool:
    root = DATASET_ROOT / dataset_label
    return _has_images(root / "train") and _has_images(root / "valid")


def _prepare_targets(requested_datasets: list[str]) -> list[str]:
    targets = set(requested_datasets)
    if "compound" in targets:
        targets.add("general")
        targets.add("personalized")
    return [label for label in PREPARE_ORDER if label in targets]


def _datasets_needing_prepare(requested_datasets: list[str], *, force: bool) -> list[str]:
    targets = _prepare_targets(requested_datasets)
    if force:
        return targets
    return [label for label in targets if not _dataset_is_ready(label)]


def _run_prepare(
    dataset_labels: list[str],
    *,
    keep_archives: bool,
    force: bool,
) -> None:
    for label in dataset_labels:
        prepare_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "prepare_datasets.py"),
            "--dataset-root",
            str(DATASET_ROOT),
            "--only",
            label,
        ]
        if keep_archives:
            prepare_cmd.append("--keep-archives")
        if force:
            prepare_cmd.append("--force")
        print(f"Preparing dataset: {label}")
        subprocess.run(prepare_cmd, cwd=SCRIPT_DIR, check=True)


def _weights_name(dataset_label: str, upscale_factor: int) -> str:
    return f"{dataset_label}_{upscale_factor}x.pth"


def _run_dir(dataset_label: str, upscale_factor: int) -> Path:
    return RUNS_DIR / dataset_label / f"{upscale_factor}x"


def _training_complete(weights_out: Path) -> bool:
    return weights_out.is_file()


def _cleanup_completed_run(run_dir: Path, weights_out: Path) -> None:
    if not weights_out.is_file():
        return

    removed: list[str] = []
    for subdir in ("checkpoints", "results"):
        path = run_dir / subdir
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path))

    legacy_images = sorted(run_dir.glob("resultsepoch_*.png"))
    if legacy_images:
        for path in legacy_images:
            path.unlink()
        removed.append(f"{len(legacy_images)} legacy validation images in {run_dir}")

    if removed:
        print("[cleanup] removed leftover artifacts for completed run:")
        for path in removed:
            print(f"  - {path}")


def train_one(
    dataset_label: str,
    upscale_factor: int,
    *,
    num_epochs: int,
    batch_size: int,
    crop_size: int,
    resume: bool,
    force_retrain: bool,
    keep_run_artifacts: bool,
    cleanup_completed: bool,
) -> None:
    train_dir = DATASET_ROOT / dataset_label / "train"
    valid_dir = DATASET_ROOT / dataset_label / "valid"
    if not train_dir.is_dir() or not valid_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset not prepared for '{dataset_label}'. "
            f"Expected {train_dir} and {valid_dir}."
        )

    run_dir = _run_dir(dataset_label, upscale_factor)
    checkpoint_dir = run_dir / "checkpoints"
    results_dir = run_dir / "results"
    logs_dir = run_dir / "logs"
    weights_out = WEIGHTS_DIR / _weights_name(dataset_label, upscale_factor)
    run_name = f"{dataset_label}_{upscale_factor}x"

    if _training_complete(weights_out) and not force_retrain:
        print(f"\n=== Skipping {run_name}: final weights already exist at {weights_out} ===")
        if cleanup_completed:
            _cleanup_completed_run(run_dir, weights_out)
        return

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train.py"),
        "--upscale_factor",
        str(upscale_factor),
        "--crop_size",
        str(crop_size),
        "--batch_size",
        str(batch_size),
        "--num_epochs",
        str(num_epochs),
        "--train-dir",
        str(train_dir),
        "--valid-dir",
        str(valid_dir),
        "--weights-out",
        str(weights_out),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--results-dir",
        str(results_dir),
        "--logs-dir",
        str(logs_dir),
        "--run-name",
        run_name,
    ]
    if resume:
        cmd.append("--resume")
    if keep_run_artifacts:
        cmd.append("--keep-run-artifacts")

    print(f"\n=== Training {run_name} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all Swift-SRGAN training jobs.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_LABELS,
        default=list(DATASET_LABELS),
        help="Dataset labels to train (default: all)",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        choices=UPSCALE_FACTORS,
        default=list(UPSCALE_FACTORS),
        help="Upscale factors to train (default: 2 4 8)",
    )
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip dataset preparation step",
    )
    parser.add_argument(
        "--force-prepare",
        action="store_true",
        help="Rebuild train/valid splits even when they already exist",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        default=KEEP_DF2K_ARCHIVES,
        help="Keep DF2K download archives during preparation",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start training from scratch even if checkpoints exist",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain runs even when final weights already exist",
    )
    parser.add_argument(
        "--keep-run-artifacts",
        action="store_true",
        help="Keep checkpoints and validation images after each run completes",
    )
    parser.add_argument(
        "--cleanup-completed",
        action="store_true",
        help="Remove leftover checkpoints/results for runs already finished (weights exist)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_prepare:
        needed = _datasets_needing_prepare(args.datasets, force=args.force_prepare)
        if not needed:
            print("Datasets already prepared (train/valid found), skipping preparation.")
        else:
            _run_prepare(needed, keep_archives=args.keep_archives, force=args.force_prepare)

    failed: list[str] = []
    for dataset_label in args.datasets:
        for upscale_factor in args.scales:
            run_name = f"{dataset_label}_{upscale_factor}x"
            try:
                train_one(
                    dataset_label,
                    upscale_factor,
                    num_epochs=args.num_epochs,
                    batch_size=args.batch_size,
                    crop_size=args.crop_size,
                    resume=not args.no_resume,
                    force_retrain=args.force_retrain,
                    keep_run_artifacts=args.keep_run_artifacts,
                    cleanup_completed=args.cleanup_completed,
                )
            except subprocess.CalledProcessError as exc:
                print(f"Training failed for {run_name}: exit code {exc.returncode}", file=sys.stderr)
                failed.append(run_name)

    if failed:
        print(f"Failed runs: {', '.join(failed)}", file=sys.stderr)
        return 1

    print("\nAll requested trainings finished.")
    print(f"Weights directory: {WEIGHTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
