#!/usr/bin/env python3
"""Download and prepare DF2K dataset for Swift-SRGAN training.

DF2K = DIV2K train + Flickr2K for training, and DIV2K valid for validation.

Default output layout (relative to current working directory):
    dataset/
      train/
      valid/

Usage:
    python download_df2k.py
    python download_df2k.py --output-dir ./dataset --keep-archives
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


DATASETS = {
    "div2k_train": {
        "url": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "archive": "DIV2K_train_HR.zip",
    },
    "div2k_valid": {
        "url": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "archive": "DIV2K_valid_HR.zip",
    },
    "flickr2k": {
        "url": "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar",
        "archive": "Flickr2K.tar",
    },
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"[skip] archive exists: {destination}")
        return

    print(f"[download] {url}")

    def _progress(blocks: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(blocks * block_size, total_size)
        pct = 100.0 * downloaded / total_size
        print(f"\r  -> {pct:5.1f}% ({downloaded // (1024 * 1024)} / {total_size // (1024 * 1024)} MiB)", end="")

    urllib.request.urlretrieve(url, destination, reporthook=_progress)
    print("\n[done] download complete")


def _extract(archive: Path, extract_root: Path) -> Path:
    target = extract_root / archive.stem

    if target.exists() and any(target.iterdir()):
        print(f"[skip] already extracted: {target}")
        return target

    target.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive.name} -> {target}")

    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target)
    elif archive.suffix.lower() == ".tar":
        with tarfile.open(archive, "r") as tf:
            tf.extractall(target)
    else:
        raise ValueError(f"Unsupported archive type: {archive}")

    return target


def _iter_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            yield p


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            return
        dst.hardlink_to(src)
    except Exception:
        if not dst.exists():
            shutil.copy2(src, dst)


def _collect_images(src_root: Path, dst_dir: Path, prefix: str) -> int:
    count = 0
    for image in _iter_images(src_root):
        out_name = f"{prefix}_{image.name}"
        out_path = dst_dir / out_name

        # Ensure unique filename in case of collisions.
        if out_path.exists():
            stem = out_path.stem
            suffix = out_path.suffix
            idx = 1
            while (dst_dir / f"{stem}_{idx}{suffix}").exists():
                idx += 1
            out_path = dst_dir / f"{stem}_{idx}{suffix}"

        _link_or_copy(image, out_path)
        count += 1
    return count


def prepare_df2k(output_dir: Path, keep_archives: bool) -> None:
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    train_count = sum(1 for _ in _iter_images(train_dir))
    valid_count = sum(1 for _ in _iter_images(valid_dir))
    if train_count > 0 and valid_count > 0:
        print(f"[skip] DF2K already prepared at {output_dir}")
        print(f"[skip] train images: {train_count}, valid images: {valid_count}")
        return

    downloads_dir = output_dir / "_downloads"
    extracts_dir = downloads_dir / "_extracts"

    extracted = {}
    for key, meta in DATASETS.items():
        archive_path = downloads_dir / meta["archive"]
        _download(meta["url"], archive_path)
        extracted[key] = _extract(archive_path, extracts_dir)

    print("[prepare] assembling dataset/train and dataset/valid")

    train_count = 0
    train_count += _collect_images(extracted["div2k_train"], train_dir, "div2k_train")
    train_count += _collect_images(extracted["flickr2k"], train_dir, "flickr2k")

    valid_count = 0
    valid_count += _collect_images(extracted["div2k_valid"], valid_dir, "div2k_valid")

    print(f"[done] train images: {train_count}")
    print(f"[done] valid images: {valid_count}")
    print(f"[done] output: {output_dir}")

    if not keep_archives:
        print(f"[cleanup] removing archives and extracts: {downloads_dir}")
        shutil.rmtree(downloads_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare DF2K dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset"),
        help="Output dataset directory (default: ./dataset)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded archives and extracted temp files",
    )

    args = parser.parse_args()
    prepare_df2k(output_dir=args.output_dir, keep_archives=args.keep_archives)


if __name__ == "__main__":
    main()
