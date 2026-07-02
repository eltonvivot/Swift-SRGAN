#!/usr/bin/env python3
"""Prepare general, personalized, and compound Swift-SRGAN datasets."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from download_df2k import IMAGE_SUFFIXES, prepare_df2k

DATASET_ROOT = Path(__file__).resolve().parent.parent / "datasets"
PERSONALIZED_SOURCE_DIRNAME = "source"
VALID_SPLIT_RATIO = 0.05
VALID_SPLIT_MIN = 10
VALID_SPLIT_MAX = 100


def _has_images(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    return any(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in directory.iterdir()
    )


def _iter_images(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def _collect_images(src_root: Path, dst_dir: Path, prefix: str) -> int:
    count = 0
    for image in _iter_images(src_root):
        out_path = dst_dir / f"{prefix}_{image.name}"
        if out_path.exists():
            stem, suffix = out_path.stem, out_path.suffix
            index = 1
            while (dst_dir / f"{stem}_{index}{suffix}").exists():
                index += 1
            out_path = dst_dir / f"{stem}_{index}{suffix}"
        _link_or_copy(image, out_path)
        count += 1
    return count


def prepare_general(
    dataset_root: Path = DATASET_ROOT / "general",
    keep_archives: bool = False,
    *,
    force: bool = False,
) -> None:
    """Download DF2K into ``datasets/general/{train,valid}``."""
    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"
    if not force and _has_images(train_dir) and _has_images(valid_dir):
        print(f"[general] train/valid already prepared at {dataset_root}, skipping download")
        return
    prepare_df2k(output_dir=dataset_root, keep_archives=keep_archives)


def prepare_personalized(
    dataset_root: Path = DATASET_ROOT / "personalized",
    source_dir: Path | None = None,
    seed: int = 42,
    *,
    force: bool = False,
) -> None:
    """Split personalized images from ``datasets/personalized/source`` into train/valid."""
    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"
    if not force and _has_images(train_dir) and _has_images(valid_dir):
        print(f"[personalized] train/valid already prepared at {dataset_root}, skipping split")
        return

    source = source_dir or (dataset_root / PERSONALIZED_SOURCE_DIRNAME)
    if not source.is_dir():
        raise FileNotFoundError(
            f"Personalized source folder not found: {source}\n"
            "Place personalized HR images in that directory before running preparation."
        )

    images = sorted(_iter_images(source))
    if not images:
        raise ValueError(f"No images found in {source}")

    _clear_dir(train_dir)
    _clear_dir(valid_dir)

    rng = random.Random(seed)
    shuffled = images[:]
    rng.shuffle(shuffled)

    valid_count = max(
        VALID_SPLIT_MIN,
        min(VALID_SPLIT_MAX, int(round(len(shuffled) * VALID_SPLIT_RATIO))),
    )
    valid_count = min(valid_count, max(1, len(shuffled) // 5))
    valid_images = shuffled[:valid_count]
    train_images = shuffled[valid_count:]

    for index, image in enumerate(train_images):
        _link_or_copy(image, train_dir / f"personalized_{index:05d}{image.suffix.lower()}")
    for index, image in enumerate(valid_images):
        _link_or_copy(image, valid_dir / f"personalized_{index:05d}{image.suffix.lower()}")

    print(f"[personalized] train images: {len(train_images)}")
    print(f"[personalized] valid images: {len(valid_images)}")
    print(f"[personalized] output: {dataset_root}")


def prepare_compound(
    dataset_root: Path = DATASET_ROOT / "compound",
    general_root: Path = DATASET_ROOT / "general",
    personalized_root: Path = DATASET_ROOT / "personalized",
    *,
    force: bool = False,
) -> None:
    """Merge general and personalized splits into ``datasets/compound``."""
    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"
    if not force and _has_images(train_dir) and _has_images(valid_dir):
        print(f"[compound] train/valid already prepared at {dataset_root}, skipping merge")
        return

    general_train = general_root / "train"
    general_valid = general_root / "valid"
    personalized_train = personalized_root / "train"
    personalized_valid = personalized_root / "valid"

    for required in (general_train, general_valid, personalized_train, personalized_valid):
        if not required.is_dir():
            raise FileNotFoundError(f"Missing dataset folder: {required}")

    _clear_dir(train_dir)
    _clear_dir(valid_dir)

    train_count = 0
    train_count += _collect_images(general_train, train_dir, "general")
    train_count += _collect_images(personalized_train, train_dir, "personalized")

    valid_count = 0
    valid_count += _collect_images(general_valid, valid_dir, "general")
    valid_count += _collect_images(personalized_valid, valid_dir, "personalized")

    print(f"[compound] train images: {train_count}")
    print(f"[compound] valid images: {valid_count}")
    print(f"[compound] output: {dataset_root}")


def prepare_all(
    dataset_root: Path = DATASET_ROOT,
    personalized_source: Path | None = None,
    keep_archives: bool = False,
) -> None:
    general_root = dataset_root / "general"
    personalized_root = dataset_root / "personalized"
    compound_root = dataset_root / "compound"

    prepare_general(general_root, keep_archives=keep_archives)
    prepare_personalized(personalized_root, source_dir=personalized_source)
    prepare_compound(compound_root, general_root, personalized_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Swift-SRGAN training datasets.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help=f"Root datasets directory (default: {DATASET_ROOT})",
    )
    parser.add_argument(
        "--personalized-source",
        type=Path,
        help="Folder with personalized images (default: datasets/personalized/source)",
    )
    parser.add_argument(
        "--only",
        choices=["general", "personalized", "compound", "all"],
        default="all",
        help="Prepare only one dataset split (default: all)",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep DF2K download archives",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild train/valid splits even when they already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root
    personalized_source = args.personalized_source

    try:
        if args.only in {"general", "all"}:
            prepare_general(
                dataset_root / "general",
                keep_archives=args.keep_archives,
                force=args.force,
            )
        if args.only in {"personalized", "all"}:
            prepare_personalized(
                dataset_root / "personalized",
                source_dir=personalized_source,
                force=args.force,
            )
        if args.only in {"compound", "all"}:
            prepare_compound(
                dataset_root / "compound",
                dataset_root / "general",
                dataset_root / "personalized",
                force=args.force,
            )
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
