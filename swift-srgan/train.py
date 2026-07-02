import os
import math
import re
import shutil
import pandas as pd
import torch
import torchvision
from pathlib import Path
from data import TrainDataset, ValDataset, display_transform
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from loss import GeneratorLoss
from metric import ssim
from tqdm import tqdm
import argparse


torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed_all(42)

CHECKPOINT_PATTERN = re.compile(r"^netG_(?P<scale>\d+)x_epoch(?P<epoch>\d+)\.pth\.tar$")


def _checkpoint_path(checkpoint_dir: Path, upscale_factor: int, epoch: int, network: str) -> Path:
    return checkpoint_dir / f"{network}_{upscale_factor}x_epoch{epoch}.pth.tar"


def _list_checkpoint_epochs(checkpoint_dir: Path, upscale_factor: int) -> list[int]:
    if not checkpoint_dir.is_dir():
        return []

    epochs: set[int] = set()
    for path in checkpoint_dir.iterdir():
        match = CHECKPOINT_PATTERN.match(path.name)
        if not match or int(match.group("scale")) != upscale_factor:
            continue
        epochs.add(int(match.group("epoch")))
    return sorted(epochs, reverse=True)


def _is_readable_checkpoint(path: Path, device: torch.device) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        checkpoint = torch.load(str(path), map_location=device)
    except Exception:
        return False
    return isinstance(checkpoint, dict) and "model" in checkpoint


def _remove_corrupt_checkpoint(path: Path, device: torch.device) -> None:
    if not path.is_file():
        return
    if _is_readable_checkpoint(path, device):
        return
    print(f"[resume] removing corrupt checkpoint: {path}")
    path.unlink(missing_ok=True)


def _cleanup_stale_temp_checkpoints(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.is_dir():
        return
    for path in checkpoint_dir.glob("*.tmp"):
        print(f"[resume] removing incomplete checkpoint write: {path}")
        path.unlink(missing_ok=True)


def _find_resumable_epoch(checkpoint_dir: Path, upscale_factor: int, device: torch.device) -> int | None:
    _cleanup_stale_temp_checkpoints(checkpoint_dir)
    for epoch in _list_checkpoint_epochs(checkpoint_dir, upscale_factor):
        netg_path = _checkpoint_path(checkpoint_dir, upscale_factor, epoch, "netG")
        netd_path = _checkpoint_path(checkpoint_dir, upscale_factor, epoch, "netD")

        netg_ok = _is_readable_checkpoint(netg_path, device)
        netd_ok = _is_readable_checkpoint(netd_path, device)
        if netg_ok and netd_ok:
            return epoch

        print(f"[resume] skipping unreadable checkpoint pair for epoch {epoch}")
        _remove_corrupt_checkpoint(netg_path, device)
        _remove_corrupt_checkpoint(netd_path, device)

    return None


def _save_checkpoint_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def _load_network_checkpoint(path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(str(path), map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return checkpoint


def _load_state_dict(checkpoint: dict) -> dict:
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def _empty_results() -> dict[str, list[float]]:
    return {
        "d_loss": [],
        "g_loss": [],
        "d_score": [],
        "g_score": [],
        "psnr": [],
        "ssim": [],
    }


def _load_results_history(logs_dir: Path, run_name: str) -> dict[str, list[float]]:
    csv_path = logs_dir / f"{run_name}_train_results.csv"
    if not csv_path.is_file():
        return _empty_results()

    data_frame = pd.read_csv(csv_path, index_col="Epoch")
    results = _empty_results()
    for column, key in (
        ("Loss_D", "d_loss"),
        ("Loss_G", "g_loss"),
        ("Score_D", "d_score"),
        ("Score_G", "g_score"),
        ("PSNR", "psnr"),
        ("SSIM", "ssim"),
    ):
        if column in data_frame.columns:
            results[key] = data_frame[column].tolist()
    return results


def _save_results_csv(logs_dir: Path, run_name: str, results: dict[str, list[float]]) -> None:
    if not results["d_loss"]:
        return

    data_frame = pd.DataFrame(
        data={
            "Loss_D": results["d_loss"],
            "Loss_G": results["g_loss"],
            "Score_D": results["d_score"],
            "Score_G": results["g_score"],
            "PSNR": results["psnr"],
            "SSIM": results["ssim"],
        },
        index=range(1, len(results["d_loss"]) + 1),
    )
    data_frame.to_csv(logs_dir / f"{run_name}_train_results.csv", index_label="Epoch")


def _remove_validation_images(run_dir: Path, results_dir: Path) -> list[str]:
    removed: list[str] = []

    if results_dir.is_dir():
        shutil.rmtree(results_dir)
        removed.append(f"validation images ({results_dir})")

    legacy_images = sorted(run_dir.glob("resultsepoch_*.png"))
    if legacy_images:
        for path in legacy_images:
            path.unlink()
        removed.append(f"legacy validation images ({len(legacy_images)} files in {run_dir})")

    return removed


def _cleanup_run_artifacts(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    results_dir: Path,
    weights_out: Path,
) -> None:
    """Remove checkpoints and validation images once final weights are saved."""
    if not weights_out.is_file():
        return

    removed: list[str] = []

    if checkpoint_dir.is_dir():
        shutil.rmtree(checkpoint_dir)
        removed.append(f"checkpoints ({checkpoint_dir})")

    removed.extend(_remove_validation_images(run_dir, results_dir))

    if removed:
        print("[cleanup] removed after successful training:")
        for item in removed:
            print(f"  - {item}")
        print(f"[cleanup] kept logs and final weights at {weights_out}")


def _maybe_resume(
    *,
    resume: bool,
    checkpoint_dir: Path,
    upscale_factor: int,
    num_epochs: int,
    netG: torch.nn.Module,
    netD: torch.nn.Module,
    optimizerG: torch.optim.Optimizer,
    optimizerD: torch.optim.Optimizer,
    device: torch.device,
    logs_dir: Path,
    run_name: str,
) -> tuple[int, dict[str, list[float]]]:
    if not resume:
        return 1, _empty_results()

    latest_epoch = _find_resumable_epoch(checkpoint_dir, upscale_factor, device)
    if latest_epoch is None:
        print("[resume] no valid checkpoints found, starting from epoch 1")
        return 1, _empty_results()

    netg_path = _checkpoint_path(checkpoint_dir, upscale_factor, latest_epoch, "netG")
    netd_path = _checkpoint_path(checkpoint_dir, upscale_factor, latest_epoch, "netD")
    netg_checkpoint = _load_network_checkpoint(netg_path, device)
    netd_checkpoint = _load_network_checkpoint(netd_path, device)

    netG.load_state_dict(_load_state_dict(netg_checkpoint))
    netD.load_state_dict(_load_state_dict(netd_checkpoint))

    if "optimizer" in netg_checkpoint:
        optimizerG.load_state_dict(netg_checkpoint["optimizer"])
    if "optimizer" in netd_checkpoint:
        optimizerD.load_state_dict(netd_checkpoint["optimizer"])

    results = netg_checkpoint.get("results")
    if not isinstance(results, dict):
        results = _load_results_history(logs_dir, run_name)

    for key in ("d_loss", "g_loss", "d_score", "g_score", "psnr", "ssim"):
        results.setdefault(key, [])
        while len(results[key]) < latest_epoch:
            results[key].append(float("nan"))

    if latest_epoch >= num_epochs:
        print(f"[resume] training already complete at epoch {latest_epoch}/{num_epochs}")
        return num_epochs + 1, results

    print(f"[resume] loaded checkpoints from epoch {latest_epoch}, continuing at epoch {latest_epoch + 1}")
    return latest_epoch + 1, results


def main(opt):

    train_dir = Path(opt.train_dir)
    valid_dir = Path(opt.valid_dir)
    results_dir = Path(opt.results_dir)
    checkpoint_dir = Path(opt.checkpoint_dir)
    logs_dir = Path(opt.logs_dir)
    weights_out = Path(opt.weights_out)

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    run_dir = results_dir.parent

    CROP_SIZE = opt.crop_size
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    train_set = TrainDataset(
        str(train_dir), crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR
    )
    val_set = ValDataset(
        str(valid_dir), crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR
    )

    train_loader = DataLoader(
        dataset=train_set,
        num_workers=os.cpu_count(),
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(dataset=val_set, num_workers=1, batch_size=1, shuffle=False)

    netG = Generator(upscale_factor=UPSCALE_FACTOR).to(DEVICE)
    print("# generator parameters:", sum(param.numel() for param in netG.parameters()))

    netD = Discriminator().to(DEVICE)
    print(
        "# discriminator parameters:", sum(param.numel() for param in netD.parameters())
    )

    generator_criterion = GeneratorLoss().to(DEVICE)

    optimizerG = torch.optim.AdamW(netG.parameters(), lr=1e-3)
    optimizerD = torch.optim.AdamW(netD.parameters(), lr=1e-3)

    start_epoch, results = _maybe_resume(
        resume=opt.resume,
        checkpoint_dir=checkpoint_dir,
        upscale_factor=UPSCALE_FACTOR,
        num_epochs=NUM_EPOCHS,
        netG=netG,
        netD=netD,
        optimizerG=optimizerG,
        optimizerD=optimizerD,
        device=DEVICE,
        logs_dir=logs_dir,
        run_name=opt.run_name,
    )

    if start_epoch > NUM_EPOCHS:
        _save_checkpoint_atomic(weights_out, {"model": netG.state_dict()})
        print(f"Saved generator weights to {weights_out}")
        if not opt.keep_run_artifacts:
            _cleanup_run_artifacts(
                run_dir=run_dir,
                checkpoint_dir=checkpoint_dir,
                results_dir=results_dir,
                weights_out=weights_out,
            )
        return

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader, total=len(train_loader))
        running_results = {
            "batch_sizes": 0,
            "d_loss": 0,
            "g_loss": 0,
            "d_score": 0,
            "g_score": 0,
        }

        netG.train()
        netD.train()
        for lr_img, hr_img in train_bar:
            batch_size = lr_img.size(0)
            running_results["batch_sizes"] += batch_size

            ############################
            # (1) Update D network: maximize D(x)-1-D(G(z))
            ###########################
            hr_img = hr_img.to(DEVICE)
            lr_img = lr_img.to(DEVICE)

            sr_img = netG(lr_img)

            if sr_img.shape != hr_img.shape:
                raise RuntimeError(
                    f"Generator output shape {tuple(sr_img.shape)} does not match "
                    f"HR target shape {tuple(hr_img.shape)}. "
                    "Check --upscale_factor and generator upsample configuration."
                )

            netD.zero_grad()
            real_out = netD(hr_img).mean()
            fake_out = netD(sr_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            ############################
            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            ###########################
            netG.zero_grad()

            sr_img = netG(lr_img)
            fake_out = netD(sr_img).mean()

            g_loss = generator_criterion(fake_out, sr_img, hr_img)
            g_loss.backward()

            optimizerG.step()

            # loss for current after before optimization
            running_results["g_loss"] += g_loss.item() * batch_size
            running_results["d_loss"] += d_loss.item() * batch_size
            running_results["d_score"] += real_out.item() * batch_size
            running_results["g_score"] += fake_out.item() * batch_size

            train_bar.set_description(
                desc="[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f"
                % (
                    epoch,
                    NUM_EPOCHS,
                    running_results["d_loss"] / running_results["batch_sizes"],
                    running_results["g_loss"] / running_results["batch_sizes"],
                    running_results["d_score"] / running_results["batch_sizes"],
                    running_results["g_score"] / running_results["batch_sizes"],
                )
            )

        netG.eval()

        with torch.no_grad():
            val_bar = tqdm(val_loader, total=len(val_loader))
            valing_results = {
                "mse": 0,
                "ssims": 0,
                "psnr": 0,
                "ssim": 0,
                "batch_sizes": 0,
            }
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results["batch_sizes"] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                # Forward
                sr = netG(lr)
                # Loss & metrics
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results["mse"] += batch_mse * batch_size
                batch_ssim = ssim(sr, hr).item()

                valing_results["ssims"] += batch_ssim * batch_size
                valing_results["psnr"] = 10 * math.log10(
                    (hr.max() ** 2)
                    / (valing_results["mse"] / valing_results["batch_sizes"])
                )
                valing_results["ssim"] = (
                    valing_results["ssims"] / valing_results["batch_sizes"]
                )
                val_bar.set_description(
                    desc="[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f"
                    % (valing_results["psnr"], valing_results["ssim"])
                )

                val_images.extend(
                    [
                        display_transform(val_hr_restore.squeeze(0)),
                        display_transform(hr.data.cpu().squeeze(0)),
                        display_transform(sr.data.cpu().squeeze(0)),
                    ]
                )
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc="[saving training results]")
            index = 1
            for image in val_save_bar:
                image = torchvision.utils.make_grid(image, nrow=3, padding=5)
                torchvision.utils.save_image(
                    image,
                    results_dir / f"epoch_{epoch}_index_{index}.png",
                    padding=5,
                )
                index += 1

        # save model parameters
        netG.train()
        netD.train()
        results["d_loss"].append(
            running_results["d_loss"] / running_results["batch_sizes"]
        )
        results["g_loss"].append(
            running_results["g_loss"] / running_results["batch_sizes"]
        )
        results["d_score"].append(
            running_results["d_score"] / running_results["batch_sizes"]
        )
        results["g_score"].append(
            running_results["g_score"] / running_results["batch_sizes"]
        )
        results["psnr"].append(valing_results["psnr"])
        results["ssim"].append(valing_results["ssim"])

        _save_checkpoint_atomic(
            _checkpoint_path(checkpoint_dir, UPSCALE_FACTOR, epoch, "netG"),
            {
                "epoch": epoch,
                "model": netG.state_dict(),
                "optimizer": optimizerG.state_dict(),
                "results": results,
            },
        )
        _save_checkpoint_atomic(
            _checkpoint_path(checkpoint_dir, UPSCALE_FACTOR, epoch, "netD"),
            {
                "epoch": epoch,
                "model": netD.state_dict(),
                "optimizer": optimizerD.state_dict(),
            },
        )

        if epoch % 10 == 0 and epoch != 0:
            _save_results_csv(logs_dir, opt.run_name, results)

    _save_results_csv(logs_dir, opt.run_name, results)
    _save_checkpoint_atomic(weights_out, {"model": netG.state_dict()})
    print(f"Saved generator weights to {weights_out}")
    if not opt.keep_run_artifacts:
        _cleanup_run_artifacts(
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            weights_out=weights_out,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Swift-SRGAN')
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='super resolution upscale factor')
    parser.add_argument('--crop_size', default=96, type=int, help='training images crop size')
    parser.add_argument('--batch_size', default=16, type=int, help='training batch size')
    parser.add_argument('--num_epochs', default=100, type=int, help='number of epochs to train')
    parser.add_argument('--train-dir', type=Path, default=Path('./dataset/train'), help='training images directory')
    parser.add_argument('--valid-dir', type=Path, default=Path('./dataset/valid'), help='validation images directory')
    parser.add_argument('--weights-out', type=Path, default=Path('../weights/swift_srgan_4x.pth'), help='final generator weights path')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('./checkpoints'), help='per-epoch checkpoint directory')
    parser.add_argument('--results-dir', type=Path, default=Path('./results'), help='validation image output directory')
    parser.add_argument('--logs-dir', type=Path, default=Path('./logs'), help='training metrics CSV directory')
    parser.add_argument('--run-name', default='ssrgan_4x', help='prefix used for logs and run folders')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from the latest checkpoint in --checkpoint-dir',
    )
    parser.add_argument(
        '--keep-run-artifacts',
        action='store_true',
        help='Keep checkpoints and validation images after training completes',
    )
    opt = parser.parse_args()
    main(opt)