"""Command-line entry points for BGDM."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
import time
from dataclasses import replace
from http.client import HTTPException
from pathlib import Path

from .assets import checkpoint_path, dataset_path, default_asset_dir, download_task
from .tasks import TASKS, Task


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pretrained BGDM on MRI or CT.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the pretrained tasks.")
    download = commands.add_parser("download", help="Download and verify the published weights.")
    check = commands.add_parser("check", help="Load the checkpoint and run a denoiser pass.")
    run = commands.add_parser("reconstruct", help="Reconstruct one or more 2D samples.")
    for command in (download, check, run):
        command.add_argument("task", choices=tuple(TASKS))
        command.add_argument("--assets-dir", type=Path, default=default_asset_dir(),
                             help="Weight and test data cache (default: %(default)s).")
    download.add_argument("--with-data", action="store_true", help="Also download the published test archive.")
    for command in (check, run):
        command.add_argument("--checkpoint", type=Path, help="Override the pretrained weight path.")
        command.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N (default: auto).")
        command.add_argument("--precision", choices=("fp32", "fp16"), default="fp32",
                             help="Denoiser precision. FP16 requires CUDA.")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Simulate measurements from a reference .npy or .npz array.")
    source.add_argument("--synthetic", action="store_true", help="Use one analytic phantom.")
    source.add_argument("--kspace", type=Path, help="Reconstruct full-grid MRI k-space; requires --mask.")
    run.add_argument("--key", default="all_imgs", help="Array key for .npz input (default: %(default)s).")
    run.add_argument("--normalization", choices=("none", "uint8", "unit", "max-magnitude", "clip-unit"),
                     help="Reference image scaling. Defaults to the task's original preprocessing.")
    run.add_argument("--start", type=int, default=0, help="Index of the first image, starting from zero.")
    run.add_argument("--num-samples", type=int, default=1, help="Number of images (default: %(default)s).")
    run.add_argument("--steps", type=int, help="Number of reverse diffusion steps (preset default if omitted).")
    run.add_argument("--method", choices=("bgdm", "ddnm", "dps"), default="bgdm")
    run.add_argument("--eta", type=float, help="DDIM stochasticity in [0, 1] (preset default if omitted).")
    run.add_argument("--sigma-y", type=float, default=0.0,
                     help="Gaussian noise scale to simulate, or noise scale of provided measurements.")
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--views", type=int, help="Number of sparse CT projections.")
    sensing = run.add_mutually_exclusive_group()
    sensing.add_argument("--acceleration", type=float, help="MRI line or area acceleration.")
    sensing.add_argument("--mask", type=Path, help="Override the MRI mask with a binary .npy or .npz array.")
    run.add_argument("--mask-type", choices=("uniform1d", "gaussian1d"), help="MRI Cartesian mask type.")
    run.add_argument("--mask-key", default="mask", help="Array key for an .npz MRI mask.")
    run.add_argument("--output", type=Path, required=True, help="New directory for reconstructions and metrics.")
    run.add_argument("--no-progress", action="store_true", help="Disable the progress bar.")
    return parser


def _device(selection):
    import torch

    if selection == "auto":
        selection = "cuda" if torch.cuda.is_available() else "cpu"
    if selection == "cpu":
        # Avoid excessive thread-launch overhead for one-slice CPU inference.
        torch.set_num_threads(min(torch.get_num_threads(), 8))
        return torch.device("cpu")
    if selection == "cuda" or (selection.startswith("cuda:") and selection[5:].isdigit()):
        index = int(selection[5:]) if selection != "cuda" else None
        if not torch.cuda.is_available() or (index is not None and index >= torch.cuda.device_count()):
            raise ValueError(f"Requested device {selection} is unavailable. Try --device cpu.")
        return torch.device(selection)
    raise ValueError("Device must be auto, cpu, cuda, or cuda:N.")


def _validate_run(args: argparse.Namespace, task: Task) -> None:
    if args.start < 0 or args.num_samples < 1:
        raise ValueError("Use a non-negative start index and a positive number of samples.")
    if args.synthetic and (args.start != 0 or args.num_samples != 1 or args.normalization is not None):
        raise ValueError("--synthetic produces one pre-normalized sample with index 0.")
    if args.kspace is not None and (
        args.mask is None or args.normalization is not None or task.name == "lidc"
    ):
        raise ValueError("--kspace requires an MRI task and --mask. Supply already normalized k-space.")
    if args.mask is not None and args.mask_type is not None:
        raise ValueError("Choose --mask or --mask-type.")
    if args.views is not None and (task.name != "lidc" or not 1 <= args.views <= 180):
        raise ValueError("--views accepts 1 to 180 views with the lidc task.")
    if task.name == "lidc" and (args.acceleration is not None or args.method != "bgdm"):
        raise ValueError("The lidc preset uses --views and supports the bgdm method.")
    if not 1 <= args.steps <= task.diffusion_steps:
        raise ValueError(f"Use between 1 and {task.diffusion_steps} steps.")
    if not 0 <= args.eta <= 1 or not math.isfinite(args.sigma_y) or args.sigma_y < 0:
        raise ValueError("Eta must be in [0, 1] and noise sigma must be finite and non-negative.")
    if not 0 <= args.seed < 2**63:
        raise ValueError("Seed must be a non-negative 63-bit integer.")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Output path already exists: {args.output}. Choose a new directory.")


def _check(task, checkpoint, device, fp16) -> None:
    import torch

    from .model import load_model, predict_noise

    model = load_model(task, checkpoint, device, fp16=fp16)
    shape = (1, 2 if task.is_complex else 1, task.image_size, task.image_size)
    with torch.inference_mode():
        image = torch.zeros(shape, device=device)
        prediction = predict_noise(model, image, task.diffusion_steps // 2)
    if prediction.shape != image.shape or not torch.isfinite(prediction).all():
        raise RuntimeError("Denoiser produced an invalid prediction.")
    print(f"{task.name}: verified {checkpoint} on {device}. Output shape {tuple(prediction.shape)}.")


def _run(args, task, checkpoint, device) -> None:
    import numpy as np
    import torch
    from tqdm import tqdm

    from .data import load_array, load_images, synthetic_image
    from .degradation import build_degradation
    from .model import load_model
    from .output import reference_metrics, save_array, save_image
    from .sampler import reconstruct

    args.steps = task.steps if args.steps is None else args.steps
    args.eta = task.eta if args.eta is None else args.eta
    _validate_run(args, task)
    if args.acceleration is not None:
        task = replace(task, acceleration=args.acceleration)
    if args.views is not None:
        task = replace(task, views=args.views)

    mask = load_array(args.mask, args.mask_key) if args.mask is not None else None
    operator = build_degradation(task, device=device, seed=args.seed, mask_type=args.mask_type, mask=mask)
    external = args.kspace is not None
    input_path = args.kspace or args.input or dataset_path(args.assets_dir, task)
    normalization = "none" if external else args.normalization or task.normalization
    if args.synthetic:
        samples = synthetic_image(task.image_size, is_complex=task.is_complex)
    else:
        if not input_path.is_file():
            raise FileNotFoundError(
                f"Input not found: {input_path}. Use the published data, --input, or --synthetic."
            )
        samples = load_images(
            input_path, size=task.image_size, is_complex=external or task.is_complex,
            normalization=normalization, key=args.key, start=args.start, limit=args.num_samples,
        )
        if len(samples) != args.num_samples:
            raise ValueError(f"Only {len(samples)} sample(s) available from index {args.start}.")
    model = load_model(task, checkpoint, device, fp16=args.precision == "fp16")

    args.output.mkdir(parents=True, exist_ok=False)
    if operator.mask is not None:
        np.save(args.output / "mask.npy", operator.mask[0, 0].cpu().numpy(), allow_pickle=False)
    metadata = {
        "task": task.name,
        "method": args.method,
        "steps": args.steps,
        "eta": args.eta,
        "sigma_y": args.sigma_y,
        "seed": args.seed,
        "device": str(device),
        "precision": args.precision,
        "checkpoint": str(checkpoint),
        "input": "synthetic" if args.synthetic else str(input_path),
        "input_mode": "kspace" if external else "reference",
        "normalization": "none" if args.synthetic else normalization,
        "start": args.start,
        "num_samples": args.num_samples,
        "acceleration": None if mask is not None else task.acceleration,
        "mask_type": None if mask is not None else args.mask_type or task.mask_type,
        "views": task.views,
    }
    with (args.output / "run.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")
    with (args.output / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("sample", "psnr", "ssim", "relative_measurement_residual", "seconds"))
        for index, sample in enumerate(samples, start=args.start):
            started = time.perf_counter()
            sample = sample[None].to(device)
            seed = (args.seed + index) % 2**63
            if external:
                reference = None
                measurement = operator.prepare_measurement(sample)
            else:
                reference = sample
                measurement = operator.forward(reference)
                if args.sigma_y:
                    generator = torch.Generator(device=device).manual_seed(seed ^ 0xD32A)
                    noise_values = measurement.real if task.name == "lidc" else measurement
                    noise = torch.randn(noise_values.shape, dtype=noise_values.dtype, device=device,
                                        generator=generator)
                    measurement = operator.prepare_measurement(measurement + args.sigma_y * noise)
            preview = operator.preview(measurement)
            with tqdm(total=args.steps, desc=f"Sample {index}", disable=args.no_progress) as bar:
                result = reconstruct(
                    model, operator, measurement, task, steps=args.steps, method=args.method,
                    eta=args.eta, sigma_y=args.sigma_y, seed=seed, progress=bar.update,
                )
            prefix = f"{index:06d}"
            save_image(preview, args.output / f"{prefix}_measurement.png")
            save_image(result, args.output / f"{prefix}_reconstruction.png")
            save_array(result, args.output / f"{prefix}_reconstruction.npy")
            psnr = ssim = ""
            if reference is not None:
                psnr_value, ssim_value = reference_metrics(result, reference)
                psnr, ssim = f"{psnr_value:.4f}", f"{ssim_value:.4f}"
                save_image(reference, args.output / f"{prefix}_reference.png")
            residual = torch.linalg.vector_norm(operator.residual(result, measurement))
            denom = torch.linalg.vector_norm(measurement).clamp_min(1e-12)
            writer.writerow((index, psnr, ssim, f"{(residual / denom).item():.6g}",
                             f"{time.perf_counter() - started:.2f}"))
            file.flush()
    print(f"Saved {len(samples)} reconstruction(s) to {args.output}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list":
        for task in TASKS.values():
            print(f"{task.name:<15} {task.image_size:>3} × {task.image_size:<3}  {task.description}")
        return 0

    task = TASKS[args.task]
    try:
        if args.command == "download":
            download_task(args.assets_dir, task, with_data=args.with_data)
        else:
            device = _device(args.device)
            checkpoint = args.checkpoint or checkpoint_path(args.assets_dir, task)
            if args.command == "check":
                _check(task, checkpoint, device, args.precision == "fp16")
            else:
                _run(args, task, checkpoint, device)
        return 0
    except (OSError, ValueError, RuntimeError, HTTPException, pickle.UnpicklingError) as error:
        print(f"BGDM: {error}", file=sys.stderr)
        return 1
