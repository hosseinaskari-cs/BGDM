"""Load the original one-channel BGDM denoisers."""

from pathlib import Path

import torch

from ._guided.unet import UNetModel
from .tasks import Task


def load_model(task: Task, checkpoint: Path, device: torch.device, *, fp16: bool) -> UNetModel:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}. See the README for download links.")
    if fp16 and device.type != "cuda":
        raise ValueError("FP16 denoiser inference requires CUDA. Use FP32 on CPU.")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"Expected a tensor state dictionary in {checkpoint}.")
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}

    model = UNetModel(
        image_size=task.image_size,
        in_channels=1,
        model_channels=32,
        out_channels=2 if task.learn_sigma else 1,
        num_res_blocks=3,
        attention_resolutions=tuple(task.image_size // resolution for resolution in task.attention),
        channel_mult=task.channel_mult,
        num_heads=4,
        use_scale_shift_norm=True,
        resblock_updown=task.resblock_updown,
        use_fp16=fp16,
        use_checkpoint=True,
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError(f"{checkpoint} does not match the {task.name} denoiser:\n{error}") from error
    if fp16:
        model.convert_to_fp16()
    return model.to(device).eval().requires_grad_(False)


def predict_noise(model: UNetModel, image: torch.Tensor, timestep: int) -> torch.Tensor:
    """Apply the one-channel checkpoint to each image component independently."""
    batch, channels, height, width = image.shape
    components = image.reshape(batch * channels, 1, height, width)
    time = torch.full((len(components),), timestep, dtype=torch.long, device=image.device)
    return model(components, time)[:, :1].reshape_as(image)
