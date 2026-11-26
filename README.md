# Bi-Level Guided Diffusion Models (BGDM)

Pretrained PyTorch inference for **Training-Free Medical Image Inverses via Bi-Level Guided Diffusion Models**, by Hossein Askari, Fred Roosta, and Hongfu Sun (WACV 2025). BGDM uses a pretrained denoiser and a measurement operator to reconstruct images from incomplete measurements. [Paper](https://openaccess.thecvf.com/content/WACV2025/html/Askari_Training-Free_Medical_Image_Inverses_via_Bi-Level_Guided_Diffusion_Models_WACV_2025_paper.html)

## Quick start

Run these commands from the repository root. Use Python 3.10 or later. The example installs the CPU wheel. For CUDA, replace the PyTorch installation line with the command that matches your NVIDIA driver. [PyTorch installation guide](https://pytorch.org/get-started/locally/)

```bash
python -m venv .venv
source .venv/bin/activate
# For CPU-only setup
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
```

List the tasks, fetch one pretrained checkpoint, and run a small demo:

```bash
bgdm list
bgdm download brats
bgdm check brats
bgdm reconstruct brats --synthetic --steps 8 --output outputs/brats-phantom
```

The demo uses a generated phantom and needs only the checkpoint. A short run checks the pipeline. Use the preset step count to evaluate reconstruction quality. The runner selects CUDA when available. Use `--device cpu` or `--device cuda:0` to override it. FP32 is the default. CUDA users can opt into `--precision fp16`. Both `python -m bgdm` and `python main.py` accept the same commands.

### Debug BraTS in VS Code with Conda

On an x86-64 Linux or Windows machine, run these commands from the repository root to create the workspace Conda environment with the CUDA 12.1 build of PyTorch:

```bash
conda env create --prefix ./.conda --file environment.yml
conda run --prefix ./.conda python -m bgdm download brats --with-data
```

Open the repository folder in VS Code with the Python and Python Debugger extensions installed. In **Run and Debug**, select **BGDM: Debug BraTS MRI** and press **F5**. The launch file uses the workspace Conda interpreter and runs one BraTS test slice with the standard BGDM settings. Set a breakpoint in `reconstruct()` in [bgdm/sampler.py](bgdm/sampler.py) to step through sampling. Accept `outputs/brats-debug` for the first run and choose a fresh output directory when you rerun.

## Pretrained tasks

| Preset | Task | Resolution | Default measurement | Steps |
| --- | --- | --- | --- | ---: |
| `brats` | Real-valued MRI reconstruction | 240 × 240 | 8× Cartesian acceleration | 200 |
| `fastmri-knee` | Complex-valued MRI reconstruction | 320 × 320 | Nominal 4× Gaussian Cartesian acceleration | 30 |
| `fastmri-brain` | MRI super-resolution | 320 × 320 | Central 80 × 80 k-space crop | 50 |
| `lidc` | Sparse-view CT reconstruction | 320 × 320 | 23 parallel-beam views | 350 |

To download each pretrained checkpoint with its published reference images and reconstruct the first slice using the preset settings:

```bash
for task in brats fastmri-knee fastmri-brain lidc; do
    bgdm download "$task" --with-data
    bgdm reconstruct "$task" --output "outputs/$task"
done
```

The largest test archive, fastMRI knee, is about 782 MiB. Use `--start 10 --num-samples 3` for slices 10–12 or `--num-samples 1000` for a full test archive. The seed controls the synthetic measurement mask and the starting noise. With the same settings, a slice uses the same seed whether run individually or as part of a range.

For the knee dataset, use `--mask-type uniform1d --acceleration 8` to change the acquisition. Gaussian sampling can redraw the same line, so the realized acceleration is determined by the saved mask. For brain super-resolution, `--acceleration 16` refers to the retained **area** of k-space and corresponds to a 4× reduction along each axis. For CT, use `--views 10` to choose 10 views. The default method is `bgdm`. The MRI presets also accept `--method ddnm` and `--method dps`.

### Existing or manual downloads

By default, assets are read from `~/.cache/bgdm` (or `$XDG_CACHE_HOME/bgdm` if set). The downloader checks each file against its SHA-256. Use `--assets-dir /path/to/bgdm-assets` to change the cache directory across all commands. To supply a checkpoint by path, add `--checkpoint /path/to/model.pt` to `check` or `reconstruct`.

| Preset | Weight path under the cache | Data path under the cache |
| --- | --- | --- |
| `brats` | [`checkpoints/BraTS/ema_0.9999_2020000.pt`](https://drive.google.com/file/d/1pwaDNuDIuAJzCLkrVwj4DIxegAlwsLbd/view) | [`test_data/BraTS.npz`](https://drive.google.com/file/d/1RRdop20U6bwfWWgO0DzHdb98EJR5igAs/view) |
| `fastmri-knee` | [`checkpoints/fastMRI_knee/model2230000.pt`](https://drive.google.com/file/d/1AgV6Smd_NiwJnUSZc-PAeeu69fqlawUh/view) | [`test_data/fastMRI_knee.npz`](https://drive.google.com/file/d/1Hom4MdfUrvb1jTuA3ExCG5z0s_AaFour/view) |
| `fastmri-brain` | [`checkpoints/fastMRI_brain/ema_0.9999_340000.pt`](https://drive.google.com/file/d/1lFOYJEDk_ksThgJJvcRcaEm3AL7R-3nk/view) | [`test_data/fastMRI_brain.npz`](https://drive.google.com/file/d/19U3JGAL0KqJtZ0PlB1yDCc2m5s5SHWFg/view) |
| `lidc` | [`checkpoints/LIDC/model4200000.pt`](https://drive.google.com/file/d/11xh4kCYx4Xsbgb1T5m0DmDgqa63uBOiv/view) | [`test_data/LIDC_320_vv.npz`](https://drive.google.com/file/d/1xocXIAsKZXj3cMiL7zOrd79I-4Kw80Mn/view) |

## Use your own data

### Simulate an inverse problem from reference images

```bash
bgdm reconstruct brats --input images.npy --normalization unit \
    --acceleration 8 --num-samples 3 --output outputs/custom-brats
```

Single-slice and batch arrays can be shaped `[H,W]`, `[N,H,W]`, or `[N,1,H,W]` at the selected preset's native resolution. Complex knee images can also use two real channels `[N,2,H,W]` ordered real then imaginary. The default NPZ key is `all_imgs`. Use `--key` for another key. The CLI reads NumPy arrays as single-channel images. Choose the preset for your modality and prepare any resizing, cropping, or coil combination before running it.

With the original archives, keep the preset normalization. BraTS and LIDC divide intensities by 255, knee divides by the maximum magnitude of the full archive, and brain clips intensities to [0, 1]. For custom reference images use `--normalization unit` to validate real inputs already in [0, 1], `--normalization uint8` for values in [0, 255], or `--normalization none` to keep values in your input units.

### Reconstruct from acquired MRI k-space

```bash
bgdm reconstruct fastmri-knee --kspace acquired.npy --mask mask.npy \
    --output outputs/acquired-knee
```

Supply a single-coil or coil-combined, full-resolution grid of **centered, orthonormal** k-space in the model's intensity scale. The supported shapes match the complex image inputs above. The binary mask must be `[H,W]` with optional singleton leading axes. For NPZ masks, the default key is `mask` and can be changed with `--mask-key`. Values outside the mask are discarded. An image-to-grid conversion for synthetic measurements in NumPy is `fftshift(fft2(ifftshift(image), norm="ortho"))` for a single 2D slice. For batches, apply the shifts and FFT on the last two axes. The CLI simulates CT projections from reference images.

## Outputs and scope

Use a new directory for `--output`. Each run writes `run.json` with its settings, `metrics.csv` with measurements of data consistency and any reference metrics, and `mask.npy` for MRI. Each slice has a measurement preview, a reconstruction PNG, and the unclipped numerical reconstruction as a 2D NPY. The knee NPY retains complex phase. When a reference image is provided, the output also includes its preview, PSNR, and SSIM. Knee metrics are computed on magnitudes after each image is scaled by its own maximum.

The release uses the original denoiser weights and the local BGDM sampling presets. Real-valued MRI enforces the conjugate structure of k-space in the consistency update. Sparse-view CT uses an independent PyTorch Fourier-slice simulation. Its discretization differs from the historical JAX operator, so its metrics should be reported with this geometry. These commands are intended for research use.

If a run fails to load a checkpoint, verify the preset and the case-sensitive filenames in the asset table. If memory is tight, process one slice at a time and use `--precision fp16` on CUDA or `--device cpu`. CPU inference can be significantly slower at the default step counts.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic arrays and run without checkpoint downloads. Use `bgdm check <preset>` to check a specific pretrained denoiser. The release was also tested on one slice per published archive at the preset step counts using Python 3.10, PyTorch 2.4.0, and an NVIDIA RTX 3090.

## Citation

```bibtex
@InProceedings{Askari_2025_WACV,
  author    = {Askari, Hossein and Roosta, Fred and Sun, Hongfu},
  title     = {Training-Free Medical Image Inverses via Bi-Level Guided Diffusion Models},
  booktitle = {Proceedings of the Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2025},
  month     = {February},
  pages     = {75--84}
}
```

See [third-party notices](THIRD_PARTY_NOTICES.md) for code provenance and upstream licenses. Pretrained weights and reference datasets are obtained separately from their published locations.
