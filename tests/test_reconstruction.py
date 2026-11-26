from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch

from bgdm._guided.unet import UNetModel
from bgdm.data import load_images, synthetic_image
from bgdm.degradation import build_degradation
from bgdm.model import load_model, predict_noise
from bgdm.sampler import reconstruct
from bgdm.tasks import TASKS


CPU = torch.device("cpu")
SIZE = 16


class TinyDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.timesteps = []

    def forward(self, image, timestep):
        self.timesteps.append(timestep.clone())
        noise = 0.05 * image
        return torch.cat((noise, torch.full_like(noise, 1000)), dim=1)


class ImageInputTests(TestCase):
    def test_complex_formats_keep_phase_and_scale_when_limited(self):
        native = np.empty((2, 1, SIZE, SIZE), dtype=np.complex64)
        native[0] = 3 + 4j
        native[1] = 6 + 8j
        split = np.concatenate((native.real, native.imag), axis=1)

        with TemporaryDirectory() as directory:
            for array in (native, split):
                with self.subTest(dtype=array.dtype):
                    path = Path(directory) / "images.npz"
                    np.savez(path, slices=array)
                    loaded = load_images(
                        path, size=SIZE, is_complex=True,
                        normalization="max-magnitude", key="slices", limit=1,
                    )
                    torch.testing.assert_close(loaded, torch.from_numpy(native[:1] / 10))

    def test_rejects_wrong_spatial_shape_or_real_channel_count(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "images.npy"
            for shape, error in [((SIZE, SIZE + 1), "got shape"), ((1, 2, SIZE, SIZE), "one real channel")]:
                with self.subTest(shape=shape):
                    np.save(path, np.zeros(shape, dtype=np.float32))
                    with self.assertRaisesRegex(ValueError, error):
                        load_images(path, size=SIZE, is_complex=False, normalization="none", key="unused")


class MeasurementTests(TestCase):
    def test_complex_mri_projection_has_expected_sampled_frequencies(self):
        task = replace(TASKS["fastmri-knee"], image_size=SIZE)
        mask = torch.zeros((SIZE, SIZE), dtype=torch.bool)
        mask[:, 3::4] = True
        operator = build_degradation(task, device=CPU, seed=5, mask=mask)
        reference = synthetic_image(SIZE, is_complex=True)
        proposal = reference.roll(3, dims=-1) * 0.5j
        measurement = operator.forward(reference)

        projection = operator.project(proposal, measurement)
        torch.testing.assert_close(operator.forward(projection), measurement)
        damped = operator.project(proposal, measurement, damping=3.0)
        torch.testing.assert_close(operator.forward(damped), (measurement + 3 * operator.forward(proposal)) / 4)

    def test_ct_projection_reduces_residual_for_a_phantom(self):
        task = replace(TASKS["lidc"], image_size=SIZE, views=7, expansion=2)
        operator = build_degradation(task, device=CPU, seed=5)
        reference = synthetic_image(SIZE, is_complex=False)
        measurement = operator.prepare_measurement(operator.forward(reference))
        initial = torch.zeros_like(reference)

        projected = operator.project(initial, measurement)
        self.assertEqual(projected.shape, reference.shape)
        residual = torch.linalg.vector_norm(operator.residual(projected, measurement))
        self.assertLess(residual, torch.linalg.vector_norm(measurement))


class SamplerTests(TestCase):
    def test_wrapped_checkpoint_supports_gradients_with_frozen_weights(self):
        size = 4
        task = replace(TASKS["brats"], image_size=size, channel_mult=(1,), attention=())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(13)
            fixture = UNetModel(
                image_size=size, in_channels=1, model_channels=32, out_channels=2,
                num_res_blocks=3, attention_resolutions=(), channel_mult=(1,),
                num_heads=4, use_scale_shift_norm=True,
            )
        with torch.no_grad():
            fixture.out[-1].weight[0, 0, 1, 1] = 1
        state = {f"module.{name}": value for name, value in fixture.state_dict().items()}

        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pt"
            torch.save({"state_dict": state}, path)
            model = load_model(task, path, CPU, fp16=False)

        self.assertFalse(model.training)
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        image = torch.linspace(-1, 1, size * size).reshape(1, 1, size, size).requires_grad_()
        prediction = predict_noise(model, image, timestep=5)
        (gradient,) = torch.autograd.grad(prediction.square().sum(), image)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().max().item(), 0)

    def test_complex_noise_prediction_applies_the_denoiser_to_both_components(self):
        # Use multiple images and distinct channel values to catch swapped axes.
        image = torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3)
        model = TinyDenoiser()
        prediction = predict_noise(model, image, timestep=17)

        torch.testing.assert_close(prediction, 0.05 * image)
        torch.testing.assert_close(model.timesteps[0], torch.full((4,), 17))

    def test_sampler_methods_produce_finite_complex_images(self):
        task = replace(TASKS["fastmri-knee"], image_size=SIZE, diffusion_steps=24)
        operator = build_degradation(task, device=CPU, seed=4)
        measurement = operator.forward(synthetic_image(SIZE, is_complex=True))
        options = dict(steps=4, eta=0.5, sigma_y=0.0, seed=17)

        for method in ("bgdm", "ddnm", "dps"):
            with self.subTest(method=method):
                model = TinyDenoiser()
                result = reconstruct(model, operator, measurement, task, method=method, **options)
                self.assertEqual(result.shape, (1, 1, SIZE, SIZE))
                self.assertTrue(result.is_complex())
                self.assertTrue(torch.isfinite(result).all())
                self.assertEqual(len(model.timesteps), options["steps"])
                if method in {"bgdm", "ddnm"}:
                    torch.testing.assert_close(operator.forward(result), measurement)

        reference = reconstruct(TinyDenoiser(), operator, measurement, task, method="bgdm", **options)
        repeated = reconstruct(TinyDenoiser(), operator, measurement, task, method="bgdm", **options)
        torch.testing.assert_close(reference, repeated, rtol=0, atol=0)
