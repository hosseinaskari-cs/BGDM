from dataclasses import replace
import unittest

import torch

from bgdm.degradation import build_degradation
from bgdm.tasks import TASKS


class DegradationTests(unittest.TestCase):
    def test_real_mri_projects_asymmetric_mask(self):
        for size in (7, 8):
            with self.subTest(size=size):
                task = replace(TASKS["brats"], image_size=size)
                center = size // 2
                mask = torch.zeros((size, size), dtype=torch.bool)
                mask[center, center + 1] = True
                mask[center, center] = True
                operator = build_degradation(task, device=torch.device("cpu"), seed=0, mask=mask)
                generator = torch.Generator().manual_seed(size)
                reference = torch.randn((2, 1, size, size), generator=generator)
                start = torch.randn(reference.shape, generator=generator)
                measured = operator.forward(reference)

                preview = operator.preview(measured)
                projected = operator.project(start, measured)
                self.assertFalse(preview.is_complex())
                self.assertFalse(projected.is_complex())
                torch.testing.assert_close(operator.forward(preview), measured)
                torch.testing.assert_close(operator.forward(projected), measured)

    def test_real_mri_damping_satisfies_normal_equations(self):
        size = 8
        task = replace(TASKS["brats"], image_size=size)
        mask = torch.zeros((size, size), dtype=torch.bool)
        mask[0, 0] = True     # A self-conjugate Nyquist bin
        mask[4, 5] = True     # One observed partner
        mask[2, 4] = True     # Both partners, with independent measurement values
        mask[6, 4] = True
        operator = build_degradation(task, device=torch.device("cpu"), seed=0, mask=mask)
        generator = torch.Generator().manual_seed(123)
        start = torch.randn((1, 1, size, size), generator=generator, dtype=torch.float64)
        measured = torch.randn(start.shape, generator=generator, dtype=torch.complex128) * mask
        damping = 0.37

        projected = operator.project(start, measured, damping=damping).requires_grad_()
        data_error = operator.residual(projected, measured).abs().square().sum()
        change = (projected - start).square().sum()
        gradient, = torch.autograd.grad(data_error + damping * change, projected)
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-12, rtol=0)

    def test_complex_mri_does_not_infer_conjugate_partner(self):
        size = 8
        task = replace(TASKS["fastmri-knee"], image_size=size)
        mask = torch.zeros((size, size), dtype=torch.bool)
        mask[4, 5] = True
        operator = build_degradation(task, device=torch.device("cpu"), seed=0, mask=mask)
        raw = torch.zeros((1, 1, size, size), dtype=torch.complex64)
        raw[..., 4, 5] = 1 + 2j
        raw[..., 4, 3] = 5j
        measured = operator.prepare_measurement(raw)
        expected = raw * mask

        torch.testing.assert_close(measured, expected)
        preview = operator.preview(measured)
        self.assertTrue(preview.is_complex())
        torch.testing.assert_close(operator.forward(preview), expected)
        centered = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(preview), norm="ortho"))
        torch.testing.assert_close(centered, expected)


if __name__ == "__main__":
    unittest.main()
