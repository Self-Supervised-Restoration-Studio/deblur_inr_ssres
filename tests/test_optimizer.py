"""Tests for DeblurINROptimizer."""

import pytest
import torch

from deblur_inr_ssres.config import DeblurINRConfig, OptimizationStage
from deblur_inr_ssres.optimizer import DeblurINROptimizer


def _make_config():
    return DeblurINRConfig(
        kernel_size=8,
        num_iterations=30,
        hidden_features=16,
        hidden_layers=1,
        skip_channels=[32, 32],
        stages=[
            OptimizationStage(
                start_iter=0, end_iter=10, scale=0.5, optimize_kernel=True, optimize_image=False
            ),
            OptimizationStage(
                start_iter=10, end_iter=20, scale=0.5, optimize_kernel=True, optimize_image=True
            ),
            OptimizationStage(
                start_iter=20,
                end_iter=30,
                scale=1.0,
                optimize_kernel=True,
                optimize_image=True,
                kernel_lr=2.5e-5,
                image_lr=2.5e-3,
            ),
        ],
    )


class TestDeblurINROptimizer:
    def test_deblur_output_shapes(self):
        config = _make_config()
        optimizer = DeblurINROptimizer(config, device="cpu")
        blurred = torch.rand(1, 1, 32, 32)
        image, kernel = optimizer.deblur(blurred, show_progress=False)
        assert image.shape == (1, 1, 32, 32)
        assert kernel.shape[0] == 1
        assert kernel.shape[1] == 1

    def test_kernel_sums_to_one(self):
        config = _make_config()
        optimizer = DeblurINROptimizer(config, device="cpu")
        blurred = torch.rand(1, 1, 32, 32)
        _, kernel = optimizer.deblur(blurred, show_progress=False)
        assert kernel.sum().item() == pytest.approx(1.0, abs=0.05)

    def test_unbatched_input(self):
        config = _make_config()
        optimizer = DeblurINROptimizer(config, device="cpu")
        blurred = torch.rand(1, 32, 32)  # (C, H, W) without batch
        image, _kernel = optimizer.deblur(blurred, show_progress=False)
        assert image.shape == (1, 1, 32, 32)

    def test_deblur_3d_output_shapes(self):
        config = DeblurINRConfig(
            spatial_dims=3,
            kernel_size=8,
            num_iterations=5,
            hidden_features=16,
            hidden_layers=1,
            skip_channels=[16, 16],
            num_frequencies=4,
            stages=[
                OptimizationStage(
                    start_iter=0, end_iter=5, scale=1.0,
                    optimize_kernel=True, optimize_image=True,
                ),
            ],
        )
        optimizer = DeblurINROptimizer(config, device="cpu")
        # Use 16^3 so the SkipNetwork3D output (halved to ~12) still exceeds kernel=9
        blurred = torch.rand(1, 1, 16, 16, 16)
        image, kernel = optimizer.deblur(blurred, show_progress=False)
        assert image.ndim == 5
        assert image.shape[:2] == (1, 1)
        assert kernel.ndim == 5

    def test_callbacks(self):
        config = _make_config()
        optimizer = DeblurINROptimizer(config, device="cpu")
        blurred = torch.rand(1, 1, 32, 32)
        callback_calls = []

        def my_callback(iteration, loss, model, image, kernel):
            callback_calls.append(iteration)

        optimizer.deblur(blurred, callbacks=[my_callback], show_progress=False)
        assert len(callback_calls) == config.num_iterations

    def test_deblur_image_convenience(self):
        from deblur_inr_ssres import deblur_image

        # 128x128 needed: default 5-level SkipNetwork requires spatial >= 2
        # at the deepest level (128 * 0.25 scale = 32, padded ~34, after 5
        # downsamples → min spatial = 2 which satisfies BatchNorm).
        image, _kernel = deblur_image(
            torch.rand(1, 1, 128, 128),
            kernel_size=8,
            num_iterations=3,
            device="cpu",
            show_progress=False,
        )
        assert image.shape == (1, 1, 128, 128)
