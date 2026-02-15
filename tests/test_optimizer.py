"""Tests for DeblurINROptimizer."""

import pytest
import torch

from deblur_inr_ssrs.config import DeblurINRConfig, OptimizationStage
from deblur_inr_ssrs.optimizer import DeblurINROptimizer


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

    def test_callbacks(self):
        config = _make_config()
        optimizer = DeblurINROptimizer(config, device="cpu")
        blurred = torch.rand(1, 1, 32, 32)
        callback_calls = []

        def my_callback(iteration, loss, model, image, kernel):
            callback_calls.append(iteration)

        optimizer.deblur(blurred, callbacks=[my_callback], show_progress=False)
        assert len(callback_calls) == config.num_iterations
