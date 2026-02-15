"""Tests for Deblur-INR Pydantic configuration."""

import pytest
from pydantic import ValidationError

from deblur_inr_ssres.config import DeblurINRConfig, OptimizationStage


class TestOptimizationStage:
    def test_defaults(self):
        stage = OptimizationStage(start_iter=0, end_iter=100, scale=0.5)
        assert stage.optimize_kernel is True
        assert stage.optimize_image is True
        assert stage.kernel_lr == 5e-5
        assert stage.image_lr == 5e-3


class TestDeblurINRConfig:
    def test_defaults(self):
        config = DeblurINRConfig()
        assert config.spatial_dims == 2
        assert config.kernel_size == 51
        assert config.num_iterations == 5000
        assert config.image_channels == 1
        assert len(config.stages) == 4

    def test_fourier_input_depth_2d(self):
        config = DeblurINRConfig(num_frequencies=16, spatial_dims=2)
        assert config.fourier_input_depth == 64  # 16 * 2 * 2

    def test_fourier_input_depth_3d(self):
        config = DeblurINRConfig(num_frequencies=8, spatial_dims=3)
        assert config.fourier_input_depth == 48  # 8 * 2 * 3

    def test_create_3d_default(self):
        config = DeblurINRConfig.create_3d_default()
        assert config.spatial_dims == 3
        assert config.kernel_size == 32
        assert config.hidden_features == 128
        assert len(config.stages) == 4

    def test_validation_kernel_size(self):
        with pytest.raises(ValidationError):
            DeblurINRConfig(kernel_size=1)

    def test_validation_num_iterations(self):
        with pytest.raises(ValidationError):
            DeblurINRConfig(num_iterations=0)

    def test_serialization_roundtrip(self):
        config = DeblurINRConfig(kernel_size=32, num_frequencies=8)
        data = config.model_dump()
        restored = DeblurINRConfig(**data)
        assert restored.kernel_size == 32
        assert restored.num_frequencies == 8
        assert restored.fourier_input_depth == config.fourier_input_depth

    def test_custom_stages(self):
        stages = [
            OptimizationStage(start_iter=0, end_iter=50, scale=0.5),
            OptimizationStage(start_iter=50, end_iter=100, scale=1.0),
        ]
        config = DeblurINRConfig(stages=stages, num_iterations=100)
        assert len(config.stages) == 2
        assert config.stages[0].scale == 0.5
