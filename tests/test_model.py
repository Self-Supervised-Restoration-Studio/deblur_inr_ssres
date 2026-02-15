"""Tests for DeblurINRModel."""

import torch

from deblur_inr_ssres.config import DeblurINRConfig
from deblur_inr_ssres.model import DeblurINRModel


class TestDeblurINRModel:
    def test_forward_returns_tuple(self):
        config = DeblurINRConfig(
            kernel_size=8,
            skip_channels=[32, 32],
            hidden_features=16,
            hidden_layers=1,
        )
        model = DeblurINRModel(config=config)
        x = torch.randn(1, 64, 32, 32)
        image, kernel = model(x, kernel_size=8)
        assert isinstance(image, torch.Tensor)
        assert isinstance(kernel, torch.Tensor)

    def test_kernel_shape(self):
        config = DeblurINRConfig(
            kernel_size=8,
            skip_channels=[32, 32],
            hidden_features=16,
            hidden_layers=1,
        )
        model = DeblurINRModel(config=config)
        x = torch.randn(1, 64, 32, 32)
        _, kernel = model(x, kernel_size=8)
        assert kernel.shape == (1, 1, 8, 8)

    def test_parameter_groups_disjoint(self):
        config = DeblurINRConfig(
            kernel_size=8,
            skip_channels=[32, 32],
            hidden_features=16,
            hidden_layers=1,
        )
        model = DeblurINRModel(config=config)
        kernel_params = {id(p) for p in model.get_kernel_parameters()}
        image_params = {id(p) for p in model.get_image_parameters()}
        assert kernel_params.isdisjoint(image_params)

    def test_parameter_groups_cover_all(self):
        config = DeblurINRConfig(
            kernel_size=8,
            skip_channels=[32, 32],
            hidden_features=16,
            hidden_layers=1,
        )
        model = DeblurINRModel(config=config)
        all_params = {id(p) for p in model.parameters()}
        kernel_params = {id(p) for p in model.get_kernel_parameters()}
        image_params = {id(p) for p in model.get_image_parameters()}
        assert all_params == kernel_params | image_params

    def test_config_based_construction(self):
        config = DeblurINRConfig(
            kernel_size=16,
            image_channels=3,
            hidden_features=32,
            hidden_layers=2,
            skip_channels=[64, 64, 64],
        )
        model = DeblurINRModel(config=config)
        assert model.in_channels == 3
        assert model.out_channels == 3
        assert model.kernel_size == 16
        assert model.spatial_dims == 2

    def test_3d_forward(self):
        config = DeblurINRConfig(
            spatial_dims=3,
            kernel_size=8,
            skip_channels=[16, 16],
            hidden_features=16,
            hidden_layers=1,
            num_frequencies=4,
        )
        model = DeblurINRModel(config=config)
        # fourier_input_depth = num_frequencies * 2 * spatial_dims = 4 * 2 * 3 = 24
        x = torch.randn(1, 24, 8, 8, 8)
        image, kernel = model(x, kernel_size=8)
        assert image.ndim == 5
        assert image.shape[:2] == (1, 1)
        assert kernel.shape == (1, 1, 8, 8, 8)

    def test_direct_construction(self):
        model = DeblurINRModel(
            kernel_size=8,
            image_channels=1,
            hidden_features=16,
            hidden_layers=1,
            skip_channels=[32, 32],
        )
        assert model.in_channels == 1
        assert model.kernel_size == 8
