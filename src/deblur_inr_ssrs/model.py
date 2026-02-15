"""Combined Deblur-INR Model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from ssrs_toolbox import SIRENKernelEstimator, SkipNetwork, SkipNetwork3D
from torch import Tensor

if TYPE_CHECKING:
    from .config import DeblurINRConfig

SpatialDims = Literal[2, 3]


class DifferentiableCentralLayer(nn.Module):
    """Differentiable kernel centering via F.grid_sample.

    Shifts the kernel so its center of mass aligns with the spatial center.
    Uses bilinear interpolation for sub-pixel accuracy and full gradient flow,
    enabling the SIREN to learn pre-centered kernels via backprop through the shift.

    :param eps: Small value for numerical stability
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, kernel: Tensor) -> Tensor:
        """Center kernel around its center of mass.

        :param kernel: Input kernel (B, C, H, W)
        :returns: Centered and re-normalized kernel
        """
        _, _, H, W = kernel.shape

        # Compute center of mass
        y_grid = torch.arange(H, device=kernel.device, dtype=kernel.dtype)
        x_grid = torch.arange(W, device=kernel.device, dtype=kernel.dtype)
        yy, xx = torch.meshgrid(y_grid, x_grid, indexing="ij")

        k_sum = kernel.sum().clamp(min=self.eps)
        centroid_y = (kernel * yy).sum() / k_sum
        centroid_x = (kernel * xx).sum() / k_sum

        # Convert shift to normalized [-1, 1] coordinates
        shift_y = (centroid_y - H / 2.0) * 2.0 / H
        shift_x = (centroid_x - W / 2.0) * 2.0 / W

        # Build sampling grid with shift applied
        base_y = torch.linspace(-1, 1, H, device=kernel.device, dtype=kernel.dtype)
        base_x = torch.linspace(-1, 1, W, device=kernel.device, dtype=kernel.dtype)
        grid_y, grid_x = torch.meshgrid(base_y, base_x, indexing="ij")

        # grid_sample expects (N, H, W, 2) with (x, y) ordering
        grid = torch.stack([grid_x + shift_x, grid_y + shift_y], dim=-1).unsqueeze(0)

        centered = F.grid_sample(
            kernel, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        # Re-normalize (zeros padding can lose mass at edges)
        return centered / centered.sum().clamp(min=self.eps)


class DifferentiableCentralLayer3D(nn.Module):
    """3D version of DifferentiableCentralLayer for volumetric kernels.

    :param eps: Small value for numerical stability
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, kernel: Tensor) -> Tensor:
        """Center 3D kernel around its center of mass.

        :param kernel: Input kernel (B, C, D, H, W)
        :returns: Centered and re-normalized kernel
        """
        _, _, D, H, W = kernel.shape

        z_grid = torch.arange(D, device=kernel.device, dtype=kernel.dtype)
        y_grid = torch.arange(H, device=kernel.device, dtype=kernel.dtype)
        x_grid = torch.arange(W, device=kernel.device, dtype=kernel.dtype)
        zz, yy, xx = torch.meshgrid(z_grid, y_grid, x_grid, indexing="ij")

        k_sum = kernel.sum().clamp(min=self.eps)
        centroid_z = (kernel * zz).sum() / k_sum
        centroid_y = (kernel * yy).sum() / k_sum
        centroid_x = (kernel * xx).sum() / k_sum

        shift_z = (centroid_z - D / 2.0) * 2.0 / D
        shift_y = (centroid_y - H / 2.0) * 2.0 / H
        shift_x = (centroid_x - W / 2.0) * 2.0 / W

        base_z = torch.linspace(-1, 1, D, device=kernel.device, dtype=kernel.dtype)
        base_y = torch.linspace(-1, 1, H, device=kernel.device, dtype=kernel.dtype)
        base_x = torch.linspace(-1, 1, W, device=kernel.device, dtype=kernel.dtype)
        grid_z, grid_y, grid_x = torch.meshgrid(base_z, base_y, base_x, indexing="ij")

        # grid_sample 3D expects (N, D, H, W, 3) with (x, y, z) ordering
        grid = torch.stack(
            [grid_x + shift_x, grid_y + shift_y, grid_z + shift_z], dim=-1
        ).unsqueeze(0)

        centered = F.grid_sample(
            kernel, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return centered / centered.sum().clamp(min=self.eps)


class DeblurINRModel(nn.Module):
    """Combined Deblur-INR model with kernel estimation and image reconstruction.

    Composes ssrs_toolbox.SIRENKernelEstimator + ssrs_toolbox.SkipNetwork for
    joint blind deconvolution. Uses differentiable kernel centering (F.grid_sample)
    so gradients flow through the centering operation during test-time optimization.

    Supports both 2D and 3D via config.spatial_dims.
    """

    def __init__(
        self,
        config: DeblurINRConfig | None = None,
        kernel_size: int = 64,
        image_channels: int = 1,
        hidden_features: int = 64,
        hidden_layers: int = 3,
        omega_0: float = 30.0,
        skip_channels: list[int] | None = None,
    ):
        super().__init__()

        skip_num_channels = 16
        spatial_dims: SpatialDims = 2

        if config is not None:
            kernel_size = config.kernel_size
            image_channels = config.image_channels
            hidden_features = config.hidden_features
            hidden_layers = config.hidden_layers
            omega_0 = config.omega_0
            skip_channels = config.skip_channels
            skip_num_channels = config.skip_num_channels
            spatial_dims = config.spatial_dims

        if skip_channels is None:
            skip_channels = [128, 128, 128, 128, 128]

        self.kernel_size = kernel_size
        self._in_channels = image_channels
        self._out_channels = image_channels
        self._spatial_dims = spatial_dims

        # Disable built-in centering; we apply differentiable centering ourselves
        self.kernel_net = SIRENKernelEstimator(
            kernel_size=kernel_size,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            omega_0=omega_0,
            spatial_dims=spatial_dims,
            use_centering=False,
        )

        self.central_layer: DifferentiableCentralLayer | DifferentiableCentralLayer3D
        if spatial_dims == 3:
            self.central_layer = DifferentiableCentralLayer3D()
        else:
            self.central_layer = DifferentiableCentralLayer()

        fourier_depth = config.fourier_input_depth if config else 16 * 4
        num_skip = [skip_num_channels] * len(skip_channels)

        if spatial_dims == 3:
            self.skip_net = SkipNetwork3D(
                input_depth=fourier_depth,
                output_channels=image_channels,
                num_channels_down=skip_channels,
                num_channels_up=skip_channels,
                num_channels_skip=num_skip,
            )
        else:
            self.skip_net = SkipNetwork(
                input_depth=fourier_depth,
                output_channels=image_channels,
                num_channels_down=skip_channels,
                num_channels_up=skip_channels,
                num_channels_skip=num_skip,
            )

    def forward(
        self,
        image_input: Tensor,
        kernel_coords: Tensor | None = None,
        kernel_size: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass producing both reconstructed image and estimated kernel.

        :param image_input: Fourier feature input for image network
        :param kernel_coords: Optional kernel coordinate grid
        :param kernel_size: Optional kernel size override
        :returns: Tuple of (reconstructed_image, estimated_kernel)
        """
        kernel = self.kernel_net(kernel_coords, kernel_size)
        kernel = self.central_layer(kernel)
        image = self.skip_net(image_input)
        return image, kernel

    def get_parameters(self) -> list[nn.Parameter]:
        return list(self.parameters())

    def get_kernel_parameters(self) -> list[nn.Parameter]:
        """Get only kernel network parameters."""
        return list(self.kernel_net.parameters())

    def get_image_parameters(self) -> list[nn.Parameter]:
        """Get only image network parameters."""
        return list(self.skip_net.parameters())

    @property
    def spatial_dims(self) -> SpatialDims:
        return self._spatial_dims

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels
