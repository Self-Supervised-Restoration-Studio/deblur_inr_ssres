"""Test-time optimization for blind image deblurring."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import torch
import torch.nn.functional as F
from ssrs_toolbox import FourierFeatureEncoding
from ssrs_toolbox.losses import SmoothnessLoss
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from .config import DeblurINRConfig, OptimizationStage
from .losses import _DeblurFFTLoss, _DeblurSSIMLoss, _DeblurTVLoss
from .model import DeblurINRModel


def _build_pixel_skip_pyramid(image: Tensor, spatial_dims: int = 2) -> dict[float, Tensor]:
    """Build multi-scale pyramid using pixel-skipping (NOT interpolation).

    This matches the reference Deblur-INR implementation which uses strided
    pixel selection rather than bilinear downsampling.

    :param image: Input image (B, C, H, W) for 2D or (B, C, D, H, W) for 3D
    :param spatial_dims: Number of spatial dimensions (2 or 3)
    :returns: Pyramid {1.0: original, 0.5: half, 0.25: quarter}
    """
    pyramid = {1.0: image}
    current = image
    skip = (slice(None, None, 2),) * spatial_dims
    for scale in [0.5, 0.25]:
        current = current[(..., *skip)]
        pyramid[scale] = current
    return pyramid


def _get_scaled_kernel_size(base_size: int, scale: float) -> int:
    """Get kernel size for a given scale, ensuring odd size."""
    k = max(3, math.ceil(base_size * scale))
    return k | 1  # Ensure odd


class DeblurINROptimizer:
    """Test-time optimization for blind image deblurring.

    Handles the complete optimization loop for deblurring a single image,
    including multi-scale pyramid optimization with per-stage learning rates
    and parameter freezing. Supports both 2D and 3D inputs via config.spatial_dims.
    """

    def __init__(
        self,
        config: DeblurINRConfig | None = None,
        device: str = "cuda",
    ):
        self.config = config or DeblurINRConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.spatial_dims = self.config.spatial_dims

        self.ssim_loss = _DeblurSSIMLoss().to(self.device)
        self.fft_loss = _DeblurFFTLoss(normalize=True).to(self.device)
        self.tv_loss = _DeblurTVLoss().to(self.device)
        self.smoothness_loss = SmoothnessLoss(mode="l2").to(self.device)

    def _create_fourier_input(self, *spatial_sizes: int) -> Tensor:
        """Create Fourier feature input for image network.

        Uses [0, 1] coordinate range and configurable frequency base
        to match the reference Deblur-INR implementation.

        :param spatial_sizes: Spatial dimensions (H, W) for 2D or (D, H, W) for 3D
        :returns: Fourier feature tensor (1, C, *spatial_sizes)
        """
        # Reference uses [0, 1] range (NOT [-1, 1])
        linspaces = [torch.linspace(0, 1, s) for s in spatial_sizes]
        grids = torch.meshgrid(*linspaces, indexing="ij")
        coords = torch.stack(grids, dim=-1)  # (*spatial_sizes, ndim)

        encoding = FourierFeatureEncoding(
            in_features=len(spatial_sizes),
            num_frequencies=self.config.num_frequencies,
            include_input=False,
            base=self.config.fourier_base,
        )

        coords_flat = coords.view(-1, len(spatial_sizes))
        features = encoding(coords_flat)
        features = features.view(*spatial_sizes, -1)

        # Move channel dim to front: (*spatial_sizes, C) -> (C, *spatial_sizes)
        perm = [len(spatial_sizes), *range(len(spatial_sizes))]
        features = features.permute(*perm)

        return features.unsqueeze(0).to(self.device)

    def _get_stage(self, iteration: int) -> OptimizationStage:
        """Get the optimization stage for a given iteration."""
        for stage in self.config.stages:
            if stage.start_iter <= iteration < stage.end_iter:
                return stage
        return self.config.stages[-1]

    def _create_optimizer_and_scheduler(
        self,
        model: DeblurINRModel,
        stage: OptimizationStage,
    ) -> tuple[Adam, MultiStepLR]:
        """Create optimizer with per-stage learning rates.

        :param model: The deblur model
        :param stage: Current optimization stage
        :returns: (optimizer, scheduler) tuple
        """
        param_groups = [
            {"params": model.skip_net.parameters(), "lr": stage.image_lr},
            {"params": model.kernel_net.parameters(), "lr": stage.kernel_lr},
        ]

        optimizer = Adam(param_groups)

        # Adjust milestones relative to stage start
        milestones = [m for m in self.config.lr_milestones if m > stage.start_iter]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=self.config.lr_gamma)

        return optimizer, scheduler

    def _set_requires_grad(self, model: DeblurINRModel, stage: OptimizationStage):
        """Freeze/unfreeze parameters based on stage config."""
        for p in model.skip_net.parameters():
            p.requires_grad = stage.optimize_image
        for p in model.kernel_net.parameters():
            p.requires_grad = stage.optimize_kernel

    def deblur(
        self,
        blurred_image: Tensor,
        callbacks: list[Callable] | None = None,
        show_progress: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Deblur a single image using test-time optimization.

        :param blurred_image: Blurred input image
            2D: (C, H, W) or (B, C, H, W)
            3D: (C, D, H, W) or (B, C, D, H, W)
        :param callbacks: Optional list of callback functions
        :param show_progress: Whether to show progress bar
        :returns: Tuple of (deblurred_image, estimated_kernel)
        """
        expected_ndim = self.spatial_dims + 2  # B, C, + spatial
        if blurred_image.dim() == expected_ndim - 1:
            blurred_image = blurred_image.unsqueeze(0)

        blurred_image = blurred_image.to(self.device)
        spatial_sizes = blurred_image.shape[2:]  # (H, W) or (D, H, W)

        # Build image pyramid using pixel-skipping
        pyramid = _build_pixel_skip_pyramid(blurred_image, self.spatial_dims)

        # Create model
        model = DeblurINRModel(config=self.config).to(self.device)

        # Prepare Fourier feature inputs for different scales
        inputs: dict[float, Tensor] = {}
        kernel_sizes: dict[float, int] = {}
        for scale in pyramid:
            scaled_sizes = pyramid[scale].shape[2:]
            k_size = _get_scaled_kernel_size(self.config.kernel_size, scale)
            kernel_sizes[scale] = k_size
            pad = k_size - 1
            padded_sizes = tuple(s + pad for s in scaled_sizes)
            inputs[scale] = self._create_fourier_input(*padded_sizes)

        # Select convolution function
        conv_fn = F.conv3d if self.spatial_dims == 3 else F.conv2d
        interp_mode = "trilinear" if self.spatial_dims == 3 else "bilinear"

        # Track current stage for optimizer recreation
        current_stage = self._get_stage(0)
        self._set_requires_grad(model, current_stage)
        optimizer, scheduler = self._create_optimizer_and_scheduler(model, current_stage)

        # Optimization loop
        iterator = range(self.config.num_iterations)
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Deblurring")
            except ImportError:
                pass

        for iteration in iterator:
            stage = self._get_stage(iteration)

            # Recreate optimizer when stage changes
            if stage is not current_stage:
                current_stage = stage
                self._set_requires_grad(model, stage)
                optimizer, scheduler = self._create_optimizer_and_scheduler(model, stage)

            scale = stage.scale
            k_size = kernel_sizes[scale]

            optimizer.zero_grad()

            image_input = inputs[scale]
            recon_image, kernel = model(image_input, kernel_size=k_size)

            # Use grouped convolution to apply the same kernel to each channel
            C = recon_image.shape[1]
            kernel_for_conv = kernel.repeat(C, 1, *([1] * self.spatial_dims))
            conv_result = conv_fn(recon_image, kernel_for_conv, padding=0, groups=C)

            target = pyramid[scale]

            if conv_result.shape[2:] != target.shape[2:]:
                conv_result = F.interpolate(
                    conv_result,
                    size=target.shape[2:],
                    mode=interp_mode,
                    align_corners=False,
                )

            loss = self.config.ssim_weight * self.ssim_loss(conv_result, target)

            if self.config.fft_weight > 0:
                loss = loss + self.config.fft_weight * self.fft_loss(conv_result, target)

            if self.config.tv_weight > 0:
                loss = loss + self.config.tv_weight * self.tv_loss(recon_image, None)

            if self.config.smoothness_weight > 0:
                loss = loss + self.config.smoothness_weight * self.smoothness_loss(kernel)

            loss.backward()

            if self.config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)

            optimizer.step()
            scheduler.step()

            if callbacks:
                for callback in callbacks:
                    callback(iteration, loss.item(), model, recon_image, kernel)

        # Final output at full resolution
        with torch.no_grad():
            final_input = inputs[1.0]
            final_image, final_kernel = model(final_input, kernel_size=self.config.kernel_size)

            pad = (self.config.kernel_size - 1) // 2
            slices: list[slice] = [slice(None), slice(None)]  # B, C
            for s in spatial_sizes:
                slices.append(slice(pad, pad + s))
            final_image = final_image[tuple(slices)]

        return final_image, final_kernel


def deblur_image(
    blurred_image: Tensor,
    kernel_size: int = 51,
    num_iterations: int = 5000,
    device: str = "cuda",
    show_progress: bool = True,
    spatial_dims: Literal[2, 3] = 2,
) -> tuple[Tensor, Tensor]:
    """Convenience function for blind image deblurring.

    :param blurred_image: Blurred input image
    :param kernel_size: Size of blur kernel to estimate
    :param num_iterations: Number of optimization iterations
    :param device: Device to run on
    :param show_progress: Whether to show progress bar
    :param spatial_dims: Number of spatial dimensions (2 or 3)
    :returns: Tuple of (deblurred_image, estimated_kernel)
    """
    config = DeblurINRConfig(
        kernel_size=kernel_size,
        num_iterations=num_iterations,
        spatial_dims=spatial_dims,
    )

    opt = DeblurINROptimizer(config, device=device)
    return opt.deblur(blurred_image, show_progress=show_progress)
