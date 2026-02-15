"""Configuration for Deblur-INR optimization."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, computed_field

SpatialDims = Literal[2, 3]


class OptimizationStage(BaseModel):
    """Configuration for a single optimization stage.

    :param start_iter: Starting iteration for this stage
    :param end_iter: Ending iteration for this stage
    :param scale: Image scale factor (1.0 = full, 0.5 = half, etc.)
    :param optimize_kernel: Whether to optimize kernel in this stage
    :param optimize_image: Whether to optimize image network in this stage
    :param kernel_lr: Learning rate for kernel network
    :param image_lr: Learning rate for image network
    """

    start_iter: int
    end_iter: int
    scale: float
    optimize_kernel: bool = True
    optimize_image: bool = True
    kernel_lr: float = 5e-5
    image_lr: float = 5e-3


class DeblurINRConfig(BaseModel):
    """Configuration for Deblur-INR optimization.

    Pydantic model with validation, serialization, and preset support.
    """

    spatial_dims: SpatialDims = Field(2, description="Number of spatial dimensions (2 or 3)")
    kernel_size: int = Field(51, ge=3, description="Size of blur kernel to estimate")
    num_iterations: int = Field(5000, ge=1, description="Total optimization iterations")
    image_channels: int = Field(1, ge=1, description="Number of image channels")
    hidden_features: int = Field(64, ge=8, description="Hidden layer size for SIREN")
    hidden_layers: int = Field(3, ge=1, description="Hidden layers in SIREN")
    omega_0: float = Field(30.0, gt=0, description="SIREN frequency factor")
    num_frequencies: int = Field(16, ge=1, description="Fourier feature frequencies")
    fourier_base: float = Field(
        default=2 ** (8 / 7), gt=1.0, description="Base for Fourier frequency bands"
    )
    skip_channels: list[int] = Field(
        default_factory=lambda: [128, 128, 128, 128, 128],
        description="Channel sizes for Skip network levels",
    )
    skip_num_channels: int = Field(16, ge=1, description="Skip connection channels")

    # Loss weights
    ssim_weight: float = Field(1.0, ge=0, description="Weight for SSIM loss")
    fft_weight: float = Field(0.001, ge=0, description="Weight for FFT loss")
    tv_weight: float = Field(0.0, ge=0, description="Weight for TV regularization")
    kernel_reg_weight: float = Field(0.0, ge=0, description="Kernel regularization weight")
    smoothness_weight: float = Field(0.0001, ge=0, description="Kernel smoothness weight")

    # Optimizer settings
    image_lr: float = Field(5e-3, gt=0, description="Image network learning rate")
    kernel_lr: float = Field(5e-5, gt=0, description="Kernel network learning rate")

    # MultiStepLR settings
    lr_milestones: list[int] = Field(
        default_factory=lambda: [2000, 4000], description="LR decay milestones"
    )
    lr_gamma: float = Field(0.5, gt=0, le=1, description="LR decay factor")

    # Gradient clipping
    max_grad_norm: float = Field(1.0, ge=0, description="Gradient clipping max norm (0=disabled)")

    stages: list[OptimizationStage] = Field(
        default_factory=lambda: [
            OptimizationStage(
                start_iter=0,
                end_iter=500,
                scale=0.25,
                optimize_kernel=True,
                optimize_image=False,
                kernel_lr=5e-5,
                image_lr=5e-3,
            ),
            OptimizationStage(
                start_iter=500,
                end_iter=1000,
                scale=0.5,
                optimize_kernel=True,
                optimize_image=False,
                kernel_lr=5e-5,
                image_lr=5e-3,
            ),
            OptimizationStage(
                start_iter=1000,
                end_iter=1500,
                scale=0.5,
                optimize_kernel=True,
                optimize_image=True,
                kernel_lr=5e-5,
                image_lr=5e-3,
            ),
            OptimizationStage(
                start_iter=1500,
                end_iter=5000,
                scale=1.0,
                optimize_kernel=True,
                optimize_image=True,
                kernel_lr=2.5e-5,
                image_lr=2.5e-3,
            ),
        ],
        description="Multi-stage optimization schedule",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fourier_input_depth(self) -> int:
        """Fourier feature input depth.

        2D: num_frequencies * 2 * 2 (e.g. 16 * 4 = 64)
        3D: num_frequencies * 2 * 3 (e.g. 8 * 6 = 48)
        """
        return self.num_frequencies * 2 * self.spatial_dims

    @classmethod
    def create_3d_default(cls) -> DeblurINRConfig:
        """Create a config with 3D-optimal defaults for volumetric data."""
        return cls(
            spatial_dims=3,
            kernel_size=32,
            hidden_features=128,
            num_frequencies=8,
            skip_channels=[64, 64, 64, 64],
            skip_num_channels=8,
            stages=[
                OptimizationStage(
                    start_iter=0,
                    end_iter=500,
                    scale=0.5,
                    optimize_kernel=True,
                    optimize_image=False,
                    kernel_lr=5e-5,
                    image_lr=5e-3,
                ),
                OptimizationStage(
                    start_iter=500,
                    end_iter=1000,
                    scale=0.5,
                    optimize_kernel=True,
                    optimize_image=True,
                    kernel_lr=5e-5,
                    image_lr=5e-3,
                ),
                OptimizationStage(
                    start_iter=1000,
                    end_iter=1500,
                    scale=1.0,
                    optimize_kernel=True,
                    optimize_image=True,
                    kernel_lr=5e-5,
                    image_lr=5e-3,
                ),
                OptimizationStage(
                    start_iter=1500,
                    end_iter=5000,
                    scale=1.0,
                    optimize_kernel=True,
                    optimize_image=True,
                    kernel_lr=2.5e-5,
                    image_lr=2.5e-3,
                ),
            ],
        )
