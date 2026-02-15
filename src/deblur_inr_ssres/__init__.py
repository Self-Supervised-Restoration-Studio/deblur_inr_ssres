"""Deblur-INR — blind deblurring via implicit neural representations.

Works standalone or as an ssres plugin (discovered automatically via entry points).
"""

from .config import DeblurINRConfig, OptimizationStage
from .model import DeblurINRModel
from .optimizer import DeblurINROptimizer, deblur_image

__all__ = [
    "DeblurINRConfig",
    "OptimizationStage",
    "DeblurINRModel",
    "DeblurINROptimizer",
    "deblur_image",
    "register",
]


def register():
    """Register Deblur-INR components as an ssres plugin.

    Called automatically by ssres plugin discovery via entry points.
    Returns None if ssres is not installed (standalone usage).
    """
    try:
        from ssres.core.plugin_discovery import PluginContribution, PluginManifest
    except ImportError:
        return None

    return PluginManifest(
        name="deblur_inr",
        contributions=[
            PluginContribution(
                kind="model",
                name="deblur_inr_model",
                factory=lambda: DeblurINRModel,
            ),
            PluginContribution(
                kind="optimizer",
                name="deblur_inr_optimizer",
                factory=lambda: DeblurINROptimizer,
            ),
        ],
    )
