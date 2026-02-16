# deblur-inr-ssres

Blind image deblurring via implicit neural representations (Deblur-INR). Part of the [Self-Supervised Restoration Studio](https://github.com/Self-Supervised-Restoration-Studio) ecosystem.

Supports both 2D and 3D volumetric data. Uses test-time optimization with SIREN kernel estimation and skip network image reconstruction.

Works standalone or as an [ssres](https://github.com/Self-Supervised-Restoration-Studio/ssres) plugin (discovered automatically via entry points).

## Install

```bash
uv add deblur-inr-ssres
```

For development (editable, with local ssres_toolbox):

```bash
git clone https://github.com/Self-Supervised-Restoration-Studio/deblur_inr_ssres.git
cd deblur_inr_ssres
uv sync --extra dev
```

## What's included

| Module | Key exports | Description |
|--------|------------|-------------|
| `model` | `DeblurINRModel` | SIREN kernel + skip network image reconstruction |
| `optimizer` | `DeblurINROptimizer`, `deblur_image` | Test-time optimization with multi-scale coarse-to-fine |
| `config` | `DeblurINRConfig`, `OptimizationStage` | Configuration with per-stage learning rates and scales |

## Usage

```python
from deblur_inr_ssres import deblur_image

deblurred, kernel = deblur_image(blurred_tensor, kernel_size=51, num_iterations=5000)
```

## Dependencies

- [ssres_toolbox](https://github.com/Self-Supervised-Restoration-Studio/ssres_toolbox) — SIREN, SkipNetwork, FourierFeatureEncoding
- PyTorch >= 2.7.0
- Pydantic >= 2.0.0
- tqdm >= 4.66.2

## License

[MIT](LICENSE)
