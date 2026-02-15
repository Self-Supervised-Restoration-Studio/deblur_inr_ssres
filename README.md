# deblur-inr-ssrs

Blind image deblurring via implicit neural representations (Deblur-INR). Part of the [Self-Supervised Restoration Studio](https://github.com/Self-Supervised-Restoration-Studio) ecosystem.

Supports both 2D and 3D volumetric data. Uses test-time optimization with SIREN kernel estimation and skip network image reconstruction.

Works standalone or as an [ssrs](https://github.com/Self-Supervised-Restoration-Studio/ssrs) plugin (discovered automatically via entry points).

## Install

```bash
uv add deblur-inr-ssrs
```

For development (editable, with local ssrs_toolbox):

```bash
git clone https://github.com/Self-Supervised-Restoration-Studio/deblur_inr_ssrs.git
cd deblur_inr_ssrs
uv sync --extra dev
```

## Usage

```python
from deblur_inr_ssrs import deblur_image

deblurred, kernel = deblur_image(blurred_tensor, kernel_size=51, num_iterations=5000)
```

## Dependencies

- [ssrs_toolbox](https://github.com/Self-Supervised-Restoration-Studio/ssrs_toolbox) — SIREN, SkipNetwork, FourierFeatureEncoding
- PyTorch >= 2.7.0
- Pydantic >= 2.0.0
- tqdm >= 4.66.2

## License

[MIT](LICENSE)
