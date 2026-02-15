# deblur-inr-ssrs

Blind image deblurring via implicit neural representations (Deblur-INR) for SSRS projects.

Supports both 2D and 3D volumetric data. Uses test-time optimization with SIREN kernel estimation and skip network image reconstruction.

## Installation

```bash
uv sync
```

## Usage

```python
from deblur_inr_ssrs import deblur_image

deblurred, kernel = deblur_image(blurred_tensor, kernel_size=51, num_iterations=5000)
```

## Dependencies

- `torch>=2.7.0`
- `ssrs-toolbox>=0.1.0` (SIREN, SkipNetwork, FourierFeatureEncoding)
- `pydantic>=2.0.0`
- `tqdm>=4.66.2`
