# Building NumPy with Apple Accelerate BLAS

On Apple Silicon (macOS ARM64), NumPy can benefit from **Apple Accelerate BLAS** (up to 10x speedup for some linear algebra operations) compared to the default OpenBLAS.

When installed via `pip` or `uv`, NumPy uses **OpenBLAS by default**. To use Apple Accelerate, you must rebuild NumPy from source.

## Quick Check: Which BLAS is NumPy using?

```bash
python -c "import numpy as np; np.show_config()"
```

Look for:
- **✓ Using Accelerate**: Config shows `accelerate` or `veclib`
- **✗ Using OpenBLAS**: Config shows `openblas` (suboptimal on Apple Silicon)

## Solution 1: Use Conda (Recommended for Quick Setup)

Conda has pre-built NumPy with Accelerate BLAS on conda-forge.

### Step 1: Create a new environment with conda

```bash
# Install conda if you don't have it (via Miniforge)
# https://github.com/conda-forge/miniforge

# Create an Apple Accelerate-enabled environment
conda create -n dlc-accelerate \
    python=3.10 \
    "numpy::libblas=*=*accelerate" \
    opencv-python-headless \
    onnx \
    # ... other dependencies
```

### Step 2: Activate and verify

```bash
conda activate dlc-accelerate
python -c "import numpy as np; np.show_config()"
# Should show "accelerate" or "veclib"
```

**Limitation**: Conda environment != project's uv-managed venv. You'll need to install other dependencies separately or maintain two parallel envs.

## Solution 2: Build NumPy from Source with Accelerate (Advanced)

This rebuilds NumPy using the Accelerate BLAS library shipped with macOS.

### Prerequisites

```bash
# Ensure Xcode command line tools are installed
xcode-select --install

# Install meson and ninja (required for NumPy 2.x builds)
pip install meson ninja meson-python
```

### Step 1: Deactivate the current venv and build NumPy

```bash
# Deactivate any virtual environment
deactivate

# Clone NumPy source
git clone https://github.com/numpy/numpy.git
cd numpy

# Check out the version matching your current NumPy
# (e.g., v1.26.4)
git checkout v1.26.4

# Build with Accelerate BLAS
# Create a meson setup config file
cat > meson_setup.ini << 'EOF'
[properties]
blas = 'accelerate'
lapack = 'accelerate'
EOF

# Build and install
pip install --no-build-isolation -e . \
    -Csetup-args="-Dblas=accelerate" \
    -Csetup-args="-Dlapack=accelerate"
```

### Step 2: Verify and reinstall to your venv

After NumPy is built with Accelerate:

```bash
# Return to your project
cd /path/to/Deep-Live-Cam

# Reactivate your venv
source .venv/bin/activate

# Reinstall the built NumPy into your venv
pip install --no-deps /path/to/numpy/dist/numpy-*.whl

# Or, if you built it in-place:
# Just re-activate the venv and verify
python -c "import numpy as np; np.show_config()"
```

## Solution 3: Use `uv` with a Custom Build (Experimental)

As of uv 0.5.x, there is no native support for building NumPy from source with Accelerate. If you need to use uv exclusively, you have two options:

1. **Wait for uv #13103** (tracked in [uv GitHub issue #13103](https://github.com/astral-sh/uv/issues/13103)) to add build configuration support
2. **Patch the NumPy wheel** after installation:
   ```bash
   uv pip install --compile numpy  # Forces compile, but doesn't support BLAS config
   ```

## Expected Performance Improvement

| Operation | OpenBLAS | Accelerate | Speedup |
|-----------|----------|-----------|---------|
| `np.dot()` (1000×1000 matrices) | ~15ms | ~2ms | 7.5x |
| `np.linalg.norm()` | ~8ms | ~1ms | 8x |
| Face embedding normalization | N/A | Measured after optimization |

Real-world impact in Deep-Live-Cam: **1-5% FPS improvement** (BLAS ops are not the bottleneck; inference dominates).

## Troubleshooting

### "NumPy is still using OpenBLAS"

1. Verify your build environment used Accelerate:
   ```bash
   pip show numpy
   # Check the Location — it should point to your rebuilt NumPy, not site-packages
   ```

2. Reinstall without caching:
   ```bash
   pip install --no-cache-dir --force-reinstall numpy
   ```

3. Rebuild from scratch:
   ```bash
   pip uninstall numpy -y
   # Rebuild using one of the solutions above
   ```

### "Cannot import numpy" after rebuild

If the build failed partway:

```bash
# Force-install the pre-built wheel again
pip install --force-reinstall numpy==1.26.4
```

## References

- [NumPy BLAS/LAPACK Configuration](https://numpy.org/doc/stable/f2py/buildtools/blas_lapack.html)
- [uv issue #13103 — NumPy Accelerate support](https://github.com/astral-sh/uv/issues/13103)
- [Apple Accelerate Framework](https://developer.apple.com/accelerate/)
- [Performance Analysis: Deep-Live-Cam on M4 Pro](../performance-analysis-m4-pro.md)
