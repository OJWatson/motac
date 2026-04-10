# motac

JAX/NumPyro package for discrete-time mobility-constrained Hawkes-style count models.

## Python version

This project targets **Python 3.11**.

## Install

### Standard (CPU) setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -e .[dev]
```

### GPU setup (CUDA)

For GPU acceleration with JAX, use the following workflow:

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install project dependencies
pip install -e .[dev]

# Install JAX with CUDA support
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# If extras don't work, install explicitly:
pip install --upgrade jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --upgrade jax-cuda12-pjrt jax-cuda12-plugin
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12

# Set CUDA library path (add to venv activate script)
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH' >> .venv/bin/activate

# Verify GPU detection
python -c "import jax; print(jax.devices())"  # Should show [CudaDevice(id=0)]
```

**Note:** Adjust the CUDA path (`/usr/local/cuda-12.4`) to match your installation. Use `nvcc --version` to check your CUDA version.

### Development setup with uv (recommended)

```bash
# Create environment
uv venv --python 3.11
source .venv/bin/activate

# Install project
uv pip install -e .[dev]

# For GPU support:
uv pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip install nvidia-cudnn-cu12 nvidia-cublas-cu12

# Set CUDA library path
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/targets/x86_64-linux/lib:$LD_LIBRARY_PATH' >> .venv/bin/activate
```

## Run tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

## Build docs (local artifacts)

Notebooks/docs are local build artifacts and are not intended as notebook CI gates.

```bash
.venv/bin/pip install -e .[docs]
.venv/bin/python -m sphinx -b html docs docs/_build/html
```

## Backend benchmarks

```bash
python3.11 scripts/run_bench.py
# or, after install:
motac-bench
```

## Layout

- `motac/`: package source
- `tests/`: unit + integration smoke tests
- `docs/`: Sphinx + MyST + nbsphinx documentation
