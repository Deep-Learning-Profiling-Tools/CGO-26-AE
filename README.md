# CGO-26-AE

# Install Environment

## Init Git Submodules

## Install uv

## Install NV Env

cd uv_env/nv
uv sync
source .venv/bin/activate

## Install AMD Env

cd uv_env/amd
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv sync
source .venv/bin/activate
