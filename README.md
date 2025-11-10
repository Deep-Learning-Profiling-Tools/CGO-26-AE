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

# Figure 13 and Figure 14

## NVIDIA

1. Activate virtual environment by `source uv_env/nv/.venv/bin/activate`
2. `cd fig13_14`
3. Start experiments by `python run_experiment_nv.py`. The results will be both displayed in terminal and stored in a json file.
4. Plot Figure 13 by `python fig13.py`. The script will automatically read json file and plot NVIDIA part of the figure.
5. Plot Figure 14 by `python fig14.py`.

## AMD
1. Activate virtual environment by `source uv_env/amd/.venv/bin/activate`
2. `cd fig13_14`
3. Start experiments by `python run_experiment_amd.py`. The results will be both displayed in terminal and stored in a json file.
4. Plot Figure 13 by `python fig13.py`. The script will automatically read json file and plot AMD part of the figure.
5. Plot Figure 14 by `python fig14.py`.

# Figure 15

TODO...
