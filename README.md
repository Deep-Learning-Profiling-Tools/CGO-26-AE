# CGO-26-AE

# Install Environment

## Init Git Submodules

```bash
git submodule init
git submodule update
```

## Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install NV Env if you are on GH200

```bash
UV_SKIP_WHEEL_FILENAME_CHECK=1
uv sync --extra cuda
. .venv/bin/activate
```

## Install AMD Env if you are on MI300X

```bash
UV_SKIP_WHEEL_FILENAME_CHECK=1
uv sync --extra rocm
. .venv/bin/activate
```

# Figure 13 and Figure 14


1. `cd fig13_14`
2. Start experiments by `python run_experiment_nv.py` if on NVIDIA or `python run_experiment_amd.py` if on AMD. The results will be both displayed in terminal and stored in a json file.
3. Plot Figure 13 by `python fig13.py`. The script will automatically read json file and plot AMD part of the figure.
4. Plot Figure 14 by `python fig14.py`.

# Figure 15

1. `cd fig15`
2. Start experiments by `bash run_exp_and_visualize.sh`
3. Plot Figure 15 by `python fig15.py`
