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

# Instructions to Reproduce Experiments

Before proceeding, make sure you have activated python env by `. .venv/bin/activate`.

## Major Results

###  Figure 13 and Figure 14


1. `cd fig13_14`
2. Start experiments by `python run_experiment_nv.py` if on NVIDIA or `python run_experiment_amd.py` if on AMD. The results will be both displayed in terminal and stored in a json file.
3. Plot Figure 13 by `python fig13.py`. The script will automatically read json file and plot AMD part of the figure.
4. Plot Figure 14 by `python fig14.py`.
5. Check generated `fig13.pdf` and `fig14.pdf`

### Figure 15

1. `cd fig15`
2. Start experiments by `bash run_exp_and_visualize.sh`
3. Plot Figure 15 by `python fig15.py`. The generated figure is `fig15.pdf`

## Optional Results

### Figure 8

```bash
cd fig8
bash run.sh
```

The proton profile will be visualized and printed in the terminal outputs.

### Figure 9

```bash
cd fig9
bash run.sh
```

Data and figures would be generated under `fig9/triton_kernels/bench/logs/gpt-oss-x2`

### Figure 10

```bash
cd fig10
bash run.sh
```

> [!NOTE] 
> this experiment does not work with ROCM 7.0 due to ROCM bugs, and it requires the user to install ROCM 6.4. Please try `uv pip install https://download.pytorch.org/whl/rocm6.4/torch-2.9.0%2Brocm6.4-cp313-cp313-manylinux_2_28_x86_64.whl`

Similar to figure 9, data and figures would be generated under `fig10/triton_kernels/bench/logs/gpt-oss-x2`

### Figure 12

```bash
cd fig12
bash run.sh
```

Similar to figure 8, the proton profile will be visualized and printed in the terminal.
