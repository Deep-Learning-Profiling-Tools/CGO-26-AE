#!/bin/bash

# It only works for rocm 6.4

uv pip install -e triton_kernels/

(
    set -x
    cd triton_kernels/bench

    export NCCL_DEBUG=WARN
    torchrun --standalone --nproc-per-node=8 ./bench_mlp.py --tp 1 --ep 8 --name gpt-oss-x2 
    torchrun --standalone --nproc-per-node=8 ./bench_mlp.py --tp 2 --ep 4 --name gpt-oss-x2 
    torchrun --standalone --nproc-per-node=8 ./bench_mlp.py --tp 4 --ep 2 --name gpt-oss-x2 
    torchrun --standalone --nproc-per-node=8 ./bench_mlp.py --tp 8 --ep 1 --name gpt-oss-x2 

    python plot.py
    cd -
)