#!/bin/bash

# It only works for rocm 6.4

cd triton_kernels/bench

torchrun --nproc-per-node=8 ./bench_mlp.py --tp 1 --ep 8 --name gpt-oss-x2
torchrun --nproc-per-node=8 ./bench_mlp.py --tp 2 --ep 4 --name gpt-oss-x2
torchrun --nproc-per-node=8 ./bench_mlp.py --tp 4 --ep 2 --name gpt-oss-x2
torchrun --nproc-per-node=8 ./bench_mlp.py --tp 8 --ep 1 --name gpt-oss-x2