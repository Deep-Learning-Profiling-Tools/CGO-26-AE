# Instructions to reproduce Fig.15

## Experiments included

We run these kerenls with and without proton intra-kernel profiling to measure proton's intra-kernel profiling overheads

Kernels are selected based on:

1. They are working well on both NVIDIA H200 and AMD MI300X  
2. They represent a major workload in LLM training or inference.  
3. They have a unique nature of workload characteristics



| Name | Description | Link | Comments |
| :---- | :---- | :---- | :---- |
| matrix-multiplication | Naive matmul | triton/python/tutorials/03-matrix-multiplication.py |  |
| fused-softmax | loss function | triton/python/tutorials/02-fused-softmax.py |  |
| low-memory-dropout | mem-efficient dropout | triton/python/tutorials/04-low-memory-dropout.py |  |
| layer-norm | LayerNorm in Transformers | triton/python/tutorials/05-layer-norm.py |  |
| fused-attention | Flash attention v2 | triton/python/tutorials/06-fused-attention.py |  |
| grouped-gemm | Batched GEMM | triton/python/tutorials/08-grouped-gemm.py | |
| persistant-matmul | small batch training/inference | triton/python/tutorials/09-persistent-matmul.py |  |
| swiglu  | activation function | triton/python/triton\_kernels/triton\_kernels/swiglu\_details/\_swiglu.py |  |
| topk | TopK in MoE | triton/python/triton\_kernels/triton\_kernels/topk\_details/\_topk\_forward.py |  |
