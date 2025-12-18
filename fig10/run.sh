
uv pip install -e triton_kernels/

(
    set -x
    cd triton_kernels/bench
    
    export NCCL_DEBUG=WARN
    python bench_mlp.py
    cd -
)