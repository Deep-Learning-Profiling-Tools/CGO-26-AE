# Detect GPU platform and set output CSV filename
if nvidia-smi &> /dev/null; then
    export CUPTI_TIMING_CSV_FILENAME="cupti_profile_timings_nv.csv"
    echo "Detected NVIDIA GPU, output will be saved to cupti_profile_timings_nv.csv"
elif rocm-smi &> /dev/null; then
    export CUPTI_TIMING_CSV_FILENAME="cupti_profile_timings_amd.csv"
    echo "Detected AMD GPU, output will be saved to cupti_profile_timings_amd.csv"
else
    export CUPTI_TIMING_CSV_FILENAME="cupti_profile_timings.csv"
    echo "GPU type unknown, output will be saved to cupti_profile_timings.csv"
fi

# run intra-kernel profiling
(
    # make shell output verbose
    set -x
    DUMP_DIR="$(pwd)/ttgir_dump"
    KERNELS_DIR="$(pwd)/kernels"
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/fused_softmax.py --profile --buffer-size 1024 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/grouped_gemm_original.py --profile --buffer-size 1024 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/layer_norm.py --profile --buffer-size 1024 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/low_memory_dropout.py --profile --buffer-size 512 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/matmul.py --profile --buffer-size 1024 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/swiglu.py --profile --buffer-size 1024 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/topk_original.py --profile --buffer-size 2048 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/persistent_matmul.py --profile --buffer-size 512 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/fused_attention.py --profile --buffer-size 1024 --simple --use-cuda-event

    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/fused_softmax.py --profile --instrument --buffer-size 1024 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/grouped_gemm_original.py --profile --instrument --buffer-size 1024 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/layer_norm.py --profile --instrument --buffer-size 1024 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/low_memory_dropout.py --profile --instrument --buffer-size 512 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/matmul.py --profile --instrument --buffer-size 1024 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/swiglu.py --profile --instrument --buffer-size 1024 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/topk_original.py --profile --instrument --buffer-size 2048 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/persistent_matmul.py --profile --instrument --buffer-size 512 --use-cuda-event
    TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python kernels/fused_attention.py --profile  --instrument --buffer-size 1024 --simple --use-cuda-event
    # TODO (Hao): visualize results and produce the figure

)
