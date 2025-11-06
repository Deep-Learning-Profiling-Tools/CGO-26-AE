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

    # run kernels without profiling. Timing are reported by cudaEvent
    python kernels/fused_softmax.py 
    python kernels/grouped_gemm_original.py
    python kernels/layer_norm.py 
    python kernels/low_memory_dropout.py 
    python kernels/swiglu.py 
    python kernels/topk_original.py 
    python kernels/persistent_matmul.py 
    python kernels/fused_attention.py --simple

    # TODO (Tianle): extract results from profiles and cudaEvent reports


    # TODO (Hao): visualize results and produce the figure

)