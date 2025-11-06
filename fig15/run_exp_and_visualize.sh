# run intra-kernel profiling
DUMP_DIR="$(pwd)/ttgir_dump"
KERNELS_DIR="$(pwd)/start_end"
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python start_end/fused_softmax.py --profile --buffer-size 1024
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python start_end/grouped_gemm_original.py --profile --buffer-size 1024
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python start_end/layer_norm.py --profile --buffer-size 1024
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python start_end/low_memory_dropout.py --profile --buffer-size 512
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python start_end/matmul.py --profile --buffer-size 1024
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python start_end/swiglu.py --profile --buffer-size 1024
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python start_end/topk_original.py --profile --buffer-size 2048
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python start_end/persistent_matmul.py --profile --buffer-size 512
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$DUMP_DIR" python start_end/fused_attention.py --profile --buffer-size 1024

# run kernels without profiling. Timing are reported by cudaEvent
python start_end/fused_softmax.py 
python start_end/grouped_gemm_original.py
python start_end/layer_norm.py 
python start_end/low_memory_dropout.py 
python start_end/swiglu.py 
python start_end/topk_original.py 
python start_end/persistent_matmul.py 
python start_end/fused_attention.py 

# TODO (Tianle): extract results from profiles and cudaEvent reports


# TODO (Hao): visualize results and produce the figure