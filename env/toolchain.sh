# Source this file. Pins the CUDA 13.4 toolkit (/usr/local/cuda points to 13.2 on this node).
export CUDA_ROOT=/usr/local/cuda-13.4
export CUDA_LIB=$CUDA_ROOT/targets/x86_64-linux/lib
export NVCC=$CUDA_ROOT/bin/nvcc
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # make CUDA indices match nvidia-smi
# sm_86 = RTX 3090, sm_89 = RTX 4090, sm_120 = RTX 5060 Ti
export NVCC_ARCH="-gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120"
export NVCC_FLAGS="-O3 -std=c++17 $NVCC_ARCH -Xlinker -rpath=$CUDA_LIB -L$CUDA_LIB"
# GPUs chosen for the first round: 4 = RTX 3090 (PCIe Gen4 x16), 2 = RTX 5060 Ti (Gen3 x8 slot)
export PATHB_GPUS="4 2"
