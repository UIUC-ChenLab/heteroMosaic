# Skip on Ryzen AI Max+ 395 (Radeon 8060S).
if ! lscpu | grep -q "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"; then
  export HSA_OVERRIDE_GFX_VERSION=11.0.0
fi

# Optional: Enable experimental memory-efficient attention on AMD GPUs
# This can improve performance and reduce memory usage, but is still experimental
# Uncomment the line below to enable it:
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

# Set repo root directory
export HETEROMOSAIC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "$HETEROMOSAIC_ROOT/utils/rocmPytorch/bin/activate"

# Add local torch lib to LD_LIBRARY_PATH
if [ -d "$HETEROMOSAIC_ROOT/utils/rocmPytorch/lib/python3.12/site-packages/torch/lib" ]; then
    export LD_LIBRARY_PATH="$HETEROMOSAIC_ROOT/utils/rocmPytorch/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH"
fi

if [ -n "${ROCM_PATH:-}" ] && [ -d "$ROCM_PATH" ]; then
    export ROCM_PATH
elif [ -d /opt/rocm ]; then
    export ROCM_PATH=/opt/rocm
elif [ -d /opt/rocm-7.2.0 ]; then
    export ROCM_PATH=/opt/rocm-7.2.0
else
    export ROCM_PATH=/opt/rocm
fi

echo "Using ROCm (${ROCM_PATH})"

export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LD_LIBRARY_PATH
export CPATH=$ROCM_PATH/include:$CPATH
export LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$LIBRARY_PATH
source /opt/xilinx/xrt/setup.sh 
