set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${ROCM_PATH}/include")
# Test Layout (NPU) executable
add_executable(test_layout_npu
    test_layout_npu.cpp
)

target_include_directories(test_layout_npu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Add PyTorch include directories
if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(test_layout_npu PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(test_layout_npu PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

target_link_libraries(test_layout_npu PRIVATE 
    unified_llm_w4a16_hetero_lib
    "${TORCH_PYTHON_PATH}/lib/libtorch.so"
    "${TORCH_PYTHON_PATH}/lib/libtorch_cpu.so"
    "${TORCH_PYTHON_PATH}/lib/libc10.so"
    npu_matmul_func
    ${HSA_RUNTIME_LIB}
    amdhip64
)

target_compile_definitions(test_layout_npu PRIVATE 
    __HIP_PLATFORM_AMD__=1
)

# Find OpenMP (usually available by default in most environments)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(test_layout_npu PRIVATE OpenMP::OpenMP_CXX)
endif()

# Test Layout (HIP) executable
add_executable(test_layout_hip
    test_layout_hip.cpp
)

target_include_directories(test_layout_hip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Add PyTorch include directories
if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(test_layout_hip PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(test_layout_hip PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

target_link_libraries(test_layout_hip PRIVATE 
    unified_llm_w4a16_hetero_lib
    hipkernels
    "${TORCH_PYTHON_PATH}/lib/libtorch.so"
    "${TORCH_PYTHON_PATH}/lib/libtorch_cpu.so"
    "${TORCH_PYTHON_PATH}/lib/libc10.so"
    amdhip64
    hipblas
)

set_source_files_properties(test_layout_hip.cpp PROPERTIES LANGUAGE HIP)
target_compile_definitions(test_layout_hip PRIVATE __HIP_PLATFORM_AMD__=1)

set_target_properties(test_layout_hip PROPERTIES CXX_STANDARD 17)

# Test Layout (CPU) executable
add_executable(test_layout_cpu
    test_layout_cpu.cpp
)

target_include_directories(test_layout_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Add PyTorch include directories
if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(test_layout_cpu PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(test_layout_cpu PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

target_link_libraries(test_layout_cpu PRIVATE 
    unified_llm_w4a16_hetero_lib
    "${TORCH_PYTHON_PATH}/lib/libtorch.so"
    "${TORCH_PYTHON_PATH}/lib/libtorch_cpu.so"
    "${TORCH_PYTHON_PATH}/lib/libc10.so"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_layout_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()

set_target_properties(test_layout_cpu PROPERTIES CXX_STANDARD 17)

# Hetero GEMV Benchmark (CPU/GPU Sweep)
add_executable(hetero_gemv_cpu_gpu
    hetero_gemv_cpu_gpu.cpp
)

target_include_directories(hetero_gemv_cpu_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(hetero_gemv_cpu_gpu PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(hetero_gemv_cpu_gpu PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

target_link_libraries(hetero_gemv_cpu_gpu PRIVATE 
    unified_llm_w4a16_hetero_lib
    "${TORCH_PYTHON_PATH}/lib/libtorch.so"
    "${TORCH_PYTHON_PATH}/lib/libtorch_cpu.so"
    "${TORCH_PYTHON_PATH}/lib/libc10.so"
    npu_matmul_func
    ${HSA_RUNTIME_LIB}
    amdhip64
)

target_compile_definitions(hetero_gemv_cpu_gpu PRIVATE 
    __HIP_PLATFORM_AMD__=1
)


if(OpenMP_CXX_FOUND)
    target_link_libraries(hetero_gemv_cpu_gpu PRIVATE OpenMP::OpenMP_CXX)
endif()

# Tensor Parallel GEMM NPU/GPU M-Split Sweep
add_executable(tensor_parallel_gemm_npu_gpu_sweepM
    tensor_parallel_gemm_npu_gpu_sweepM.cpp
)

target_include_directories(tensor_parallel_gemm_npu_gpu_sweepM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(tensor_parallel_gemm_npu_gpu_sweepM PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(tensor_parallel_gemm_npu_gpu_sweepM PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

target_link_libraries(tensor_parallel_gemm_npu_gpu_sweepM PRIVATE
    unified_llm_w4a16_hetero_lib
    hipkernels
    "${TORCH_PYTHON_PATH}/lib/libtorch.so"
    "${TORCH_PYTHON_PATH}/lib/libtorch_cpu.so"
    "${TORCH_PYTHON_PATH}/lib/libc10.so"
    npu_matmul_func
    ${HSA_RUNTIME_LIB}
    amdhip64
    hipblas
)

target_compile_definitions(tensor_parallel_gemm_npu_gpu_sweepM PRIVATE
    __HIP_PLATFORM_AMD__=1
)

set_target_properties(tensor_parallel_gemm_npu_gpu_sweepM PROPERTIES CXX_STANDARD 17)

if(OpenMP_CXX_FOUND)
    target_link_libraries(tensor_parallel_gemm_npu_gpu_sweepM PRIVATE OpenMP::OpenMP_CXX)
endif()

# Torch BF16 GEMM sweep
add_executable(tensor_parallel_gemm_torch_bf16_sweepM
    tensor_parallel_gemm_torch_bf16_sweepM.cpp
)

target_include_directories(tensor_parallel_gemm_torch_bf16_sweepM PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(tensor_parallel_gemm_torch_bf16_sweepM PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(tensor_parallel_gemm_torch_bf16_sweepM PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

target_link_libraries(tensor_parallel_gemm_torch_bf16_sweepM PRIVATE
    torch
    ${TORCH_LIBRARIES}
)

set_target_properties(tensor_parallel_gemm_torch_bf16_sweepM PROPERTIES CXX_STANDARD 17)

foreach(SWEEP_DIM K N)
    set(SWEEP_TARGET tensor_parallel_gemm_npu_gpu_sweep${SWEEP_DIM})

    add_executable(${SWEEP_TARGET}
        tensor_parallel_gemm_npu_gpu_sweep${SWEEP_DIM}.cpp
    )

    target_include_directories(${SWEEP_TARGET} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    )

    if(TORCH_PYTHON_PATH)
        if(EXISTS "${TORCH_PYTHON_PATH}/include")
            target_include_directories(${SWEEP_TARGET} PRIVATE "${TORCH_PYTHON_PATH}/include")
        endif()
        if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
            target_include_directories(${SWEEP_TARGET} PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        endif()
    endif()

    target_link_libraries(${SWEEP_TARGET} PRIVATE
        unified_llm_w4a16_hetero_lib
        hipkernels
        "${TORCH_PYTHON_PATH}/lib/libtorch.so"
        "${TORCH_PYTHON_PATH}/lib/libtorch_cpu.so"
        "${TORCH_PYTHON_PATH}/lib/libc10.so"
        npu_matmul_func
        ${HSA_RUNTIME_LIB}
        amdhip64
        hipblas
    )

    target_compile_definitions(${SWEEP_TARGET} PRIVATE
        __HIP_PLATFORM_AMD__=1
    )

    set_target_properties(${SWEEP_TARGET} PROPERTIES CXX_STANDARD 17)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(${SWEEP_TARGET} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endforeach()

# Test Layout (NPU GEMV) executable
add_executable(test_layout_npu_gemv
    test_layout_npu_gemv.cpp
)

target_include_directories(test_layout_npu_gemv PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(test_layout_npu_gemv PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(test_layout_npu_gemv PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

target_link_libraries(test_layout_npu_gemv PRIVATE 
    unified_llm_w4a16_hetero_lib
    "${TORCH_PYTHON_PATH}/lib/libtorch.so"
    "${TORCH_PYTHON_PATH}/lib/libtorch_cpu.so"
    "${TORCH_PYTHON_PATH}/lib/libc10.so"
    npu_matmul_func
    ${HSA_RUNTIME_LIB}
    amdhip64
)

target_compile_definitions(test_layout_npu_gemv PRIVATE 
    __HIP_PLATFORM_AMD__=1
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_layout_npu_gemv PRIVATE OpenMP::OpenMP_CXX)
endif()


# Test Layout (GPU Q4_K) executable
add_executable(test_layoutGPU_hip
    test_layoutGPU_hip.cpp
)

target_include_directories(test_layoutGPU_hip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(test_layoutGPU_hip PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(test_layoutGPU_hip PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

target_link_libraries(test_layoutGPU_hip PRIVATE 
    unified_llm_w4a16_hetero_lib
    hipkernels
    "${TORCH_PYTHON_PATH}/lib/libtorch.so"
    "${TORCH_PYTHON_PATH}/lib/libtorch_cpu.so"
    "${TORCH_PYTHON_PATH}/lib/libc10.so"
    amdhip64
    hipblas
)

target_compile_definitions(test_layoutGPU_hip PRIVATE 
    __HIP_PLATFORM_AMD__=1
)

set_target_properties(test_layoutGPU_hip PROPERTIES CXX_STANDARD 17)

# Flash Attention Decode microbenchmark
add_executable(test_flash_attn_decode_hip
    test_flash_attn_decode_hip.cpp
)

target_include_directories(test_flash_attn_decode_hip PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(test_flash_attn_decode_hip PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(test_flash_attn_decode_hip PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

target_link_libraries(test_flash_attn_decode_hip PRIVATE 
    unified_llm_w4a16_hetero_lib
    hipkernels
    "${TORCH_PYTHON_PATH}/lib/libtorch.so"
    "${TORCH_PYTHON_PATH}/lib/libtorch_cpu.so"
    "${TORCH_PYTHON_PATH}/lib/libc10.so"
    amdhip64
    hipblas
)

target_compile_definitions(test_flash_attn_decode_hip PRIVATE 
    __HIP_PLATFORM_AMD__=1
)

set_target_properties(test_flash_attn_decode_hip PROPERTIES CXX_STANDARD 17)

# Test LM Head (HIP) executable
add_executable(test_layout_hip_lm_head
    test_layout_hip_lm_head.cpp
)

target_include_directories(test_layout_hip_lm_head PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Add PyTorch include directories
if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(test_layout_hip_lm_head PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(test_layout_hip_lm_head PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

target_link_libraries(test_layout_hip_lm_head PRIVATE 
    unified_llm_w4a16_hetero_lib
    hipkernels
    "${TORCH_PYTHON_PATH}/lib/libtorch.so"
    "${TORCH_PYTHON_PATH}/lib/libtorch_cpu.so"
    "${TORCH_PYTHON_PATH}/lib/libc10.so"
    amdhip64
    hipblas
)

target_compile_definitions(test_layout_hip_lm_head PRIVATE __HIP_PLATFORM_AMD__=1)
set_target_properties(test_layout_hip_lm_head PROPERTIES CXX_STANDARD 17)
