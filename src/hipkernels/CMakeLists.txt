# CMakeLists.txt for HIP kernels

# Find HIP and Torch
find_package(hip REQUIRED)
find_package(Torch REQUIRED)

add_library(hipkernels STATIC
    w4a16_gemm_packed.cu
    w4a16_gemv_packed.cu
    w4a16_gemm_unpacked.cu
    w4a16_gemv_unpacked.cu
    embedding.cu
    lm_head.cu

    rope.cu
    rmsnorm.cu
    w4a16_gemv_packedGPU.cu
    w4a16_gemm_packedGPU.cu
    flash_attn_decode.cu
    softmax.cu
)

# Enable Position Independent Code (required for linking into shared libs)
set_target_properties(hipkernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Set language to HIP
set_source_files_properties(w4a16_gemm_packed.cu w4a16_gemv_packed.cu w4a16_gemm_unpacked.cu w4a16_gemv_unpacked.cu rope.cu rmsnorm.cu w4a16_gemv_packedGPU.cu w4a16_gemm_packedGPU.cu flash_attn_decode.cu embedding.cu lm_head.cu softmax.cu PROPERTIES LANGUAGE HIP)

# Include directories (including Torch) - PUBLIC for headers, PRIVATE for internal
target_include_directories(hipkernels PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_include_directories(hipkernels PUBLIC
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hip
    ${TORCH_INCLUDE_DIRS}
)

# Link HIP runtime and Torch - PRIVATE so they don't propagate compile options
target_link_libraries(hipkernels PRIVATE
    hip::device
    "${TORCH_LIBRARIES}"
)

# Public link just amdhip64 without the device target that brings compile options
target_link_libraries(hipkernels PUBLIC
    amdhip64
)

# Compile definitions - keep __HIP_PLATFORM_AMD__ private
target_compile_definitions(hipkernels PRIVATE
    __HIP_PLATFORM_AMD__=1
    RDNA3=1
)

# Use the first HIP architecture for offload flags.
set(HIP_OFFLOAD_ARCH "${CMAKE_HIP_ARCHITECTURES}")
if(HIP_OFFLOAD_ARCH MATCHES ";")
    list(GET CMAKE_HIP_ARCHITECTURES 0 HIP_OFFLOAD_ARCH)
endif()

# HIP compile options - only for HIP language
target_compile_options(hipkernels PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:
        --offload-arch=${HIP_OFFLOAD_ARCH}
        -O3
        -ffast-math
    >
)


# Test Executable
