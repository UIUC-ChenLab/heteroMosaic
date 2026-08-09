cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(heteroMosaic)

# Check if the environmental variable is set
if(NOT DEFINED ENV{HOME_LIBS})
    message(FATAL_ERROR "HOME_LIBS environmental variable is not set. Please set it before running CMake.")
endif()
# Set CMake variable based on the environmental variable
set(HOME_LIBS "$ENV{HOME_LIBS}")

# Find Python
find_package(Python COMPONENTS Interpreter REQUIRED)

# Try to find Torch from Python environment first
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/share/cmake/Torch")
    set(CMAKE_PREFIX_PATH "${TORCH_PYTHON_PATH}/share/cmake/Torch" ${CMAKE_PREFIX_PATH})
    message(STATUS "Found PyTorch in Python environment: ${TORCH_PYTHON_PATH}")
else()
    # Fallback to LibTorch path
    # Fallback to LibTorch path
    set(CMAKE_PREFIX_PATH "${HOME_LIBS}/libtorch_7.1.0")
    set(TORCH_PYTHON_PATH "${HOME_LIBS}/libtorch_7.1.0")
endif()

# Set the XDNA driver path
set(DRIVER_PATH "${HOME_LIBS}/xdna-driver-05-19-25")

# Pin the toolchain to the ROCm 7.1.1 installation.
set(ROCM_PATH "/opt/rocm-7.1.1")
set(ENV{ROCM_PATH} "${ROCM_PATH}")
message(STATUS "Using ROCM_PATH: ${ROCM_PATH}")
link_directories(${ROCM_PATH}/lib)

# Find required packages
find_package(Torch REQUIRED)

# Set runtime output directory for all executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# include directories for the project
include_directories(${CMAKE_SOURCE_DIR}/include)

# Enable HIP language and set architectures
set(HETEROMOSAIC_HIP_ARCH_DEFAULT "gfx1100")
execute_process(
    COMMAND lscpu
    OUTPUT_VARIABLE LSCUPOUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(LSCUPOUT MATCHES "Model name:[ \t]*AMD RYZEN AI MAX\\+ 395 w/ Radeon 8060S")
    set(HETEROMOSAIC_HIP_ARCH_DEFAULT "gfx1151")
endif()
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES OR CMAKE_HIP_ARCHITECTURES STREQUAL "")
    set(CMAKE_HIP_ARCHITECTURES "${HETEROMOSAIC_HIP_ARCH_DEFAULT}" CACHE STRING "HIP architectures" FORCE)
endif()
message(STATUS "Using HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")
enable_language(HIP)

# Include subdirectories
add_subdirectory(src/hipkernels)
add_subdirectory(src/npu_matmul_func)

# Implementation library
add_subdirectory(src/unified_llm_w4a16_hetero)
add_subdirectory(src/microbenchmarks)

# PyBind modules
# PyBind modules are now built from their implementation directories.

# SDPA Performance Test
add_executable(test_sdpa_perf src/microbenchmarks/test_sdpa_perf.cpp)
target_link_libraries(test_sdpa_perf PRIVATE torch ${TORCH_LIBRARIES})
set_property(TARGET test_sdpa_perf PROPERTY CXX_STANDARD 17)
