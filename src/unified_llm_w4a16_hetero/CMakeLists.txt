add_library(unified_llm_w4a16_hetero_lib STATIC
    unified_llm_w4a16.cpp
    npuSetup.cpp
    hetero_compute.cpp
    helper.cpp
    ../cpu_avx_kernels/w4a16_gemv_avx_packed.cpp
    ../cpu_avx_kernels/w4a16_gemv_avx_unpacked.cpp
)

if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH} AND EXISTS "$ENV{ROCM_PATH}")
        set(ROCM_PATH "$ENV{ROCM_PATH}")
    elseif(EXISTS "/opt/rocm")
        set(ROCM_PATH "/opt/rocm")
    elseif(EXISTS "/opt/rocm-7.2.0")
        set(ROCM_PATH "/opt/rocm-7.2.0")
    else()
        set(ROCM_PATH "/opt/rocm")
    endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${ROCM_PATH}/include")
set(ENV{ROCM_PATH} "${ROCM_PATH}")

find_package(hip REQUIRED)
message(STATUS "Using ROCM_PATH: ${ROCM_PATH}")

target_include_directories(unified_llm_w4a16_hetero_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${ROCM_PATH}/include
    ${ROCM_PATH}/include/hip
)

find_library(HSA_RUNTIME_LIB hsa-runtime64 PATHS ${ROCM_PATH}/lib NO_DEFAULT_PATH)
if(NOT HSA_RUNTIME_LIB)
    message(FATAL_ERROR "hsa-runtime64 not found in ${ROCM_PATH}/lib")
endif()

target_link_libraries(unified_llm_w4a16_hetero_lib PUBLIC
    "${TORCH_LIBRARIES}"
    npu_matmul_func
    hipkernels
    ${HSA_RUNTIME_LIB}
    amdhip64
)

target_compile_definitions(unified_llm_w4a16_hetero_lib PUBLIC
    __HIP_PLATFORM_AMD__=1
)

target_compile_options(unified_llm_w4a16_hetero_lib PRIVATE
    -march=native
    -O3
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(unified_llm_w4a16_hetero_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

set_target_properties(unified_llm_w4a16_hetero_lib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Python bindings were merged into this directory from unified_llm_w4a16_hetero_pybind.
find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_PYTHON_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/share/cmake/Torch")
    set(CMAKE_PREFIX_PATH "${TORCH_PYTHON_PATH}/share/cmake/Torch" ${CMAKE_PREFIX_PATH})
    message(STATUS "Found PyTorch in Python environment: ${TORCH_PYTHON_PATH}")
endif()

find_package(Torch QUIET)
if(NOT Torch_FOUND)
    message(STATUS "PyTorch not found in Python environment, using standalone LibTorch")
endif()

if(CMAKE_CXX_FLAGS)
    string(REPLACE "-Wno-duplicate-decl-specifier" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()
if(TORCH_CXX_FLAGS)
    string(REPLACE "-Wno-duplicate-decl-specifier" "" TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
endif()

get_target_property(PYBIND11_INCLUDE_DIR pybind11::headers INTERFACE_INCLUDE_DIRECTORIES)
if(NOT PYBIND11_INCLUDE_DIR)
    get_target_property(PYBIND11_INCLUDE_DIR pybind11::pybind11 INTERFACE_INCLUDE_DIRECTORIES)
endif()
if(NOT PYBIND11_INCLUDE_DIR)
    set(PYBIND11_INCLUDE_DIR ${Python_INCLUDE_DIRS})
endif()

add_library(unified_llm_w4a16_hetero_libtorch MODULE
    python_bindings.cpp
)

set_target_properties(unified_llm_w4a16_hetero_libtorch PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    PREFIX ""
    OUTPUT_NAME "unified_llm_w4a16_hetero_libtorch"
)

target_include_directories(unified_llm_w4a16_hetero_libtorch PRIVATE
    ${PYBIND11_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
    ${DRIVER_PATH}/src/include/uapi/drm_local
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

if(TORCH_PYTHON_PATH)
    if(EXISTS "${TORCH_PYTHON_PATH}/include")
        target_include_directories(unified_llm_w4a16_hetero_libtorch PRIVATE "${TORCH_PYTHON_PATH}/include")
    endif()
    if(EXISTS "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
        target_include_directories(unified_llm_w4a16_hetero_libtorch PRIVATE "${TORCH_PYTHON_PATH}/include/torch/csrc/api/include")
    endif()
endif()

if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/lib")
    link_directories("${TORCH_PYTHON_PATH}/lib")
endif()
link_directories(${ROCM_PATH}/lib)

target_link_libraries(unified_llm_w4a16_hetero_libtorch PRIVATE
    unified_llm_w4a16_hetero_lib
    "${TORCH_LIBRARIES}"
    pybind11::module
    pybind11::windows_extras
    npu_matmul_func
    ${HSA_RUNTIME_LIB}
)

if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/lib")
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_PYTHON_PATH}/lib" NO_DEFAULT_PATH)
    if(TORCH_PYTHON_LIBRARY)
        target_link_libraries(unified_llm_w4a16_hetero_libtorch PRIVATE ${TORCH_PYTHON_LIBRARY})
        message(STATUS "Found torch_python library: ${TORCH_PYTHON_LIBRARY}")
    else()
        message(WARNING "torch_python library not found in ${TORCH_PYTHON_PATH}/lib")
    endif()
endif()

if(TORCH_PYTHON_PATH AND EXISTS "${TORCH_PYTHON_PATH}/lib")
    set_target_properties(unified_llm_w4a16_hetero_libtorch PROPERTIES
        INSTALL_RPATH "${TORCH_PYTHON_PATH}/lib"
        BUILD_RPATH "${TORCH_PYTHON_PATH}/lib"
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
else()
    set_target_properties(unified_llm_w4a16_hetero_libtorch PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()

target_compile_definitions(unified_llm_w4a16_hetero_libtorch PRIVATE
    VERSION_INFO=${pybind11_VERSION}
    __HIP_PLATFORM_AMD__=1
)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/py/unified_llm_w4a16")
set_target_properties(unified_llm_w4a16_hetero_libtorch PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/py/unified_llm_w4a16"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/py/unified_llm_w4a16"
)
