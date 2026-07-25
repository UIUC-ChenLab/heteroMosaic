cmake_minimum_required(VERSION 3.10)

include_directories(${ROCM_PATH}/include /usr/include/libdrm ${DRIVER_PATH}/src/include/uapi/drm_local)
include_directories(${CMAKE_SOURCE_DIR}/include)

# Create a library from npu_matmul_func.cpp
add_library(npu_matmul_func STATIC npu_matmul_func.cpp)
target_compile_options(npu_matmul_func PRIVATE -O0 -g -fPIC) # Must compile with -O0 or -O1, -fPIC for shared library linking
target_include_directories(npu_matmul_func PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${ROCM_PATH}/include
    /usr/include/libdrm
    ${DRIVER_PATH}/src/include/uapi/drm_local
)

find_library(HSA_RUNTIME_LIB hsa-runtime64 PATHS ${ROCM_PATH}/lib NO_DEFAULT_PATH)
if(NOT HSA_RUNTIME_LIB)
    message(FATAL_ERROR "hsa-runtime64 not found in ${ROCM_PATH}/lib")
endif()

target_link_libraries(npu_matmul_func PRIVATE ${HSA_RUNTIME_LIB} drm)
set_property(TARGET npu_matmul_func PROPERTY CXX_STANDARD 17)

message(STATUS "npu_matmul_func library configured")

