#pragma once

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>

// Macro Function to handle HIP errors
#define HIP_CHECK(call)                                                                                                                    \
    {                                                                                                                                      \
        hipError_t err = (call);                                                                                                           \
        if (err != hipSuccess) {                                                                                                           \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " << hipGetErrorString(err) << " (" << err << ")"            \
                      << std::endl;                                                                                                        \
            std::exit(EXIT_FAILURE);                                                                                                       \
        }                                                                                                                                  \
    }

#define HIPBLASLT_CHECK(call)                                                                                                              \
    {                                                                                                                                      \
        hipblasStatus_t status = (call);                                                                                                   \
        if (status != HIPBLAS_STATUS_SUCCESS) {                                                                                            \
            std::cerr << "HIPBLASLT error at " << __FILE__ << ":" << __LINE__ << ": status=" << static_cast<int>(status) << std::endl;     \
            std::exit(EXIT_FAILURE);                                                                                                       \
        }                                                                                                                                  \
    }
