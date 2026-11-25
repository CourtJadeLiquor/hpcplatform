#ifndef hpcplatform_SRC_BACKENDS_CUDA_CUDA_COMMON_H_
#define hpcplatform_SRC_BACKENDS_CUDA_CUDA_COMMON_H_

#include <cuda_runtime_api.h>

#include <string>

#include "hpcplatform/core/error.h"

namespace hpcplatform
{

    inline Status CudaStatus(cudaError_t error, const char* operation)
    {
        if (error == cudaSuccess) return Status::Ok();
        const std::string message = std::string(operation) + ": " + cudaGetErrorString(error);
        if (error == cudaErrorMemoryAllocation) return Status::OutOfMemory(message);
        return Status::Internal(message);
    }

}  // namespace hpcplatform

#endif  // hpcplatform_SRC_BACKENDS_CUDA_CUDA_COMMON_H_
