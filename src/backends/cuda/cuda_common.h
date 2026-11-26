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

    // Device-side prefetch (cudaMemLocationTypeDevice) requires non-zero
    // concurrentManagedAccess; the runtime rejects it with cudaErrorInvalidDevice
    // otherwise. The result is cached per device ordinal (identical writes race
    // benignly).
    inline bool CudaManagedPrefetchSupported(int device)
    {
        if (device < 0 || device >= 16) return false;
        static bool known[16] = {};
        static bool supported[16] = {};
        if (!known[device])
        {
            int value = 0;
            cudaDeviceGetAttribute(&value, cudaDevAttrConcurrentManagedAccess, device);
            supported[device] = value != 0;
            known[device] = true;
        }
        return supported[device];
    }

    // Prefetches a whole managed-memory block to a device on the given stream.
    // Pass cudaCpuDeviceId as the device to migrate the block back to the host.
    // A no-op on devices without concurrentManagedAccess, where migration falls
    // back to on-demand page faults (correct, only slightly slower on first
    // touch). Adapts to the CUDA 13 cudaMemLocation signature.
    inline Status CudaPrefetch(void* data, size_t bytes, int device, cudaStream_t stream)
    {
        if (!CudaManagedPrefetchSupported(device)) return Status::Ok();
#if CUDART_VERSION >= 13000
        const cudaMemLocation location{cudaMemLocationTypeDevice, device};
        return CudaStatus(cudaMemPrefetchAsync(data, bytes, location, 0, stream),
                          "cudaMemPrefetchAsync");
#else
        return CudaStatus(cudaMemPrefetchAsync(data, bytes, device, stream),
                          "cudaMemPrefetchAsync");
#endif
    }

}  // namespace hpcplatform

#endif  // hpcplatform_SRC_BACKENDS_CUDA_CUDA_COMMON_H_
