#include "backends/cuda/cuda_backend.h"

#include <cuda_runtime_api.h>

#include <memory>
#include <string>

#include "backends/backend_internal.h"
#include "backends/cuda/cuda_common.h"

namespace hpcplatform
{

    Status CudaBackend::Initialize()
    {
        int device_count = 0;
        const cudaError_t count_status = cudaGetDeviceCount(&device_count);
        if (count_status != cudaSuccess)
        {
            return Status::Unsupported(std::string("CUDA is unavailable: ") +
                                       cudaGetErrorString(count_status));
        }
        if (device_count == 0) return Status::Unsupported("no CUDA device is available");
        if (device_.index < 0 || device_.index >= device_count)
        {
            return Status::InvalidArgument("CUDA device index is out of range");
        }
        hpcplatform_RETURN_IF_ERROR(CudaStatus(cudaSetDevice(device_.index), "cudaSetDevice"));
        cudaDeviceProp properties{};
        hpcplatform_RETURN_IF_ERROR(CudaStatus(cudaGetDeviceProperties(&properties, device_.index),
                                               "cudaGetDeviceProperties"));
        if (!properties.managedMemory)
        {
            return Status::Unsupported(
                "CUDA device does not support unified (managed) memory: " +
                std::string(properties.name));
        }
        device_name_ = properties.name;
        compute_major_ = properties.major;
        compute_minor_ = properties.minor;
        hpcplatform_RETURN_IF_ERROR(stream_.Initialize());
        // Allocation-time prefetches join the backend stream, so they are
        // stream-ordered with every kernel launched later on that stream.
        memory_.AttachStream(stream_.handle());
        return Status::Ok();
    }

    std::string CudaBackend::Describe() const
    {
        return "CudaBackend(device=" + device_.ToString() + ", name=" + device_name_ +
               ", compute_capability=" + std::to_string(compute_major_) + "." +
               std::to_string(compute_minor_) + ")";
    }

    Result<std::shared_ptr<Backend>> CreateCudaBackend(int device_index)
    {
        auto backend = std::make_shared<CudaBackend>(device_index);
        hpcplatform_RETURN_IF_ERROR(backend->Initialize());
        return std::static_pointer_cast<Backend>(backend);
    }

}  // namespace hpcplatform
