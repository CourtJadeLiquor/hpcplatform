// Backend factory: the only place that maps a BackendType to a concrete
// backend implementation. Context::Create is the sole public entry point.

#include <memory>

#include "backends/backend_internal.h"

namespace hpcplatform
{

    Result<std::shared_ptr<Backend>> CreateBackend(BackendType type, int device_index)
    {
        switch (type)
        {
            case BackendType::kNativeCpu:
                return CreateCpuBackend(device_index);
            case BackendType::kOpenCV:
#ifdef hpcplatform_WITH_OPENCV
                return CreateOpenCvBackend(device_index);
#else
                return Status::Unsupported("opencv backend is not built");
#endif
            case BackendType::kCUDA:
#ifdef hpcplatform_WITH_CUDA
                return CreateCudaBackend(device_index);
#else
                return Status::Unsupported("cuda backend is not built");
#endif
            case BackendType::kNPU:
                return CreateNpuBackend(device_index);
        }
        return Status::InvalidArgument("unknown backend type");
    }

}  // namespace hpcplatform
