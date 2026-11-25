#include "backends/cpu/cpu_backend.h"

#include <initializer_list>

#include "backends/backend_internal.h"
#include "framework/op_registry.h"

namespace hpcplatform
{

    bool CpuBackend::SupportsKernelLevel(CpuKernelLevel level) const
    {
        // Report capabilities from the kernels that are actually linked, rather than
        // from compiler flags alone. Scanning all ops (not one probe op) keeps this
        // correct when an operator has no implementation for a given level.
        return ops_internal::HasAnyKernel(BackendType::kNativeCpu, level);
    }

    std::string CpuBackend::Describe() const
    {
        std::string description = "CpuBackend(device=" + device_.ToString() + ", kernel_levels=";
        bool first = true;
        for (CpuKernelLevel level :
             {CpuKernelLevel::kNaive, CpuKernelLevel::kOpenMP, CpuKernelLevel::kSimdGeneric,
              CpuKernelLevel::kSimdAvx2, CpuKernelLevel::kSimdNeon, CpuKernelLevel::kTbb})
        {
            if (!SupportsKernelLevel(level)) continue;
            if (!first) description += ",";
            first = false;
            description += CpuKernelLevelName(level);
        }
        description += ")";
        return description;
    }

    Result<std::shared_ptr<Backend>> CreateCpuBackend(int device_index)
    {
        std::shared_ptr<Backend> backend = std::make_shared<CpuBackend>(device_index);
        return backend;
    }

}  // namespace hpcplatform
