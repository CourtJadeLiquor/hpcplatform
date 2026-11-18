#include "backends/cpu/cpu_backend.h"

#include <initializer_list>

#include "backends/backends_internal.h"
#include "ops/op_registry.h"

namespace mycv {

bool CpuBackend::SupportsKernelLevel(CpuKernelLevel level) const {
  // Report capabilities from the kernels that are actually linked, rather than
  // from compiler flags alone. Add is registered by every CPU kernel family.
  return ops_internal::HasKernel(OpKind::kAdd, BackendType::kNativeCpu, level);
}

std::string CpuBackend::Describe() const {
  std::string description =
      "CpuBackend(device=" + device_.ToString() + ", kernel_levels=";
  bool first = true;
  for (CpuKernelLevel level :
       {CpuKernelLevel::kNaive, CpuKernelLevel::kOpenMP,
        CpuKernelLevel::kSimdGeneric, CpuKernelLevel::kSimdAvx2,
        CpuKernelLevel::kSimdNeon, CpuKernelLevel::kTbb}) {
    if (!SupportsKernelLevel(level)) continue;
    if (!first) description += ",";
    first = false;
    description += CpuKernelLevelName(level);
  }
  description += ")";
  return description;
}

Result<std::shared_ptr<Backend>> CreateCpuBackend(int device_index) {
  std::shared_ptr<Backend> backend = std::make_shared<CpuBackend>(device_index);
  return backend;
}

}  // namespace mycv
