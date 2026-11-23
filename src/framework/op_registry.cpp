#include "framework/op_registry.h"

#include <map>
#include <mutex>
#include <string>
#include <tuple>

namespace hpcplatform {
namespace ops_internal {

// Defined in src/ops/register_ops.cpp: the single explicit list of operator
// registration entry points. The framework itself never names individual
// operators (docs/architecture.md §6.3).
void RegisterAllKernels();

namespace {

using KernelKey = std::tuple<int, int, int>;

std::map<KernelKey, OpKernelFn>& Kernels() {
  static std::map<KernelKey, OpKernelFn> kernels;
  return kernels;
}

std::once_flag registration_once;

void EnsureRegistered() {
  std::call_once(registration_once, [] { RegisterAllKernels(); });
}

}  // namespace

void RegisterKernel(OpKind op, BackendType backend, CpuKernelLevel level,
                    OpKernelFn kernel) {
  Kernels()[KernelKey(static_cast<int>(op), static_cast<int>(backend),
                      static_cast<int>(level))] = kernel;
}

Result<OpKernelFn> FindKernel(OpKind op, BackendType backend,
                              CpuKernelLevel level) {
  if (backend != BackendType::kNativeCpu && level != CpuKernelLevel::kNaive) {
    return Status::InvalidArgument(
        std::string(BackendTypeName(backend)) +
        " backend accepts only CpuKernelLevel::kNaive; requested " +
        CpuKernelLevelName(level));
  }
  EnsureRegistered();
  const auto& kernels = Kernels();
  const auto it =
      kernels.find(KernelKey(static_cast<int>(op), static_cast<int>(backend),
                             static_cast<int>(level)));
  if (it != kernels.end()) return it->second;
  if (backend == BackendType::kNativeCpu) {
    return Status::Unsupported(std::string("cpu kernel '") + OpKindName(op) +
                               "' is not built for level " +
                               CpuKernelLevelName(level));
  }
  return Status::NotImplemented(std::string("op '") + OpKindName(op) +
                                "' is not implemented for backend " +
                                BackendTypeName(backend));
}

bool HasKernel(OpKind op, BackendType backend, CpuKernelLevel level) {
  EnsureRegistered();
  const auto& kernels = Kernels();
  return kernels.find(KernelKey(static_cast<int>(op), static_cast<int>(backend),
                                static_cast<int>(level))) != kernels.end();
}

bool HasAnyKernel(BackendType backend, CpuKernelLevel level) {
  EnsureRegistered();
  for (const auto& entry : Kernels()) {
    if (std::get<1>(entry.first) == static_cast<int>(backend) &&
        std::get<2>(entry.first) == static_cast<int>(level)) {
      return true;
    }
  }
  return false;
}

}  // namespace ops_internal
}  // namespace hpcplatform
