#include "hpcplatform/core/context.h"

#include "core/backend.h"

#include <string>
#include <utility>

namespace hpcplatform {

BackendType Context::backend_type() const { return backend_->type(); }

Device Context::device() const { return backend_->device(); }

std::string Context::Describe() const { return backend_->Describe(); }

MemoryManager& Context::memory() { return backend_->memory(); }

const MemoryManager& Context::memory() const { return backend_->memory(); }

Stream& Context::stream() { return backend_->stream(); }

Status Context::Synchronize() { return backend_->Synchronize(); }

bool Context::SupportsKernelLevel(CpuKernelLevel level) const {
  return backend_->SupportsKernelLevel(level);
}

::hpcplatform::MemoryStats Context::GetMemoryStats() const {
  return backend_->memory().Stats();
}

Result<std::shared_ptr<Context>> Context::Create(BackendType backend,
                                                 int device_index) {
  auto backend_result = CreateBackend(backend, device_index);
  if (!backend_result.ok()) return backend_result.status();
  // The constructor is private, so make_shared cannot reach it here.
  return std::shared_ptr<Context>(
      new Context(std::move(backend_result.value())));
}

}  // namespace hpcplatform
