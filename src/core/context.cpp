#include "hpcplatform/core/context.h"

#include "core/backend.h"

#include <string>
#include <utility>

namespace hpcplatform {

BackendType Context::backend_type() const {
  return backend_->type();
}

Device Context::device() const {
  return backend_->device();
}

std::string Context::Describe() const {
  return backend_->Describe();
}

MemoryManager& Context::memory() {
  return backend_->memory();
}

const MemoryManager& Context::memory() const {
  return backend_->memory();
}

Stream& Context::stream() {
  return backend_->stream();
}

Status Context::Synchronize() {
  return backend_->Synchronize();
}

Result<std::shared_ptr<Context>> Context::Create(BackendType backend,
                                                 int device_index) {
  auto backend_result = CreateBackend(backend, device_index);
  if (!backend_result.ok()) return backend_result.status();
  return std::make_shared<Context>(std::move(backend_result.value()));
}

Result<Tensor> Context::Wrap(void* data, const Shape& shape, DataType dtype,
                             MemoryKind kind, Buffer::Deleter deleter) {
  return Tensor::Wrap(backend_, data, shape, dtype, kind, std::move(deleter));
}

}  // namespace hpcplatform
