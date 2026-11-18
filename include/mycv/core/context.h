#ifndef MYCV_CORE_CONTEXT_H_
#define MYCV_CORE_CONTEXT_H_

#include <memory>

#include "mycv/backends/backend.h"
#include "mycv/core/device.h"
#include "mycv/core/error.h"
#include "mycv/core/memory.h"
#include "mycv/core/shape.h"
#include "mycv/core/stream.h"
#include "mycv/core/tensor.h"

namespace mycv {

// Entry point of the library: binds a backend to a device and exposes the
// unified memory manager / stream used to allocate and schedule tensors.
class Context {
 public:
  // No fallback: an unavailable backend returns kUnsupported.
  static Result<std::shared_ptr<Context>> Create(BackendType backend,
                                                 int device_index = 0);

  explicit Context(std::shared_ptr<Backend> backend)
      : backend_(std::move(backend)) {}

  BackendType backend_type() const { return backend_->type(); }
  Device device() const { return backend_->device(); }
  Backend& backend() { return *backend_; }
  const Backend& backend() const { return *backend_; }

  MemoryManager& memory() { return backend_->memory(); }
  const MemoryManager& memory() const { return backend_->memory(); }
  Stream& stream() { return backend_->stream(); }

  Status Synchronize() { return backend_->Synchronize(); }

  Result<Tensor> AllocateTensor(const Shape& shape, DataType dtype,
                                size_t alignment = 64) {
    return Tensor::Create(*this, shape, dtype, alignment);
  }

  // Copies logical elements, including non-contiguous views. The destination
  // context selects the copy path; copies are synchronized before returning.
  Status Copy(Tensor* dst, const Tensor& src, Stream* stream = nullptr);

 private:
  std::shared_ptr<Backend> backend_;
};

}  // namespace mycv

#endif  // MYCV_CORE_CONTEXT_H_
