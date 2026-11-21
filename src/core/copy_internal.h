#ifndef MYCV_SRC_CORE_COPY_INTERNAL_H_
#define MYCV_SRC_CORE_COPY_INTERNAL_H_

#include <memory>

#include "mycv/core/error.h"

namespace mycv {

class Backend;
class Stream;
class Tensor;

// Element-wise copy used by Context::Copy and Tensor::Clone. The backend that
// owns the device side drives the transfer; host-only copies use `backend`.
Status CopyTensor(Backend& backend, Tensor* dst, const Tensor& src,
                  Stream* stream);

// Returns the backend that allocated the tensor. Buffers outlive their Context,
// so when the original backend is gone a fresh one is created for the transfer
// instead of dereferencing a dead allocator.
Result<std::shared_ptr<Backend>> ResolveOwningBackend(const Tensor& tensor);

}  // namespace mycv

#endif  // MYCV_SRC_CORE_COPY_INTERNAL_H_
