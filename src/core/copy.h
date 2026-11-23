#ifndef hpcplatform_SRC_CORE_COPY_H_
#define hpcplatform_SRC_CORE_COPY_H_

#include <memory>

#include "hpcplatform/core/error.h"

// Core-internal copy contract: declared in src/core (never exposed through the
// public API) and implemented in src/core/copy.cpp. Callers are Context::Copy
// and Tensor::Clone.
namespace hpcplatform {

class Backend;
class Stream;
class Tensor;

// Copies element data from `src` into a caller-allocated `dst`; used by
// Context::Copy and Tensor::Clone. The backend that owns the device side
// drives the transfer; host-only copies use `backend`. Metadata (shape /
// dtype / strides) is the caller's responsibility and is never copied here.
Status CopyTensorData(const Backend& backend, Tensor* dst, const Tensor& src,
                      Stream* stream);

// Returns the backend that allocated the tensor. Buffers outlive their Context,
// so when the original backend is gone a fresh one is created for the transfer
// instead of dereferencing a dead allocator.
Result<std::shared_ptr<Backend>> ResolveOwningBackend(const Tensor& tensor);

}  // namespace hpcplatform

#endif  // hpcplatform_SRC_CORE_COPY_H_
