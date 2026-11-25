#ifndef hpcplatform_SRC_CORE_TENSOR_INTERNAL_H_
#define hpcplatform_SRC_CORE_TENSOR_INTERNAL_H_

#include <utility>

#include "hpcplatform/core/context.h"
#include "hpcplatform/core/tensor.h"

namespace hpcplatform {
namespace internal {

// Internal factory for tensors that alias memory owned by another runtime.
// Kept out of the public API: production conversions go through
// Tensor::From<T>, and only backend adapters / low-level tests need this.
struct TensorFactory {
  static Result<Tensor> WrapExternal(Context& context, void* data,
                                     const Shape& shape, DataType dtype,
                                     MemoryKind kind = MemoryKind::kHost,
                                     Buffer::Deleter deleter = nullptr) {
    return Tensor::WrapExternal(context, data, shape, dtype, kind,
                                std::move(deleter));
  }
};

}  // namespace internal
}  // namespace hpcplatform

#endif  // hpcplatform_SRC_CORE_TENSOR_INTERNAL_H_
