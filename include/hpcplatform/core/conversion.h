#ifndef hpcplatform_CORE_CONVERSION_H_
#define hpcplatform_CORE_CONVERSION_H_

#include <type_traits>
#include <utility>

#include "hpcplatform/core/context.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/tensor.h"

namespace hpcplatform {

// Customization point for Tensor <-> external type conversion.
//
// Specialize this trait for an external type T (e.g. cv::Mat, a device handle
// exposed by another runtime) to register a conversion. The primary template
// reports kUnsupported, so unregistered types stay diagnosable at runtime
// without exceptions, RTTI, or any dependency from core on a concrete third
// party type.
//
// Contract for specializations:
//   - From(external, context) returns a Tensor bound to `context`. The
//     specialization decides whether the conversion deep-copies host data or
//     wraps memory already owned by another runtime (zero copy); when the
//     target context is a device it also owns the H2D staging. It must document
//     the resulting lifetime semantics.
//   - Into(tensor) returns an owning external value. When the tensor lives in
//     device memory the specialization owns the D2H staging (materialize a
//     non-contiguous view first); it must still document any copy it performs.
template <typename T, typename Enable = void>
struct TensorConvert {
  static Result<Tensor> From(const T& external, Context& context) {
    (void)external;
    (void)context;
    return Status::Unsupported(
        "no Tensor::From conversion registered for this type");
  }

  static Result<T> Into(const Tensor& tensor) {
    (void)tensor;
    return Status::Unsupported(
        "no Tensor::Into conversion registered for this type");
  }
};

// Definitions of the Tensor conversion entry points declared in tensor.h.
template <typename T>
Result<Tensor> Tensor::From(const T& external, Context& context) {
  return TensorConvert<std::decay_t<T>>::From(external, context);
}

template <typename T>
Result<T> Tensor::Into(const Tensor& tensor) {
  return TensorConvert<std::decay_t<T>>::Into(tensor);
}

}  // namespace hpcplatform

#endif  // hpcplatform_CORE_CONVERSION_H_
