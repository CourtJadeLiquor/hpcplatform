#ifndef hpcplatform_INTEROP_CONVERSION_H_
#define hpcplatform_INTEROP_CONVERSION_H_

#include <type_traits>
#include <utility>

#include "hpcplatform/core/context.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/tensor.h"

namespace hpcplatform {

// Compile-time customization point for Tensor <-> external type conversion.
//
// Specialize `TensorConvert<T>` for an external type T (e.g. cv::Mat, a device
// handle exposed by another runtime) to register a conversion. The primary
// template reports kUnsupported, so unregistered types stay diagnosable at
// runtime without exceptions, RTTI, or any dependency from core on concrete
// third-party types.
//
// Why a compile-time trait instead of a virtual interface:
//   - External types (cv::Mat, Eigen::Matrix, user DeviceHandle) cannot inherit
//     an abstract interface — they are third-party code.
//   - The caller already knows the concrete type at instantiation time, so
//     runtime dispatch and a registration registry add cost for no benefit.
//   - The trait keeps Tensor itself free of all interop dependencies: opencv.h
//     sits in hpcplatform/interop/, never polluting core/.
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
  // Uses (void)param to silence `-Wunused-parameter` in the default
  // template — the primary template body never executes for a specialization
  // but the function signature still names the parameters for documentation.
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
// These live here (in interop/, not core/) so that pulling in a conversion
// specialization (interop/opencv.h etc.) also pulls in the dispatch glue,
// while pure core consumers who never call From/Into pay no interop cost.
template <typename T>
Result<Tensor> Tensor::From(const T& external, Context& context) {
  return TensorConvert<std::decay_t<T>>::From(external, context);
}

template <typename T>
Result<T> Tensor::Into(const Tensor& tensor) {
  return TensorConvert<std::decay_t<T>>::Into(tensor);
}

}  // namespace hpcplatform

#endif  // hpcplatform_INTEROP_CONVERSION_H_
