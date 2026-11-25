#ifndef hpcplatform_SRC_FRAMEWORK_DTYPE_DISPATCH_H_
#define hpcplatform_SRC_FRAMEWORK_DTYPE_DISPATCH_H_

#include "hpcplatform/core/common.h"
#include "hpcplatform/core/error.h"

// Loop-vectorization hint for portable SIMD kernels, kept next to the image
// dtype dispatch so image kernels get both with a single include. The guard
// below avoids double definition; DispatchImageDType dispatches u8/f32/f64.
#ifndef hpcplatform_CPU_IVDEP
#if defined(__GNUC__) && !defined(__clang__)
#define hpcplatform_CPU_IVDEP _Pragma("GCC ivdep")
#else
#define hpcplatform_CPU_IVDEP
#endif
#endif

namespace hpcplatform {
namespace ops_internal {

template <typename T>
struct DTypeTag {
  using type = T;
};

// Image operators support u8 / f32 / f64 only (see IsSupportedImageDType in
// framework/image_utils.h). Replaces the 8-type arithmetic dispatch that ships
// with the elementwise operators.
template <typename Fn>
Status DispatchImageDType(DataType dtype, Fn&& fn) {
  switch (dtype) {
    case DataType::kUInt8:
      return fn(DTypeTag<uint8_t>{});
    case DataType::kFloat32:
      return fn(DTypeTag<float>{});
    case DataType::kFloat64:
      return fn(DTypeTag<double>{});
    default:
      break;
  }
  return Status::InvalidArgument("image ops support only u8, f32 and f64");
}

}  // namespace ops_internal
}  // namespace hpcplatform

#endif  // hpcplatform_SRC_FRAMEWORK_DTYPE_DISPATCH_H_
