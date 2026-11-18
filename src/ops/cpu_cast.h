#ifndef MYCV_SRC_OPS_CPU_CAST_H_
#define MYCV_SRC_OPS_CPU_CAST_H_

#include "core/value_cast.h"
#include "ops/cpu_dispatch.h"
#include "ops/op_common.h"

namespace mycv {
namespace ops_internal {

inline Status CastVectorImpl(const Tensor& input, Tensor* output) {
  if (!input.IsContiguous()) {
    return Status::Unsupported("SIMD Cast requires a contiguous input");
  }
  const int64_t count = input.Numel();
  if (input.dtype() == DataType::kFloat32 && output->dtype() == DataType::kUInt8) {
    const float* __restrict source = input.Data<float>();
    uint8_t* __restrict destination = output->Data<uint8_t>();
    MYCV_CPU_IVDEP
    for (int64_t i = 0; i < count; ++i) {
      destination[i] = detail::SaturatingCast<uint8_t>(source[i]);
    }
    return Status::Ok();
  }
  if (input.dtype() == DataType::kUInt8 && output->dtype() == DataType::kFloat32) {
    const uint8_t* __restrict source = input.Data<uint8_t>();
    float* __restrict destination = output->Data<float>();
    MYCV_CPU_IVDEP
    for (int64_t i = 0; i < count; ++i) destination[i] = source[i];
    return Status::Ok();
  }
  if (input.dtype() == DataType::kFloat32 && output->dtype() == DataType::kFloat64) {
    const float* __restrict source = input.Data<float>();
    double* __restrict destination = output->Data<double>();
    MYCV_CPU_IVDEP
    for (int64_t i = 0; i < count; ++i) destination[i] = source[i];
    return Status::Ok();
  }
  if (input.dtype() == DataType::kFloat64 && output->dtype() == DataType::kFloat32) {
    const double* __restrict source = input.Data<double>();
    float* __restrict destination = output->Data<float>();
    MYCV_CPU_IVDEP
    for (int64_t i = 0; i < count; ++i) destination[i] = static_cast<float>(source[i]);
    return Status::Ok();
  }
  return Status::Unsupported(
      "SIMD Cast supports contiguous f32<->u8 and f32<->f64 only");
}

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_CPU_CAST_H_
