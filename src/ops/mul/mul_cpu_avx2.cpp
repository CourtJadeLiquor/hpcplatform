#include <immintrin.h>

#include "ops/op_common.h"

namespace mycv {
namespace ops_internal {
Status MulCpuAvx2(const OpContext&, const Tensor& input, const Tensor* other,
                  const void*, Tensor* output) {
  if (input.dtype() != DataType::kFloat32 || !input.IsContiguous() ||
      (other->Numel() != 1 && !other->IsContiguous())) {
    return Status::Unsupported("AVX2 Mul requires contiguous f32 inputs");
  }
  const float* lhs = input.Data<float>();
  const float* rhs = other->Data<float>();
  float* dst = output->Data<float>();
  const int64_t count = input.Numel();
  const bool scalar = other->Numel() == 1;
  int64_t i = 0;
  if (scalar) {
    const __m256 value = _mm256_set1_ps(rhs[0]);
    for (; i + 8 <= count; i += 8) {
      _mm256_storeu_ps(dst + i, _mm256_mul_ps(_mm256_loadu_ps(lhs + i), value));
    }
    for (; i < count; ++i) dst[i] = lhs[i] * rhs[0];
  } else {
    for (; i + 8 <= count; i += 8) {
      _mm256_storeu_ps(dst + i,
                       _mm256_mul_ps(_mm256_loadu_ps(lhs + i),
                                     _mm256_loadu_ps(rhs + i)));
    }
    for (; i < count; ++i) dst[i] = lhs[i] * rhs[i];
  }
  return Status::Ok();
}
}  // namespace ops_internal
}  // namespace mycv
