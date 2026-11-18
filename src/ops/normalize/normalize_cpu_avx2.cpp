#include <immintrin.h>

#include "ops/op_common.h"
#include "ops/op_params.h"

namespace mycv {
namespace ops_internal {
Status NormalizeCpuAvx2(const OpContext&, const Tensor& input, const Tensor*,
                        const void* params, Tensor* output) {
  const int64_t channels = input.shape().Dim(input.shape().Rank() - 1);
  if (input.dtype() != DataType::kFloat32 || !input.IsContiguous() || channels != 1) {
    return Status::Unsupported(
        "AVX2 Normalize currently requires contiguous single-channel f32 input");
  }
  const auto& options = *static_cast<const NormalizeParams*>(params);
  const __m256 mean = _mm256_set1_ps(static_cast<float>((*options.mean)[0]));
  const __m256 denominator =
      _mm256_set1_ps(static_cast<float>((*options.std)[0] + options.epsilon));
  const float* source = input.Data<float>();
  float* destination = output->Data<float>();
  int64_t i = 0;
  for (; i + 8 <= input.Numel(); i += 8) {
    const __m256 value = _mm256_loadu_ps(source + i);
    _mm256_storeu_ps(destination + i,
                     _mm256_div_ps(_mm256_sub_ps(value, mean), denominator));
  }
  const float mean_scalar = static_cast<float>((*options.mean)[0]);
  const float std_scalar = static_cast<float>((*options.std)[0] + options.epsilon);
  for (; i < input.Numel(); ++i) {
    destination[i] = (source[i] - mean_scalar) / std_scalar;
  }
  return Status::Ok();
}
}  // namespace ops_internal
}  // namespace mycv
