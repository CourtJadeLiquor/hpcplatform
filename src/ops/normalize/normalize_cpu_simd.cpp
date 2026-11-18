#include "ops/cpu_normalize.h"

namespace mycv {
namespace ops_internal {
Status NormalizeCpuSimdGeneric(const OpContext&, const Tensor& input,
                               const Tensor*, const void* params, Tensor* output) {
  return NormalizeSimdImpl(input, *static_cast<const NormalizeParams*>(params), output);
}
}  // namespace ops_internal
}  // namespace mycv
