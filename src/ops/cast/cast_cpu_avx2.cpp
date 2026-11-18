#include "ops/cpu_cast.h"

namespace mycv {
namespace ops_internal {
Status CastCpuAvx2(const OpContext&, const Tensor& input, const Tensor*,
                   const void*, Tensor* output) {
  return CastVectorImpl(input, output);
}
}  // namespace ops_internal
}  // namespace mycv
