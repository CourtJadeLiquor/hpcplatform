#include "ops/cpu_transpose.h"

namespace mycv {
namespace ops_internal {
Status TransposeCpuTbb(const OpContext&, const Tensor& input, const Tensor*,
                       const void* params, Tensor* output) {
  return TransposeTbbImpl(input, *static_cast<const TransposeParams*>(params), output);
}
}  // namespace ops_internal
}  // namespace mycv
