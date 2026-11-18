#include "ops/cpu_transpose.h"

namespace mycv {
namespace ops_internal {
Status TransposeCpuOpenMP(const OpContext&, const Tensor& input, const Tensor*,
                          const void* params, Tensor* output) {
  return TransposeOpenMPImpl(input, *static_cast<const TransposeParams*>(params), output);
}
}  // namespace ops_internal
}  // namespace mycv
