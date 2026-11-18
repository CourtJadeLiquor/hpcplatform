#include "ops/cpu_binary.h"

namespace mycv {
namespace ops_internal {
Status MulCpuTbb(const OpContext&, const Tensor& input, const Tensor* other,
                 const void*, Tensor* output) {
  return BinaryTbb(input, *other, BinaryOperation::kMul, output);
}
}  // namespace ops_internal
}  // namespace mycv
