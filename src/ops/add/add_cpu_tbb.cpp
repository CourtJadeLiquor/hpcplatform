#include "ops/cpu_binary.h"

namespace mycv {
namespace ops_internal {
Status AddCpuTbb(const OpContext&, const Tensor& input, const Tensor* other,
                 const void*, Tensor* output) {
  return BinaryTbb(input, *other, BinaryOperation::kAdd, output);
}
}  // namespace ops_internal
}  // namespace mycv
