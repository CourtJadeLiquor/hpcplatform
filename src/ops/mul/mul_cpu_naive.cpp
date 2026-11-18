#include "ops/cpu_binary.h"

namespace mycv {
namespace ops_internal {

Status MulCpuNaive(const OpContext& context, const Tensor& input,
                   const Tensor* other, const void* params, Tensor* output) {
  (void)context;
  (void)params;
  return BinaryNaive(input, *other, BinaryOperation::kMul, output);
}

}  // namespace ops_internal
}  // namespace mycv
