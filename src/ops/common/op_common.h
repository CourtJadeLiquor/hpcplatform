// Common operator layer: pre-invocation validation and output allocation.
//
// ValidateInvocation centrally checks that OpContext matches the input tensor,
// avoiding duplicate checks in each operator; AllocateOutput delegates output
// allocation to Tensor::Empty through the bound Context, ensuring the same
// backend/device as the input.

#ifndef hpcplatform_SRC_OPS_COMMON_OP_COMMON_H_
#define hpcplatform_SRC_OPS_COMMON_OP_COMMON_H_

#include <string>

#include "framework/op_registry.h"
#include "hpcplatform/core/context.h"

namespace hpcplatform {
namespace ops_internal {

// Validate an operator invocation and return the first failure:
//   1. context.context() is non-null (allocation and stream depend on it);
//   2. input is defined;
//   3. the tensor backend, device type, and device index match context;
//   4. non-CPU backends accept only kNaive (OpenMP/TBB/SIMD are CPU-only).
// Return Ok without modifying any state if validation passes; O(1).
inline Status ValidateInvocation(const OpContext& context,
                                 const Tensor& input) {
  if (context.context() == nullptr) {
    return Status::InvalidArgument("OpContext.context() must not be null");
  }
  if (!input.defined())
    return Status::InvalidArgument("input tensor is undefined");
  if (input.backend() != context.context()->backend_type() ||
      input.device().type != context.context()->device().type ||
      input.device().index != context.context()->device().index) {
    return Status::InvalidArgument(
        "input tensor belongs to a different backend/device");
  }
  if (input.backend() != BackendType::kNativeCpu &&
      context.cpu_level() != CpuKernelLevel::kNaive) {
    return Status::InvalidArgument(
        std::string(BackendTypeName(input.backend())) +
        " backend accepts only CpuKernelLevel::kNaive; requested " +
        CpuKernelLevelName(context.cpu_level()));
  }
  return Status::Ok();
}

// Allocate an output tensor through Context with the caller-provided shape and
// dtype. The Context backend determines the memory location (host/device);
// operators need not handle this detail. Complexity depends on the allocator.
inline Result<Tensor> AllocateOutput(const OpContext& context,
                                     const Shape& shape, DataType dtype) {
  return Tensor::Empty(*context.context(), shape, dtype);
}

}  // namespace ops_internal
}  // namespace hpcplatform

#endif  // hpcplatform_SRC_OPS_COMMON_OP_COMMON_H_
