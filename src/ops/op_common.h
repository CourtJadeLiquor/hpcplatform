#ifndef MYCV_SRC_OPS_OP_COMMON_H_
#define MYCV_SRC_OPS_OP_COMMON_H_

#include <string>

#include "mycv/core/context.h"
#include "mycv/core/tensor_iterator.h"
#include "ops/op_registry.h"

namespace mycv {
namespace ops_internal {

inline Status ValidateInvocation(const OpContext& context, const Tensor& input) {
  if (context.context == nullptr) {
    return Status::InvalidArgument("OpContext.context must not be null");
  }
  if (!input.defined()) return Status::InvalidArgument("input tensor is undefined");
  if (input.backend() != context.context->backend_type() ||
      input.device().type != context.context->device().type ||
      input.device().index != context.context->device().index) {
    return Status::InvalidArgument("input tensor belongs to a different backend/device");
  }
  if (input.backend() != BackendType::kNativeCpu &&
      context.cpu_level != CpuKernelLevel::kNaive) {
    return Status::InvalidArgument(
        std::string(BackendTypeName(input.backend())) +
        " backend accepts only CpuKernelLevel::kNaive; requested " +
        CpuKernelLevelName(context.cpu_level));
  }
  return Status::Ok();
}

inline Result<Tensor> AllocateOutput(const OpContext& context, const Shape& shape,
                                     DataType dtype) {
  return context.context->AllocateTensor(shape, dtype);
}

inline bool IsSupportedArithmeticDType(DataType dtype) {
  return dtype != DataType::kFloat16 && DTypeSize(dtype) != 0;
}

inline Status ValidateBinaryInputs(const Tensor& input, const Tensor* other) {
  if (other == nullptr || !other->defined()) {
    return Status::InvalidArgument("binary op requires a defined second input");
  }
  if (input.backend() != other->backend() ||
      input.device().type != other->device().type ||
      input.device().index != other->device().index ||
      input.dtype() != other->dtype()) {
    return Status::InvalidArgument("binary inputs must share backend, device and dtype");
  }
  if (input.shape() != other->shape() && other->Numel() != 1) {
    return Status::InvalidArgument("binary inputs must match shape or use a scalar rhs");
  }
  if (!IsSupportedArithmeticDType(input.dtype())) {
    return Status::Unsupported("binary op does not support this dtype");
  }
  return Status::Ok();
}

inline size_t StridedByteOffset(const Tensor& tensor, int64_t linear_index) {
  size_t offset = 0;
  for (int dim = tensor.shape().Rank() - 1; dim >= 0; --dim) {
    const int64_t extent = tensor.shape().Dim(dim);
    const int64_t coordinate = linear_index % extent;
    linear_index /= extent;
    offset += static_cast<size_t>(coordinate * tensor.Stride(dim)) *
              tensor.ElementSize();
  }
  return offset;
}

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_OP_COMMON_H_
