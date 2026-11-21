// 算子公共层：调用前校验与输出分配。
//
// ValidateInvocation 统一检查 OpContext 与输入张量是否匹配，避免每个算子
// 重复实现；AllocateOutput 把输出内存交给 Context，保证与输入同一后端/设备。

#ifndef MYCV_SRC_OPS_COMMON_OP_COMMON_H_
#define MYCV_SRC_OPS_COMMON_OP_COMMON_H_

#include <string>

#include "framework/op_registry.h"
#include "mycv/core/context.h"

namespace mycv {
namespace ops_internal {

// 校验一次算子调用的合法性，返回第一个失败项：
//   1. context.context 非空（后续分配与 stream 都依赖它）；
//   2. input 已定义；
//   3. 张量后端、设备类型与设备序号与 context 一致；
//   4. 非 CPU 后端只接受 kNaive（CPU 后端才允许 OpenMP/TBB/SIMD）。
// 校验通过返回 Ok，不修改任何状态；复杂度 O(1)。
inline Status ValidateInvocation(const OpContext& context,
                                 const Tensor& input) {
  if (context.context == nullptr) {
    return Status::InvalidArgument("OpContext.context must not be null");
  }
  if (!input.defined())
    return Status::InvalidArgument("input tensor is undefined");
  if (input.backend() != context.context->backend_type() ||
      input.device().type != context.context->device().type ||
      input.device().index != context.context->device().index) {
    return Status::InvalidArgument(
        "input tensor belongs to a different backend/device");
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

// 通过 Context 分配一个输出张量，形状与 dtype 由调用者给出。
// 具体内存位置（host/device）由 Context 的后端决定；算子无需关心。
// 复杂度取决于底层分配器。
inline Result<Tensor> AllocateOutput(const OpContext& context,
                                     const Shape& shape, DataType dtype) {
  return context.context->AllocateTensor(shape, dtype);
}

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_COMMON_OP_COMMON_H_
