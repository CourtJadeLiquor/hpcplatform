#ifndef MYCV_SRC_FRAMEWORK_OP_REGISTRY_H_
#define MYCV_SRC_FRAMEWORK_OP_REGISTRY_H_

#include "mycv/core/device.h"
#include "mycv/core/error.h"
#include "mycv/core/kernel_level.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/op.h"

namespace mycv {
namespace ops_internal {

// 内核函数签名：other 供二元算子使用；params 指向本次调用的算子参数结构体，
// 生命周期仅限同步调用期间。
using OpKernelFn = Status (*)(const OpContext&, const Tensor& input,
                              const Tensor* other, const void* params,
                              Tensor* output);

// 注册一个 (op, backend, level) 内核；重复注册同一键会被拒绝。
void RegisterKernel(OpKind op, BackendType backend, CpuKernelLevel level,
                    OpKernelFn kernel);
// 查找内核：未编译返回 kUnsupported，已编译但未实现返回 kNotImplemented。
Result<OpKernelFn> FindKernel(OpKind op, BackendType backend,
                              CpuKernelLevel level);
// 是否存在精确匹配的内核。
bool HasKernel(OpKind op, BackendType backend, CpuKernelLevel level);

// 该 (backend, level) 下是否至少注册了一个算子；用于后端能力查询，
// 不依赖任何单个算子。
bool HasAnyKernel(BackendType backend, CpuKernelLevel level);

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_FRAMEWORK_OP_REGISTRY_H_
