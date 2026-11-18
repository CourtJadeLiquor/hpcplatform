#ifndef MYCV_SRC_OPS_OP_REGISTRY_H_
#define MYCV_SRC_OPS_OP_REGISTRY_H_

#include "mycv/core/device.h"
#include "mycv/core/error.h"
#include "mycv/core/kernel_level.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/op.h"

namespace mycv {
namespace ops_internal {

// 'other' is used by binary ops; 'params' points to the op-specific immutable
// parameter struct for the duration of the synchronous kernel call.
using OpKernelFn = Status (*)(const OpContext&, const Tensor& input,
                              const Tensor* other, const void* params,
                              Tensor* output);

void RegisterKernel(OpKind op, BackendType backend, CpuKernelLevel level,
                    OpKernelFn kernel);
Result<OpKernelFn> FindKernel(OpKind op, BackendType backend,
                              CpuKernelLevel level);
bool HasKernel(OpKind op, BackendType backend, CpuKernelLevel level);

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_OP_REGISTRY_H_
