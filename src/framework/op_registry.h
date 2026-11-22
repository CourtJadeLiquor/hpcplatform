#ifndef MYCV_SRC_FRAMEWORK_OP_REGISTRY_H_
#define MYCV_SRC_FRAMEWORK_OP_REGISTRY_H_

#include "mycv/core/device.h"
#include "mycv/core/error.h"
#include "mycv/core/kernel_level.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/op.h"

namespace mycv {
namespace ops_internal {

// Kernel function signature: other is for binary operators; params points to the
// operator parameter structure for this call, with a lifetime limited to the
// synchronous call.
using OpKernelFn = Status (*)(const OpContext&, const Tensor& input,
                              const Tensor* other, const void* params,
                              Tensor* output);

// Registers a kernel for (op, backend, level); duplicate keys are rejected.
void RegisterKernel(OpKind op, BackendType backend, CpuKernelLevel level,
                    OpKernelFn kernel);
// Finds a kernel: returns kUnsupported if not compiled, or kNotImplemented if
// compiled but not implemented.
Result<OpKernelFn> FindKernel(OpKind op, BackendType backend,
                              CpuKernelLevel level);
// Returns whether an exactly matching kernel exists.
bool HasKernel(OpKind op, BackendType backend, CpuKernelLevel level);

// Returns whether at least one operator is registered for (backend, level); used
// for backend capability queries without depending on any individual operator.
bool HasAnyKernel(BackendType backend, CpuKernelLevel level);

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_FRAMEWORK_OP_REGISTRY_H_
