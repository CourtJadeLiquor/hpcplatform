#include "mycv/ops/transpose.h"

#include "ops/op_common.h"
#include "ops/op_params.h"

namespace mycv {
namespace ops_internal {

Status TransposeCpuNaive(const OpContext& context, const Tensor& input,
                         const Tensor* other, const void* params, Tensor* output);
#ifdef MYCV_WITH_OPENMP
Status TransposeCpuOpenMP(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef MYCV_WITH_TBB
Status TransposeCpuTbb(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif

void RegisterTransposeKernels() {
  RegisterKernel(OpKind::kTranspose, BackendType::kNativeCpu,
                 CpuKernelLevel::kNaive, &TransposeCpuNaive);
#ifdef MYCV_WITH_OPENMP
  RegisterKernel(OpKind::kTranspose, BackendType::kNativeCpu,
                 CpuKernelLevel::kOpenMP, &TransposeCpuOpenMP);
#endif
#ifdef MYCV_WITH_TBB
  RegisterKernel(OpKind::kTranspose, BackendType::kNativeCpu,
                 CpuKernelLevel::kTbb, &TransposeCpuTbb);
#endif
}

}  // namespace ops_internal

Result<Tensor> Transpose(const OpContext& context, const Tensor& input,
                         const std::vector<int>& order) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  auto view = input.Permute(order);
  if (!view.ok()) return view.status();
  auto kernel = ops_internal::FindKernel(OpKind::kTranspose, input.backend(),
                                         context.cpu_level);
  if (!kernel.ok()) return kernel.status();
  auto output = ops_internal::AllocateOutput(context, view.value().shape(),
                                              input.dtype());
  if (!output.ok()) return output.status();
  const ops_internal::TransposeParams params{&order};
  MYCV_RETURN_IF_ERROR(kernel.value()(context, input, nullptr, &params,
                                      &output.value()));
  return output;
}

}  // namespace mycv
