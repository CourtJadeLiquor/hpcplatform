#include "mycv/ops/add.h"

#include "ops/op_common.h"

namespace mycv {
namespace ops_internal {

Status AddCpuNaive(const OpContext& context, const Tensor& input,
                   const Tensor* other, const void* params, Tensor* output);
#ifdef MYCV_WITH_OPENMP
Status AddCpuOpenMP(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef MYCV_WITH_TBB
Status AddCpuTbb(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef MYCV_WITH_SIMD_GENERIC
Status AddCpuSimdGeneric(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef MYCV_ARCH_AVX2
Status AddCpuAvx2(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif

void RegisterAddKernels() {
  RegisterKernel(OpKind::kAdd, BackendType::kNativeCpu, CpuKernelLevel::kNaive,
                 &AddCpuNaive);
#ifdef MYCV_WITH_OPENMP
  RegisterKernel(OpKind::kAdd, BackendType::kNativeCpu, CpuKernelLevel::kOpenMP,
                 &AddCpuOpenMP);
#endif
#ifdef MYCV_WITH_TBB
  RegisterKernel(OpKind::kAdd, BackendType::kNativeCpu, CpuKernelLevel::kTbb,
                 &AddCpuTbb);
#endif
#ifdef MYCV_WITH_SIMD_GENERIC
  RegisterKernel(OpKind::kAdd, BackendType::kNativeCpu, CpuKernelLevel::kSimdGeneric,
                 &AddCpuSimdGeneric);
#endif
#ifdef MYCV_ARCH_AVX2
  RegisterKernel(OpKind::kAdd, BackendType::kNativeCpu, CpuKernelLevel::kSimdAvx2,
                 &AddCpuAvx2);
#endif
}

}  // namespace ops_internal

Result<Tensor> Add(const OpContext& context, const Tensor& input,
                   const Tensor& other) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateBinaryInputs(input, &other));
  auto kernel = ops_internal::FindKernel(OpKind::kAdd, input.backend(),
                                         context.cpu_level);
  if (!kernel.ok()) return kernel.status();
  auto output = ops_internal::AllocateOutput(context, input.shape(), input.dtype());
  if (!output.ok()) return output.status();
  MYCV_RETURN_IF_ERROR(kernel.value()(context, input, &other, nullptr,
                                      &output.value()));
  return output;
}

}  // namespace mycv
