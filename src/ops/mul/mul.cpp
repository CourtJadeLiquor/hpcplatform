#include "mycv/ops/mul.h"

#include "ops/op_common.h"

namespace mycv {
namespace ops_internal {

Status MulCpuNaive(const OpContext& context, const Tensor& input,
                   const Tensor* other, const void* params, Tensor* output);
#ifdef MYCV_WITH_OPENMP
Status MulCpuOpenMP(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef MYCV_WITH_TBB
Status MulCpuTbb(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef MYCV_WITH_SIMD_GENERIC
Status MulCpuSimdGeneric(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef MYCV_ARCH_AVX2
Status MulCpuAvx2(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif

void RegisterMulKernels() {
  RegisterKernel(OpKind::kMul, BackendType::kNativeCpu, CpuKernelLevel::kNaive,
                 &MulCpuNaive);
#ifdef MYCV_WITH_OPENMP
  RegisterKernel(OpKind::kMul, BackendType::kNativeCpu, CpuKernelLevel::kOpenMP,
                 &MulCpuOpenMP);
#endif
#ifdef MYCV_WITH_TBB
  RegisterKernel(OpKind::kMul, BackendType::kNativeCpu, CpuKernelLevel::kTbb,
                 &MulCpuTbb);
#endif
#ifdef MYCV_WITH_SIMD_GENERIC
  RegisterKernel(OpKind::kMul, BackendType::kNativeCpu, CpuKernelLevel::kSimdGeneric,
                 &MulCpuSimdGeneric);
#endif
#ifdef MYCV_ARCH_AVX2
  RegisterKernel(OpKind::kMul, BackendType::kNativeCpu, CpuKernelLevel::kSimdAvx2,
                 &MulCpuAvx2);
#endif
}

}  // namespace ops_internal

Result<Tensor> Mul(const OpContext& context, const Tensor& input,
                   const Tensor& other) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateBinaryInputs(input, &other));
  auto kernel = ops_internal::FindKernel(OpKind::kMul, input.backend(),
                                         context.cpu_level);
  if (!kernel.ok()) return kernel.status();
  auto output = ops_internal::AllocateOutput(context, input.shape(), input.dtype());
  if (!output.ok()) return output.status();
  MYCV_RETURN_IF_ERROR(kernel.value()(context, input, &other, nullptr,
                                      &output.value()));
  return output;
}

}  // namespace mycv
