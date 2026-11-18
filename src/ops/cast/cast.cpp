#include "mycv/ops/cast.h"

#include "ops/op_common.h"

namespace mycv {
namespace ops_internal {

Status CastCpuNaive(const OpContext& context, const Tensor& input,
                    const Tensor* other, const void* params, Tensor* output);
#ifdef MYCV_WITH_SIMD_GENERIC
Status CastCpuSimdGeneric(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef MYCV_ARCH_AVX2
Status CastCpuAvx2(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif

void RegisterCastKernels() {
  RegisterKernel(OpKind::kCast, BackendType::kNativeCpu, CpuKernelLevel::kNaive,
                 &CastCpuNaive);
#ifdef MYCV_WITH_SIMD_GENERIC
  RegisterKernel(OpKind::kCast, BackendType::kNativeCpu,
                 CpuKernelLevel::kSimdGeneric, &CastCpuSimdGeneric);
#endif
#ifdef MYCV_ARCH_AVX2
  RegisterKernel(OpKind::kCast, BackendType::kNativeCpu,
                 CpuKernelLevel::kSimdAvx2, &CastCpuAvx2);
#endif
}

}  // namespace ops_internal

Result<Tensor> Cast(const OpContext& context, const Tensor& input,
                    DataType dst_dtype) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  if (DTypeSize(dst_dtype) == 0) {
    return Status::Unsupported("unsupported destination dtype");
  }
  auto kernel = ops_internal::FindKernel(OpKind::kCast, input.backend(),
                                         context.cpu_level);
  if (!kernel.ok()) return kernel.status();
  auto output = ops_internal::AllocateOutput(context, input.shape(), dst_dtype);
  if (!output.ok()) return output.status();
  MYCV_RETURN_IF_ERROR(kernel.value()(context, input, nullptr, nullptr,
                                      &output.value()));
  return output;
}

}  // namespace mycv
