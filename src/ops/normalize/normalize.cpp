#include "mycv/ops/normalize.h"

#include <cmath>

#include "ops/op_common.h"
#include "ops/op_params.h"

namespace mycv {
namespace ops_internal {

Status NormalizeCpuNaive(const OpContext& context, const Tensor& input,
                         const Tensor* other, const void* params, Tensor* output);
#ifdef MYCV_WITH_OPENMP
Status NormalizeCpuOpenMP(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef MYCV_WITH_TBB
Status NormalizeCpuTbb(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef MYCV_WITH_SIMD_GENERIC
Status NormalizeCpuSimdGeneric(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef MYCV_ARCH_AVX2
Status NormalizeCpuAvx2(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif

void RegisterNormalizeKernels() {
  RegisterKernel(OpKind::kNormalize, BackendType::kNativeCpu,
                 CpuKernelLevel::kNaive, &NormalizeCpuNaive);
#ifdef MYCV_WITH_OPENMP
  RegisterKernel(OpKind::kNormalize, BackendType::kNativeCpu,
                 CpuKernelLevel::kOpenMP, &NormalizeCpuOpenMP);
#endif
#ifdef MYCV_WITH_TBB
  RegisterKernel(OpKind::kNormalize, BackendType::kNativeCpu,
                 CpuKernelLevel::kTbb, &NormalizeCpuTbb);
#endif
#ifdef MYCV_WITH_SIMD_GENERIC
  RegisterKernel(OpKind::kNormalize, BackendType::kNativeCpu,
                 CpuKernelLevel::kSimdGeneric, &NormalizeCpuSimdGeneric);
#endif
#ifdef MYCV_ARCH_AVX2
  RegisterKernel(OpKind::kNormalize, BackendType::kNativeCpu,
                 CpuKernelLevel::kSimdAvx2, &NormalizeCpuAvx2);
#endif
}

}  // namespace ops_internal

Result<Tensor> Normalize(const OpContext& context, const Tensor& input,
                         const std::vector<double>& mean,
                         const std::vector<double>& std, double epsilon) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  if (input.shape().Rank() == 0 || input.dtype() == DataType::kFloat16 ||
      !ops_internal::IsSupportedArithmeticDType(input.dtype())) {
    return Status::Unsupported("Normalize requires a non-scalar non-f16 tensor");
  }
  const int64_t channels = input.shape().Dim(input.shape().Rank() - 1);
  if (mean.size() != static_cast<size_t>(channels) ||
      std.size() != static_cast<size_t>(channels) || epsilon < 0.0) {
    return Status::InvalidArgument("mean/std must match the channel count");
  }
  for (size_t c = 0; c < std.size(); ++c) {
    if (std[c] < 0.0 || !std::isfinite(std[c] + epsilon) ||
        std[c] + epsilon == 0.0) {
      return Status::InvalidArgument("Normalize denominator must be positive and finite");
    }
  }

  auto kernel = ops_internal::FindKernel(OpKind::kNormalize, input.backend(),
                                         context.cpu_level);
  if (!kernel.ok()) return kernel.status();
  const DataType output_dtype = input.dtype() == DataType::kFloat64
                                    ? DataType::kFloat64
                                    : DataType::kFloat32;
  auto output = ops_internal::AllocateOutput(context, input.shape(), output_dtype);
  if (!output.ok()) return output.status();
  const ops_internal::NormalizeParams params{&mean, &std, epsilon};
  MYCV_RETURN_IF_ERROR(kernel.value()(context, input, nullptr, &params,
                                      &output.value()));
  return output;
}

}  // namespace mycv
