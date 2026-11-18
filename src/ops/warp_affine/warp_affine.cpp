#include "mycv/ops/warp_affine.h"

#include <cmath>

#include "ops/image_utils.h"
#include "ops/op_common.h"
#include "ops/op_params.h"

namespace mycv {
namespace ops_internal {

Status WarpAffineCpuNaive(const OpContext& context, const Tensor& input,
                          const Tensor* other, const void* params, Tensor* output);

void RegisterWarpAffineKernels() {
  RegisterKernel(OpKind::kWarpAffine, BackendType::kNativeCpu,
                 CpuKernelLevel::kNaive, &WarpAffineCpuNaive);
}

}  // namespace ops_internal

Result<Tensor> WarpAffine(const OpContext& context, const Tensor& input,
                          const std::array<double, 6>& matrix,
                          int64_t output_height, int64_t output_width,
                          InterpolationMode interpolation, double border_value,
                          ImageLayout layout) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  auto image = ops_internal::ParseImage(input.shape(), layout);
  if (!image.ok()) return image.status();
  if (!ops_internal::IsSupportedImageDType(input.dtype())) {
    return Status::Unsupported("WarpAffine supports u8, f32 and f64 images");
  }
  if (output_height <= 0 || output_width <= 0) {
    return Status::InvalidArgument("WarpAffine output dimensions must be positive");
  }
  const double determinant = matrix[0] * matrix[4] - matrix[1] * matrix[3];
  if (std::fabs(determinant) < 1e-12) {
    return Status::InvalidArgument("WarpAffine matrix must be invertible");
  }
  auto kernel = ops_internal::FindKernel(OpKind::kWarpAffine, input.backend(),
                                         context.cpu_level);
  if (!kernel.ok()) return kernel.status();
  const Shape output_shape =
      ops_internal::MakeImageShape(image.value(), output_height, output_width);
  auto output = ops_internal::AllocateOutput(context, output_shape, input.dtype());
  if (!output.ok()) return output.status();
  const ops_internal::WarpAffineParams params{&matrix, output_height, output_width,
                                              interpolation, border_value, layout};
  MYCV_RETURN_IF_ERROR(kernel.value()(context, input, nullptr, &params,
                                      &output.value()));
  return output;
}

}  // namespace mycv
