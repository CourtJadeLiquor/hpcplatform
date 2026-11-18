#include "mycv/ops/crop.h"

#include "ops/image_utils.h"
#include "ops/op_common.h"
#include "ops/op_params.h"

namespace mycv {
namespace ops_internal {

Status CropCpuNaive(const OpContext& context, const Tensor& input,
                    const Tensor* other, const void* params, Tensor* output);

void RegisterCropKernels() {
  RegisterKernel(OpKind::kCrop, BackendType::kNativeCpu, CpuKernelLevel::kNaive,
                 &CropCpuNaive);
}

}  // namespace ops_internal

Result<Tensor> Crop(const OpContext& context, const Tensor& input, int64_t top,
                    int64_t left, int64_t height, int64_t width,
                    ImageLayout layout) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  auto image = ops_internal::ParseImage(input.shape(), layout);
  if (!image.ok()) return image.status();
  if (!ops_internal::IsSupportedImageDType(input.dtype())) {
    return Status::Unsupported("Crop supports u8, f32 and f64 images");
  }
  if (top < 0 || left < 0 || height <= 0 || width <= 0 ||
      top + height > image.value().h || left + width > image.value().w) {
    return Status::InvalidArgument("Crop ROI is outside the input image");
  }
  auto kernel = ops_internal::FindKernel(OpKind::kCrop, input.backend(),
                                         context.cpu_level);
  if (!kernel.ok()) return kernel.status();
  const Shape output_shape = ops_internal::MakeImageShape(image.value(), height, width);
  auto output = ops_internal::AllocateOutput(context, output_shape, input.dtype());
  if (!output.ok()) return output.status();
  const ops_internal::CropParams params{top, left, height, width, layout};
  MYCV_RETURN_IF_ERROR(kernel.value()(context, input, nullptr, &params,
                                      &output.value()));
  return output;
}

}  // namespace mycv
