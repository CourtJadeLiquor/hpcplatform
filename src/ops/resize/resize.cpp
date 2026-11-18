#include "mycv/ops/resize.h"

#include "ops/image_utils.h"
#include "ops/op_common.h"
#include "ops/op_params.h"

namespace mycv {
namespace ops_internal {

Status ResizeCpuNaive(const OpContext& context, const Tensor& input,
                      const Tensor* other, const void* params, Tensor* output);
#ifdef MYCV_WITH_OPENMP
Status ResizeCpuOpenMP(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif
#ifdef MYCV_WITH_TBB
Status ResizeCpuTbb(const OpContext&, const Tensor&, const Tensor*, const void*, Tensor*);
#endif

void RegisterResizeKernels() {
  RegisterKernel(OpKind::kResize, BackendType::kNativeCpu,
                 CpuKernelLevel::kNaive, &ResizeCpuNaive);
#ifdef MYCV_WITH_OPENMP
  RegisterKernel(OpKind::kResize, BackendType::kNativeCpu,
                 CpuKernelLevel::kOpenMP, &ResizeCpuOpenMP);
#endif
#ifdef MYCV_WITH_TBB
  RegisterKernel(OpKind::kResize, BackendType::kNativeCpu,
                 CpuKernelLevel::kTbb, &ResizeCpuTbb);
#endif
}

}  // namespace ops_internal

Result<Tensor> Resize(const OpContext& context, const Tensor& input,
                      int64_t output_height, int64_t output_width,
                      InterpolationMode interpolation, ImageLayout layout) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  auto image = ops_internal::ParseImage(input.shape(), layout);
  if (!image.ok()) return image.status();
  if (!ops_internal::IsSupportedImageDType(input.dtype())) {
    return Status::Unsupported("Resize supports u8, f32 and f64 images");
  }
  if (output_height <= 0 || output_width <= 0) {
    return Status::InvalidArgument("Resize output dimensions must be positive");
  }
  auto kernel = ops_internal::FindKernel(OpKind::kResize, input.backend(),
                                         context.cpu_level);
  if (!kernel.ok()) return kernel.status();
  const Shape output_shape =
      ops_internal::MakeImageShape(image.value(), output_height, output_width);
  auto output = ops_internal::AllocateOutput(context, output_shape, input.dtype());
  if (!output.ok()) return output.status();
  const ops_internal::ResizeParams params{output_height, output_width,
                                          interpolation, layout};
  MYCV_RETURN_IF_ERROR(kernel.value()(context, input, nullptr, &params,
                                      &output.value()));
  return output;
}

}  // namespace mycv
