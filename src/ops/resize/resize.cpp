// Entry portion of the resize operator pipeline: kernel registration and the
// public Resize() entry point. RegisterResizeKernels only registers function
// pointers; Resize() handles backend-independent validation, shape inference,
// output allocation, and dispatch. See the backend implementation files for
// the actual resize kernels (CPU shares resize_cpu.h; CUDA/OpenCV are separate).
// This file contains no interpolation formulas.

#include "mycv/ops/resize.h"

#include "framework/image_utils.h"
#include "ops/common/op_common.h"
#include "resize_internal.h"

namespace mycv {
namespace ops_internal {

// Kernel registration: naive is mandatory for CPU; OpenMP/TBB are registered
// at the corresponding kNativeCpu levels. CUDA/OpenCV are independent backends
// and provide only kNaive (non-CPU backends accept no other levels; see
// ValidateInvocation). O(1); compile-time conditions select the entries.
void RegisterResizeKernels() {
  const BackendType cpu = BackendType::kNativeCpu;
  RegisterKernel(OpKind::kResize, cpu, CpuKernelLevel::kNaive, &ResizeCpuNaive);
#ifdef MYCV_WITH_OPENMP
  RegisterKernel(OpKind::kResize, cpu, CpuKernelLevel::kOpenMP,
                 &ResizeCpuOpenMP);
#endif
#ifdef MYCV_WITH_TBB
  RegisterKernel(OpKind::kResize, cpu, CpuKernelLevel::kTbb, &ResizeCpuTbb);
#endif
#ifdef MYCV_WITH_CUDA
  RegisterKernel(OpKind::kResize, BackendType::kCUDA, CpuKernelLevel::kNaive,
                 &ResizeCuda);
#endif
#ifdef MYCV_WITH_OPENCV
  RegisterKernel(OpKind::kResize, BackendType::kOpenCV, CpuKernelLevel::kNaive,
                 &ResizeOpenCv);
#endif
}

}  // namespace ops_internal

// Public resize entry point; returns a newly allocated output tensor.
// Inputs: context, input image tensor, target height/width, interpolation, layout.
// Steps: 1) validate the backend/device of context and the input tensor;
//        2) parse the rank-2/3/4 shape into ImageInfo according to layout;
//        3) accept only u8/f32/f64 (otherwise Unsupported) and positive dimensions;
//        4) find a registered kernel by input.backend() and context.cpu_level;
//        5) infer and allocate the output shape while preserving rank/layout;
//        6) pack ResizeParams and invoke the kernel.
// O(1) for shape handling and dispatch only; all pixel work occurs in the kernel.
Result<Tensor> Resize(const OpContext& context, const Tensor& input,
                      int64_t output_height, int64_t output_width,
                      InterpolationMode interpolation, ImageLayout layout) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  auto image = ops_internal::ParseImage(input.shape(), layout);
  if (!image.ok()) return image.status();
  // dtype allowlist shared by all three backends.
  if (!ops_internal::IsSupportedImageDType(input.dtype())) {
    return Status::Unsupported("Resize supports u8, f32 and f64 images");
  }
  if (output_height <= 0 || output_width <= 0) {
    return Status::InvalidArgument("Resize output dimensions must be positive");
  }
  auto kernel = ops_internal::FindKernel(OpKind::kResize, input.backend(),
                                         context.cpu_level);
  if (!kernel.ok()) return kernel.status();
  // Preserve the input rank and layout, replacing only H/W.
  const Shape output_shape =
      ops_internal::MakeImageShape(image.value(), output_height, output_width);
  auto output =
      ops_internal::AllocateOutput(context, output_shape, input.dtype());
  if (!output.ok()) return output.status();
  const ops_internal::ResizeParams params{output_height, output_width,
                                          interpolation, layout};
  // Single-input operator: pass nullptr as the second tensor.
  MYCV_RETURN_IF_ERROR(
      kernel.value()(context, input, nullptr, &params, &output.value()));
  return output;
}

}  // namespace mycv
