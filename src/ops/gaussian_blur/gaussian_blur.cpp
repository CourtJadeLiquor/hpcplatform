#include "mycv/ops/gaussian_blur.h"

#include <cmath>

#include "framework/image_utils.h"
#include "ops/common/op_common.h"
#include "gaussian_blur_internal.h"

// Public entry point and backend registration for gaussian_blur.
//
// GaussianBlur() only validates parameters/shape, selects a kernel by
// (backend, cpu_level), and allocates output; backends perform the computation.
// The algorithm is separable Gaussian convolution: horizontal then vertical 1D
// passes, each with O(H*W*K) multiply-adds instead of O(H*W*K^2) for direct 2D
// convolution. The horizontal pass computes
// tmp[y,x] = sum_k in[y, x-r+k] * w[k] with replicate boundaries; the vertical
// pass applies the same 1D weighting to tmp.
namespace mycv {
namespace ops_internal {

// Register every gaussian_blur kernel compiled into this binary: all enabled
// CPU levels use BackendType::kNativeCpu, while optional CUDA/OpenCV backends
// each have their own backend type. Optional backends are registered in the
// CpuKernelLevel::kNaive slot because they have no CPU levels.
void RegisterGaussianBlurKernels() {
  const BackendType cpu = BackendType::kNativeCpu;
  RegisterKernel(OpKind::kGaussianBlur, cpu, CpuKernelLevel::kNaive,
                 &GaussianBlurCpuNaive);
#ifdef MYCV_WITH_OPENMP
  RegisterKernel(OpKind::kGaussianBlur, cpu, CpuKernelLevel::kOpenMP,
                 &GaussianBlurCpuOpenMP);
#endif
#ifdef MYCV_WITH_TBB
  RegisterKernel(OpKind::kGaussianBlur, cpu, CpuKernelLevel::kTbb,
                 &GaussianBlurCpuTbb);
#endif
#ifdef MYCV_WITH_SIMD_GENERIC
  RegisterKernel(OpKind::kGaussianBlur, cpu, CpuKernelLevel::kSimdGeneric,
                 &GaussianBlurCpuSimd);
#endif
#ifdef MYCV_ARCH_AVX2
  RegisterKernel(OpKind::kGaussianBlur, cpu, CpuKernelLevel::kSimdAvx2,
                 &GaussianBlurCpuAvx2);
#endif
#ifdef MYCV_WITH_CUDA
  RegisterKernel(OpKind::kGaussianBlur, BackendType::kCUDA,
                 CpuKernelLevel::kNaive, &GaussianBlurCuda);
#endif
#ifdef MYCV_WITH_OPENCV
  RegisterKernel(OpKind::kGaussianBlur, BackendType::kOpenCV,
                 CpuKernelLevel::kNaive, &GaussianBlurOpenCv);
#endif
}

}  // namespace ops_internal

// Public operator entry point. Steps:
// 1) Validate the invocation; require odd kernel_size >= 3 and finite sigma > 0;
// 2) Parse the shape according to layout and require u8/f32/f64 dtype;
// 3) Find a registered kernel by (input.backend(), context.cpu_level);
// 4) Allocate output with the same shape and dtype, then invoke the kernel with
//    params. The kernel performs two 1D convolution passes, each O(H*W*K).
Result<Tensor> GaussianBlur(const OpContext& context, const Tensor& input,
                            int kernel_size, double sigma, ImageLayout layout) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  if (kernel_size < 3 || kernel_size % 2 == 0) {
    return Status::InvalidArgument(
        "GaussianBlur kernel_size must be an odd integer >= 3");
  }
  if (!(sigma > 0.0) || !std::isfinite(sigma)) {
    return Status::InvalidArgument("GaussianBlur sigma must be finite and > 0");
  }
  auto image = ops_internal::ParseImage(input.shape(), layout);
  if (!image.ok()) return image.status();
  if (!ops_internal::IsSupportedImageDType(input.dtype())) {
    return Status::Unsupported("GaussianBlur supports u8, f32 and f64 images");
  }
  auto kernel = ops_internal::FindKernel(OpKind::kGaussianBlur, input.backend(),
                                         context.cpu_level);
  if (!kernel.ok()) return kernel.status();
  auto output =
      ops_internal::AllocateOutput(context, input.shape(), input.dtype());
  if (!output.ok()) return output.status();
  const ops_internal::GaussianBlurParams params{kernel_size, sigma, layout};
  MYCV_RETURN_IF_ERROR(
      kernel.value()(context, input, nullptr, &params, &output.value()));
  return output;
}

}  // namespace mycv
