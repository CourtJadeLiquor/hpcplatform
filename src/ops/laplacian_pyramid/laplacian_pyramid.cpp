// Public entry point and version registration for laplacian_pyramid.
//
// This file handles two tasks:
//   1. BuildLaplacianPyramid: validate parameters, query the registry, and invoke
//      the composed stage for each level;
//   2. RegisterLaplacianPyramidKernels: register each (backend, level) combination.
// The entry point contains no pixel-level computation; host/CUDA stages perform
// the numerical work.
//
// Per-level semantics (plain text):
//   G_{i+1} = ResizeNearest(GaussianBlur(G_i))   size ceil(H/2) x ceil(W/2)
//   L_i     = G_i - upsample(G_{i+1})            same size as G_i
//   Top level L_n = G_n
// Reconstruction: G_i = L_i + upsample(G_{i+1}).

#include "hpcplatform/ops/laplacian_pyramid.h"

#include <cmath>
#include <utility>

#include "framework/image_utils.h"
#include "ops/common/op_common.h"
#include "laplacian_pyramid_internal.h"

namespace hpcplatform {
namespace ops_internal {

// Register CPU naive (plus OpenMP/TBB when enabled in the build), CUDA, and
// OpenCV versions. All CPU levels share the same composed host stage, and
// OpContext forwards the level unchanged to the suboperators.
// Complexity: O(1); only registers function pointers.
void RegisterLaplacianPyramidKernels() {
  const BackendType cpu = BackendType::kNativeCpu;
  RegisterKernel(OpKind::kLaplacianPyramid, cpu, CpuKernelLevel::kNaive,
                 &LaplacianPyramidHostStage);
#ifdef hpcplatform_WITH_OPENMP
  RegisterKernel(OpKind::kLaplacianPyramid, cpu, CpuKernelLevel::kOpenMP,
                 &LaplacianPyramidHostStage);
#endif
#ifdef hpcplatform_WITH_TBB
  RegisterKernel(OpKind::kLaplacianPyramid, cpu, CpuKernelLevel::kTbb,
                 &LaplacianPyramidHostStage);
#endif
#ifdef hpcplatform_WITH_CUDA
  RegisterKernel(OpKind::kLaplacianPyramid, BackendType::kCUDA,
                 CpuKernelLevel::kNaive, &LaplacianPyramidCudaStage);
#endif
#ifdef hpcplatform_WITH_OPENCV
  // OpenCV tensors reside in host memory, so the host composition (including
  // element-wise subtraction) also works for this backend.
  RegisterKernel(OpKind::kLaplacianPyramid, BackendType::kOpenCV,
                 CpuKernelLevel::kNaive, &LaplacianPyramidHostStage);
#endif
}

}  // namespace ops_internal

// Build a Laplacian pyramid and return levels + 1 tensors:
//   levels[0..levels-1]: band-pass L_i from fine to coarse, same size as G_i;
//   levels[levels]:      coarsest Gaussian G_levels as the low-frequency residual
//                        (L_n = G_n).
//
// Validation: levels >= 1; kernel_size is odd and >= 3; sigma is finite and > 0;
//             dtype is f32/f64 only (signed band differences cannot fit in u8);
//             rank is 2/3/4.
// Steps: allocate L_i for each level, point params.next_gaussian to a local slot,
//        and invoke the registered kernel; the kernel writes L_i and G_{i+1},
//        then H/W shrink to (h+1)/2 (ceil handles odd dimensions).
// Complexity: O(N_i * kernel_size) per level and O(N * kernel_size) overall,
// where N is the original image element count.
Result<LaplacianPyramid> BuildLaplacianPyramid(const OpContext& context,
                                               const Tensor& input, int levels,
                                               int kernel_size, double sigma,
                                               ImageLayout layout) {
  hpcplatform_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  if (levels < 1) {
    return Status::InvalidArgument("LaplacianPyramid levels must be >= 1");
  }
  if (kernel_size < 3 || kernel_size % 2 == 0) {
    return Status::InvalidArgument(
        "LaplacianPyramid kernel_size must be an odd integer >= 3");
  }
  if (!(sigma > 0.0) || !std::isfinite(sigma)) {
    return Status::InvalidArgument(
        "LaplacianPyramid sigma must be finite and > 0");
  }
  if (input.dtype() != DataType::kFloat32 &&
      input.dtype() != DataType::kFloat64) {
    return Status::Unsupported(
        "LaplacianPyramid supports f32 and f64 only: the bands are signed "
        "differences and cannot be represented in u8");
  }
  auto image = ops_internal::ParseImage(input.shape(), layout);
  if (!image.ok()) return image.status();
  auto kernel = ops_internal::FindKernel(OpKind::kLaplacianPyramid,
                                         input.backend(), context.cpu_level());
  if (!kernel.ok()) return kernel.status();

  LaplacianPyramid pyramid;
  pyramid.levels.reserve(static_cast<size_t>(levels) + 1);
  ops_internal::ImageInfo current = image.value();
  Tensor source = input;
  for (int level = 0; level < levels; ++level) {
    // Each downsampling input must be at least 2x2; the coarsest level itself may
    // be as small as 1x1.
    if (current.h < 2 || current.w < 2) {
      return Status::InvalidArgument(
          "LaplacianPyramid levels exceed what the input resolution allows");
    }
    auto laplacian = ops_internal::AllocateOutput(
        context, ops_internal::MakeImageShape(current, current.h, current.w),
        input.dtype());
    if (!laplacian.ok()) return laplacian.status();
    Tensor next;
    const ops_internal::LaplacianPyramidParams params{kernel_size, sigma,
                                                      layout, &next};
    hpcplatform_RETURN_IF_ERROR(
        kernel.value()(context, source, nullptr, &params, &laplacian.value()));
    if (!next.defined()) {
      return Status::Internal(
          "LaplacianPyramid stage did not produce the next Gaussian level");
    }
    pyramid.levels.push_back(std::move(laplacian.value()));
    source = std::move(next);
    current.h = (current.h + 1) / 2;
    current.w = (current.w + 1) / 2;
  }
  // The coarsest Gaussian level also serves as the final Laplacian residual
  // (L_n = G_n).
  pyramid.levels.push_back(std::move(source));
  return pyramid;
}

}  // namespace hpcplatform
