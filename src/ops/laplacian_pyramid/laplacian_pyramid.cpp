// laplacian_pyramid 的公共入口与版本注册。
//
// 本文件负责两件事：
//   1. BuildLaplacianPyramid：参数校验、查注册表、逐级调用 stage 组合；
//   2. RegisterLaplacianPyramidKernels：登记各 (backend, level) 组合。
// 入口本身不含像素级计算，数值工作由 host/CUDA stage 完成。
//
// 逐级语义（纯文本）：
//   G_{i+1} = ResizeNearest(GaussianBlur(G_i))   尺寸 ceil(H/2) x ceil(W/2)
//   L_i     = G_i - upsample(G_{i+1})            与 G_i 同尺寸
//   最顶层  L_n = G_n
// 重建：G_i = L_i + upsample(G_{i+1})。

#include "mycv/ops/laplacian_pyramid.h"

#include <cmath>
#include <utility>

#include "framework/image_utils.h"
#include "ops/common/op_common.h"
#include "laplacian_pyramid_internal.h"

namespace mycv {
namespace ops_internal {

// 登记 CPU naive（以及构建启用的 OpenMP/TBB）、CUDA 与 OpenCV 版本。
// CPU 各级别共用同一个 host 组合 stage，级别由 OpContext 原样透传给子算子。
// 复杂度 O(1)，仅登记函数指针。
void RegisterLaplacianPyramidKernels() {
  const BackendType cpu = BackendType::kNativeCpu;
  RegisterKernel(OpKind::kLaplacianPyramid, cpu, CpuKernelLevel::kNaive,
                 &LaplacianPyramidHostStage);
#ifdef MYCV_WITH_OPENMP
  RegisterKernel(OpKind::kLaplacianPyramid, cpu, CpuKernelLevel::kOpenMP,
                 &LaplacianPyramidHostStage);
#endif
#ifdef MYCV_WITH_TBB
  RegisterKernel(OpKind::kLaplacianPyramid, cpu, CpuKernelLevel::kTbb,
                 &LaplacianPyramidHostStage);
#endif
#ifdef MYCV_WITH_CUDA
  RegisterKernel(OpKind::kLaplacianPyramid, BackendType::kCUDA,
                 CpuKernelLevel::kNaive, &LaplacianPyramidCudaStage);
#endif
#ifdef MYCV_WITH_OPENCV
  // OpenCV 张量位于主机内存，因此 host 组合（含逐元素相减）对该后端同样成立。
  RegisterKernel(OpKind::kLaplacianPyramid, BackendType::kOpenCV,
                 CpuKernelLevel::kNaive, &LaplacianPyramidHostStage);
#endif
}

}  // namespace ops_internal

// 构建拉普拉斯金字塔，返回 levels + 1 个张量：
//   levels[0..levels-1]：带通 L_i，由细到粗，与 G_i 同尺寸；
//   levels[levels]     ：最粗高斯层 G_levels，作为低频残差（L_n = G_n）。
//
// 校验：levels >= 1；kernel_size 为 >= 3 的奇数；sigma 有限且 > 0；
//       dtype 仅 f32/f64（带通是有符号差值，u8 无法表示）；rank 2/3/4。
// 步骤：每级分配 L_i，令 params.next_gaussian 指向局部槽，调用查表内核；
//       内核写 L_i 与 G_{i+1}，随后 H/W 按 (h+1)/2 收缩（ceil 处理奇数）。
// 复杂度：每级 O(N_i * kernel_size)，合计 O(N * kernel_size)，N 为原图元素数。
Result<LaplacianPyramid> BuildLaplacianPyramid(const OpContext& context,
                                               const Tensor& input, int levels,
                                               int kernel_size, double sigma,
                                               ImageLayout layout) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
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
                                         input.backend(), context.cpu_level);
  if (!kernel.ok()) return kernel.status();

  LaplacianPyramid pyramid;
  pyramid.levels.reserve(static_cast<size_t>(levels) + 1);
  ops_internal::ImageInfo current = image.value();
  Tensor source = input;
  for (int level = 0; level < levels; ++level) {
    // 每级降采样的输入必须至少 2x2；最粗层本身可以小到 1x1。
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
    MYCV_RETURN_IF_ERROR(kernel.value()(context, source, nullptr, &params,
                                        &laplacian.value()));
    if (!next.defined()) {
      return Status::Internal(
          "LaplacianPyramid stage did not produce the next Gaussian level");
    }
    pyramid.levels.push_back(std::move(laplacian.value()));
    source = std::move(next);
    current.h = (current.h + 1) / 2;
    current.w = (current.w + 1) / 2;
  }
  // 最粗高斯层同时充当最后一个拉普拉斯残差（L_n = G_n）。
  pyramid.levels.push_back(std::move(source));
  return pyramid;
}

}  // namespace mycv
