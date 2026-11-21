#include "mycv/ops/gaussian_blur.h"

#include <cmath>

#include "framework/image_utils.h"
#include "ops/common/op_common.h"
#include "gaussian_blur_internal.h"

// gaussian_blur 的对外入口与后端注册。
//
// GaussianBlur() 只做参数/形状校验、按 (backend, cpu_level) 选内核、
// 分配输出，真正计算交给后端。算法是可分离高斯卷积：先水平后垂直
// 两趟一维卷积，每趟 O(H*W*K) 次乘加，比二维直接卷积的 O(H*W*K^2)
// 省算；水平 pass 为 tmp[y,x] = sum_k in[y, x-r+k] * w[k]（边界
// replicate），垂直 pass 对 tmp 做同样的一维加权。
namespace mycv {
namespace ops_internal {

// 注册所有编译进本二进制的 gaussian_blur 内核：所有已启用的 CPU 层
// 走 BackendType::kNativeCpu，可选后端 CUDA/OpenCV 各占一个后端类型；
// 可选后端统一登记在 CpuKernelLevel::kNaive 槽位，因为它们没有 CPU 分层。
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

// 公开算子入口。步骤：
// 1) 校验 invocation；要求 kernel_size 为 >= 3 的奇数、sigma 有限且 > 0；
// 2) 按 layout 解析形状，确认 dtype 为 u8/f32/f64；
// 3) 按 (input.backend(), context.cpu_level) 找到已注册内核；
// 4) 分配同形状同 dtype 的输出，用 params 调用内核；
//    内核内部完成两趟一维卷积，复杂度 O(H*W*K) 每 pass。
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
