// fourier 算子的入口：负责注册、参数校验、内核查找与输出分配，然后分派
// 给某个后端实现。所有后端共享 fourier_internal.h 中的参数契约。
//
// 正向流程：校验 dtype/rank/layout -> ParseImage -> FindKernel -> 分配
// Re/Im -> 以实部作为 input 调用后端（虚部写入 params.imag）。
// 逆向流程：额外校验 real/imag 的 shape/dtype/backend/device 一致 ->
// 分配输出 -> 把 imag 作为 other 传入，后端写回时乘 1/(H*W)。
//
// 本文件不做任何逐元素计算，只做编排，因此各后端看到的参数完全一致。

#include "mycv/ops/fourier.h"

#include <utility>

#include "framework/image_utils.h"
#include "ops/common/op_common.h"
#include "fourier_internal.h"

namespace mycv {
namespace ops_internal {

// 把编译进本二进制的实现登记到算子注册表：CPU 的 naive/OpenMP/TBB 三个
// 级别与可选的 CUDA 后端。GPU 实现按 CpuKernelLevel::kNaive 这一槽位登记，
// 实际分派只看 BackendType；未编译的级别在注册阶段即缺席。
void RegisterFourierKernels() {
  const BackendType cpu = BackendType::kNativeCpu;
  RegisterKernel(OpKind::kFourierTransform, cpu, CpuKernelLevel::kNaive,
                 &FourierCpuNaive);
#ifdef MYCV_WITH_OPENMP
  RegisterKernel(OpKind::kFourierTransform, cpu, CpuKernelLevel::kOpenMP,
                 &FourierCpuOpenMP);
#endif
#ifdef MYCV_WITH_TBB
  RegisterKernel(OpKind::kFourierTransform, cpu, CpuKernelLevel::kTbb,
                 &FourierCpuTbb);
#endif
#ifdef MYCV_WITH_CUDA
  RegisterKernel(OpKind::kFourierTransform, BackendType::kCUDA,
                 CpuKernelLevel::kNaive, &FourierCuda);
#endif
  // 不注册 OpenCV 版本：查找该后端时返回 kNotImplemented，而不是悄悄
  // 用主机内核去跑设备内存。
}

}  // namespace ops_internal

namespace {

// 频谱必须能表示有符号值，因此只接受 f32/f64；u8 会丢失负数与小数，
// 在这里直接返回 kUnsupported。
Status ValidateSpectrumDType(DataType dtype) {
  if (dtype != DataType::kFloat32 && dtype != DataType::kFloat64) {
    return Status::Unsupported(
        "FourierTransform supports f32 and f64 only: the spectrum carries "
        "signed values and cannot be represented in u8");
  }
  return Status::Ok();
}

}  // namespace

// 正向二维 DFT。输入为 H x W 的实值图像（可为非连续视图），输出为与输入
// 同 shape/dtype 的 Re(X) 与 Im(X) 两个连续张量。
//
// 步骤：校验调用与 dtype -> 解析布局 -> 查找后端 -> 分配两个输出 ->
// 调用后端（input 读实部，params.imag 写虚部）-> 组装 ComplexImage。
// 复杂度由后端决定：radix-2 为 O(H*W*log)，非 2 的幂回退为 O(H*W*(H+W))。
Result<ComplexImage> FourierTransform(const OpContext& context,
                                      const Tensor& input,
                                      ImageLayout layout) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  MYCV_RETURN_IF_ERROR(ValidateSpectrumDType(input.dtype()));
  auto image = ops_internal::ParseImage(input.shape(), layout);
  if (!image.ok()) return image.status();
  auto kernel = ops_internal::FindKernel(OpKind::kFourierTransform,
                                         input.backend(), context.cpu_level);
  if (!kernel.ok()) return kernel.status();

  auto real = ops_internal::AllocateOutput(context, input.shape(), input.dtype());
  if (!real.ok()) return real.status();
  auto imag = ops_internal::AllocateOutput(context, input.shape(), input.dtype());
  if (!imag.ok()) return imag.status();
  const ops_internal::FourierParams params{layout, false, &imag.value()};
  MYCV_RETURN_IF_ERROR(
      kernel.value()(context, input, nullptr, &params, &real.value()));

  ComplexImage result;
  result.real = std::move(real.value());
  result.imag = std::move(imag.value());
  return result;
}

// 逆向二维 DFT：从 Re(X)/Im(X) 重建图像，返回 x 的实部。
// 名字里的“实部”是约定：输入按实值图像处理，后端只写回实部并在写回时
// 统一乘 1/(H*W)，保证“正向 -> 逆向”是单位增益。
//
// 步骤：校验调用、dtype、real/imag 的 shape/dtype/backend/device 一致 ->
// 解析布局 -> 查找后端 -> 分配输出 -> 以 real 为 input、imag 为 other
// 调用后端（params.imag 为 nullptr，不写频谱虚部）。
Result<Tensor> InverseFourierTransform(const OpContext& context,
                                       const Tensor& real, const Tensor& imag,
                                       ImageLayout layout) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, real));
  if (!imag.defined()) {
    return Status::InvalidArgument("imag tensor is undefined");
  }
  if (real.shape() != imag.shape() || real.dtype() != imag.dtype() ||
      real.backend() != imag.backend() || real.device() != imag.device()) {
    return Status::InvalidArgument(
        "real and imag must share shape, dtype, backend and device");
  }
  MYCV_RETURN_IF_ERROR(ValidateSpectrumDType(real.dtype()));
  auto image = ops_internal::ParseImage(real.shape(), layout);
  if (!image.ok()) return image.status();
  auto kernel = ops_internal::FindKernel(OpKind::kFourierTransform,
                                         real.backend(), context.cpu_level);
  if (!kernel.ok()) return kernel.status();

  auto output =
      ops_internal::AllocateOutput(context, real.shape(), real.dtype());
  if (!output.ok()) return output.status();
  const ops_internal::FourierParams params{layout, true, nullptr};
  MYCV_RETURN_IF_ERROR(
      kernel.value()(context, real, &imag, &params, &output.value()));
  return output;
}

}  // namespace mycv
