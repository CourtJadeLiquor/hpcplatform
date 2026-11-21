// wavelet 算子的入口：负责注册、参数校验、内核查找与输出分配，然后分派
// 给某个后端实现。所有后端共享 wavelet_internal.h 中的参数契约。
//
// 正向流程：校验 dtype/rank/layout 与 H、W 为偶数 -> ParseImage ->
// FindKernel -> 分配 LL/LH/HL/HH 四个 H/2 x W/2 张量 -> 以图像为 input
// 调用后端，LH/HL/HH 通过 extra_out 传入。
// 逆向流程：额外校验四带 shape/dtype/backend/device 一致 -> 分配
// H x W 输出 -> 以 LL 为 input、另外三带通过 extra_in 传入。
// 本文件只做编排，不做逐元素计算。

#include "mycv/ops/wavelet.h"

#include <utility>

#include "framework/image_utils.h"
#include "ops/common/op_common.h"
#include "wavelet_internal.h"

namespace mycv {
namespace ops_internal {

// 把编译进本二进制的实现登记到算子注册表：CPU 的 naive/OpenMP/TBB 三个
// 级别与可选的 CUDA 后端。GPU 实现按 CpuKernelLevel::kNaive 槽位登记，
// 实际分派只看 BackendType；未编译的级别在注册阶段即缺席。
void RegisterWaveletKernels() {
  const BackendType cpu = BackendType::kNativeCpu;
  RegisterKernel(OpKind::kWaveletTransform, cpu, CpuKernelLevel::kNaive,
                 &WaveletCpuNaive);
#ifdef MYCV_WITH_OPENMP
  RegisterKernel(OpKind::kWaveletTransform, cpu, CpuKernelLevel::kOpenMP,
                 &WaveletCpuOpenMP);
#endif
#ifdef MYCV_WITH_TBB
  RegisterKernel(OpKind::kWaveletTransform, cpu, CpuKernelLevel::kTbb,
                 &WaveletCpuTbb);
#endif
#ifdef MYCV_WITH_CUDA
  RegisterKernel(OpKind::kWaveletTransform, BackendType::kCUDA,
                 CpuKernelLevel::kNaive, &WaveletCuda);
#endif
  // 不注册 OpenCV 版本：查找该后端时返回 kNotImplemented，而不是悄悄
  // 用主机内核去跑设备内存。
}

}  // namespace ops_internal

namespace {

// 细节子带带符号，只接受 f32/f64；u8 无法表示负系数，直接 kUnsupported。
Status ValidateBandDType(DataType dtype) {
  if (dtype != DataType::kFloat32 && dtype != DataType::kFloat64) {
    return Status::Unsupported(
        "WaveletTransform supports f32 and f64 only: detail bands carry "
        "signed values and cannot be represented in u8");
  }
  return Status::Ok();
}

// 单层周期性 DWT 会把两个方向都减半，因此 H、W 必须为偶数；否则子带
// 尺寸没有定义，在这里返回 kInvalidArgument。
Status ValidateEvenExtent(const ops_internal::ImageInfo& info) {
  if (info.h % 2 != 0 || info.w % 2 != 0) {
    return Status::InvalidArgument(
        "WaveletTransform requires even height and width: the periodic "
        "single-level DWT halves both extents");
  }
  return Status::Ok();
}

}  // namespace

// 正向单层二维 DWT：把 H x W 图像分解为四个 H/2 x W/2 子带。
//
// 步骤：校验调用/dtype/偶数尺寸 -> 解析布局 -> 查找后端 -> 分配四带 ->
// 调用后端（图像作为 input，LL 作为 output，其余三带经 extra_out）。
// 复杂度 O(H*W*taps)：taps 为滤波器长度（Haar 2、Db4 4）。
Result<WaveletBands> WaveletTransform(const OpContext& context,
                                      const Tensor& input, WaveletKind kind,
                                      ImageLayout layout) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, input));
  MYCV_RETURN_IF_ERROR(ValidateBandDType(input.dtype()));
  auto image = ops_internal::ParseImage(input.shape(), layout);
  if (!image.ok()) return image.status();
  MYCV_RETURN_IF_ERROR(ValidateEvenExtent(image.value()));
  auto kernel = ops_internal::FindKernel(OpKind::kWaveletTransform,
                                         input.backend(), context.cpu_level);
  if (!kernel.ok()) return kernel.status();

  const ops_internal::ImageInfo& info = image.value();
  const Shape band_shape =
      ops_internal::MakeImageShape(info, info.h / 2, info.w / 2);
  auto ll = ops_internal::AllocateOutput(context, band_shape, input.dtype());
  if (!ll.ok()) return ll.status();
  auto lh = ops_internal::AllocateOutput(context, band_shape, input.dtype());
  if (!lh.ok()) return lh.status();
  auto hl = ops_internal::AllocateOutput(context, band_shape, input.dtype());
  if (!hl.ok()) return hl.status();
  auto hh = ops_internal::AllocateOutput(context, band_shape, input.dtype());
  if (!hh.ok()) return hh.status();

  ops_internal::WaveletParams params;
  params.kind = kind;
  params.layout = layout;
  params.inverse = false;
  params.extra_out[0] = &lh.value();
  params.extra_out[1] = &hl.value();
  params.extra_out[2] = &hh.value();
  MYCV_RETURN_IF_ERROR(
      kernel.value()(context, input, nullptr, &params, &ll.value()));

  WaveletBands bands;
  bands.ll = std::move(ll.value());
  bands.lh = std::move(lh.value());
  bands.hl = std::move(hl.value());
  bands.hh = std::move(hh.value());
  return bands;
}

// 逆向单层二维 DWT：由 LL/LH/HL/HH 完美重建 H x W 图像。
//
// 步骤：校验调用、dtype、四个子带的 shape/dtype/backend/device 一致 ->
// 按子带形状解析布局（反向推出图像尺寸）-> 查找后端 -> 分配输出 ->
// 以 LL 为 input、其余三带经 extra_in 调用后端。
// 复杂度 O(H*W*taps)；`kind` 必须与分解时一致，否则重建不再是恒等。
Result<Tensor> InverseWaveletTransform(const OpContext& context,
                                       const WaveletBands& bands,
                                       WaveletKind kind, ImageLayout layout) {
  MYCV_RETURN_IF_ERROR(ops_internal::ValidateInvocation(context, bands.ll));
  const Tensor* all[4] = {&bands.ll, &bands.lh, &bands.hl, &bands.hh};
  for (const Tensor* band : all) {
    if (!band->defined()) {
      return Status::InvalidArgument(
          "InverseWaveletTransform requires all four bands");
    }
    if (band->shape() != bands.ll.shape() ||
        band->dtype() != bands.ll.dtype() ||
        band->backend() != bands.ll.backend() ||
        band->device() != bands.ll.device()) {
      return Status::InvalidArgument(
          "WaveletBands must share shape, dtype, backend and device");
    }
  }
  MYCV_RETURN_IF_ERROR(ValidateBandDType(bands.ll.dtype()));
  auto image = ops_internal::ParseImage(bands.ll.shape(), layout);
  if (!image.ok()) return image.status();
  const ops_internal::ImageInfo& band_info = image.value();
  auto kernel = ops_internal::FindKernel(OpKind::kWaveletTransform,
                                         bands.ll.backend(), context.cpu_level);
  if (!kernel.ok()) return kernel.status();

  const Shape image_shape =
      ops_internal::MakeImageShape(band_info, band_info.h * 2, band_info.w * 2);
  auto output =
      ops_internal::AllocateOutput(context, image_shape, bands.ll.dtype());
  if (!output.ok()) return output.status();

  ops_internal::WaveletParams params;
  params.kind = kind;
  params.layout = layout;
  params.inverse = true;
  params.extra_in[0] = &bands.lh;
  params.extra_in[1] = &bands.hl;
  params.extra_in[2] = &bands.hh;
  MYCV_RETURN_IF_ERROR(
      kernel.value()(context, bands.ll, nullptr, &params, &output.value()));
  return output;
}

}  // namespace mycv
