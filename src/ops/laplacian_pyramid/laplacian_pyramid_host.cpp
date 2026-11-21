// laplacian_pyramid 的主机侧组合子算子（native_cpu 与 OpenCV 后端）。
//
// 一个 stage 不写新的像素级数学，而是复用两个已注册子算子：
//   1. GaussianBlur：对当前层 G_i 做可分离高斯平滑；
//   2. Resize：先 nearest 下采样到 ceil(H/2) x ceil(W/2) 得到 G_{i+1}，
//      再 bilinear 上采样回 G_i 尺寸；
//   3. 主机逐元素相减：L_i = G_i - upsample(G_{i+1})。
// 最顶层在循环之外由入口直接把 G_n 作为残差，因此 L_n = G_n；
// 重建关系为 G_i = L_i + upsample(G_{i+1})。
//
// 后端与级别不单独传参：OpContext 原样透传给两个子算子，它们分别按
// input.backend() 与 context.cpu_level 查表。因此组合路径调用的就是
// 单算子公开入口的同一套内核与参数，语义（含边界/采样规则）完全一致。

#include <cstdint>
#include <utility>

#include "framework/image_utils.h"
#include "mycv/ops/gaussian_blur.h"
#include "mycv/ops/resize.h"
#include "laplacian_pyramid_internal.h"

namespace mycv {
namespace ops_internal {
namespace {

// 从主机内存读取一个带通样本并提升为 double；仅支持 f32/f64
// （入口已拒绝其他 dtype）。复杂度 O(1)。
inline double LoadBandValue(const char* source, DataType dtype) {
  if (dtype == DataType::kFloat32) {
    return *reinterpret_cast<const float*>(source);
  }
  return *reinterpret_cast<const double*>(source);
}

// 把 double 结果写回带通张量；f32 时窄化为 float，其余按 double。
// 输入为字节指针、dtype 与数值，无返回值；复杂度 O(1)。
inline void StoreBandValue(char* destination, DataType dtype, double value) {
  if (dtype == DataType::kFloat32) {
    *reinterpret_cast<float*>(destination) = static_cast<float>(value);
    return;
  }
  *reinterpret_cast<double*>(destination) = value;
}

// 线性下标 -> (n, y, x, c)，分解顺序与 ImageLinearIndex 一致：
// NHWC 为 n,y,x,c，NCHW 为 n,c,y,x。复杂度 O(rank)，rank <= 4。
inline void DecodeLinear(const ImageInfo& info, int64_t linear, int64_t& n,
                         int64_t& y, int64_t& x, int64_t& c) {
  int64_t remaining = linear;
  if (info.layout == ImageLayout::kNHWC) {
    c = remaining % info.c;
    remaining /= info.c;
    x = remaining % info.w;
    remaining /= info.w;
    y = remaining % info.h;
    n = remaining / info.h;
  } else {
    x = remaining % info.w;
    remaining /= info.w;
    y = remaining % info.h;
    remaining /= info.h;
    c = remaining % info.c;
    n = remaining / info.c;
  }
}

// 主机内存上的逐元素相减：L = fine - up（native_cpu 与 OpenCV 张量）。
// fine 可能是带 stride 的视图（例如调用者输入），用 ImageByteOffset 按
// (n,y,x,c) 解算真实字节偏移；up 与 output 是刚分配的连续张量，可直接用
// 线性下标寻址。复杂度 O(Numel)，每个元素一次减法与一次类型转换。
inline void SubtractBands(const Tensor& fine, const ImageInfo& info,
                          const Tensor& up, Tensor* output) {
  const char* fine_base = static_cast<const char*>(fine.data());
  const char* up_base = static_cast<const char*>(up.data());
  char* out_base = static_cast<char*>(output->data());
  const size_t up_element = up.ElementSize();
  const size_t out_element = output->ElementSize();
  for (int64_t i = 0; i < output->Numel(); ++i) {
    int64_t n = 0;
    int64_t y = 0;
    int64_t x = 0;
    int64_t c = 0;
    DecodeLinear(info, i, n, y, x, c);
    const double high = LoadBandValue(
        fine_base + ImageByteOffset(fine, info, n, y, x, c), fine.dtype());
    const double low = LoadBandValue(
        up_base + static_cast<size_t>(i) * up_element, up.dtype());
    StoreBandValue(out_base + static_cast<size_t>(i) * out_element,
                   output->dtype(), high - low);
  }
}

}  // namespace

// 主机侧 stage：计算一级带通 L_i 与下一级高斯层 G_{i+1}。
// 输入：context（原样透传给子算子）、G_i（input）、params；other 不使用。
// 输出：output 写 L_i（与 input 同尺寸）；params->next_gaussian 写 G_{i+1}。
// 步骤：先 GaussianBlur 再 ResizeNearest 得到 G_{i+1}，然后用
//       ResizeBilinear 放大回当前尺寸，最后在主机内存逐元素相减。
// 尺寸：下一级取 (h+1)/2、(w+1)/2，即 ceil 半尺寸，奇数边长与 pyrDown
//       的层形状一致；上采样目标始终是当前层的 h x w。
// 复杂度：O(N * kernel_size)，N 为当前层元素数。
Status LaplacianPyramidHostStage(const OpContext& context, const Tensor& input,
                                 const Tensor* /*other*/, const void* params,
                                 Tensor* output) {
  const auto& options = *static_cast<const LaplacianPyramidParams*>(params);
  auto image = ParseImage(input.shape(), options.layout);
  if (!image.ok()) return image.status();

  // 下采样：先高斯平滑，再用 nearest 隔行隔列取样；ceil 保证奇数边长得到
  // 经典 pyrDown 的层形状。
  const int64_t next_h = (image.value().h + 1) / 2;
  const int64_t next_w = (image.value().w + 1) / 2;
  auto blurred = GaussianBlur(context, input, options.kernel_size,
                              options.sigma, options.layout);
  if (!blurred.ok()) return blurred.status();
  auto next = Resize(context, blurred.value(), next_h, next_w,
                     InterpolationMode::kNearest, options.layout);
  if (!next.ok()) return next.status();

  // 上采样回当前层尺寸并相减，得到带通 L_i = G_i - upsample(G_{i+1})。
  auto up = Resize(context, next.value(), image.value().h, image.value().w,
                   InterpolationMode::kBilinear, options.layout);
  if (!up.ok()) return up.status();
  SubtractBands(input, image.value(), up.value(), output);

  *options.next_gaussian = std::move(next.value());
  return Status::Ok();
}

}  // namespace ops_internal
}  // namespace mycv
