#ifndef MYCV_SRC_OPS_RESIZE_RESIZE_CPU_H_
#define MYCV_SRC_OPS_RESIZE_RESIZE_CPU_H_

// resize 的共享 CPU 骨架：一个像素级核 ResizeElement，外加三种外层循环
// 组织方式（naive 串行 / OpenMP / TBB）。所有 CPU level 都调用同一个核、
// 按同一套逐元素独立的公式写输出，互不累加，因此彼此位精确一致。
//
// 算法：
//   最近邻：src = floor(dst * scale)，再截到 h-1/w-1；
//   双线性：src = (dst + 0.5) * scale - 0.5，clamp 到 >= 0 后 floor 得左上
//   角，右下角 y1/x1 越界时回退到 h-1/w-1，权重 wy/wx = src - floor(src)；
//   插值顺序是先沿 x 做两次水平插值，再沿 y 做一次垂直插值。
// 中间量用 long double 累加；u8 输出先 std::round 再饱和写回。

#include <algorithm>
#include <cmath>

#ifdef MYCV_WITH_TBB
#include <tbb/parallel_for.h>
#endif

#include "core/value_cast.h"
#include "framework/image_utils.h"
#include "ops/common/op_common.h"
#include "resize_internal.h"

namespace mycv {
namespace ops_internal {

// 计算单个输出元素 output[output_linear]。
// 输入：原图 input、输入/输出布局 image/output_info、缩放参数 options，
// 以及按 output_info.layout 编码的输出扁平下标 output_linear。
// 步骤：解码 (n,y,x,c) -> 算源坐标与权重 -> 取 1 或 4 个源像素 ->
// 最近邻直接取值或双线性插值 -> 按 dtype 写回。
// 复杂度 O(1)（至多 4 次访存）；整幅输出为 O(N*H_out*W_out*C)。
inline void ResizeElement(const Tensor& input, const ImageInfo& image,
                          const ImageInfo& output_info,
                          const ResizeParams& options, int64_t output_linear,
                          Tensor* output) {
  // 按输出布局把扁平下标还原成 (n, y, x, c)。
  int64_t n = 0;
  int64_t y = 0;
  int64_t x = 0;
  int64_t c = 0;
  int64_t remaining = output_linear;
  if (output_info.layout == ImageLayout::kNHWC) {
    c = remaining % output_info.c;
    remaining /= output_info.c;
    x = remaining % output_info.w;
    remaining /= output_info.w;
    y = remaining % output_info.h;
    n = remaining / output_info.h;
  } else {
    x = remaining % output_info.w;
    remaining /= output_info.w;
    y = remaining % output_info.h;
    remaining /= output_info.h;
    c = remaining % output_info.c;
    n = remaining / output_info.c;
  }

  // scale = 输入边长 / 输出边长（放大时 < 1，缩小时 > 1）。
  const long double scale_y =
      static_cast<long double>(image.h) / options.height;
  const long double scale_x = static_cast<long double>(image.w) / options.width;
  int64_t y0 = 0;
  int64_t x0 = 0;
  int64_t y1 = 0;
  int64_t x1 = 0;
  long double wy = 0;
  long double wx = 0;
  if (options.interpolation == InterpolationMode::kNearest) {
    // 最近邻：左上角对齐，floor(dst*scale) 后截到边界内。
    y0 = std::min(image.h - 1, static_cast<int64_t>(y * scale_y));
    x0 = std::min(image.w - 1, static_cast<int64_t>(x * scale_x));
  } else {
    // 双线性：半像素中心映射，clamp 下界到 0 后取 floor。
    const long double source_y = (y + 0.5L) * scale_y - 0.5L;
    const long double source_x = (x + 0.5L) * scale_x - 0.5L;
    const long double clamped_y = std::max(0.0L, source_y);
    const long double clamped_x = std::max(0.0L, source_x);
    y0 = static_cast<int64_t>(std::floor(clamped_y));
    x0 = static_cast<int64_t>(std::floor(clamped_x));
    // 右下邻点越界时回退到最后一个有效行/列（边界复制）。
    y1 = std::min(image.h - 1, y0 + 1);
    x1 = std::min(image.w - 1, x0 + 1);
    wy = clamped_y - y0;
    wx = clamped_x - x0;
  }

  // 输入可能非连续，统一按步长寻址；数值统一提升到 long double。
  const char* source = static_cast<const char*>(input.data());
  long double value = 0;
  if (options.interpolation == InterpolationMode::kNearest) {
    value = detail::LoadAsLongDouble(
        source + ImageByteOffset(input, image, n, y0, x0, c), input.dtype());
  } else {
    const long double top_left = detail::LoadAsLongDouble(
        source + ImageByteOffset(input, image, n, y0, x0, c), input.dtype());
    const long double top_right = detail::LoadAsLongDouble(
        source + ImageByteOffset(input, image, n, y0, x1, c), input.dtype());
    const long double bottom_left = detail::LoadAsLongDouble(
        source + ImageByteOffset(input, image, n, y1, x0, c), input.dtype());
    const long double bottom_right = detail::LoadAsLongDouble(
        source + ImageByteOffset(input, image, n, y1, x1, c), input.dtype());
    // 先水平：上下两行各按 wx 插值；再垂直：用 wy 混合两行结果。
    const long double top = top_left + (top_right - top_left) * wx;
    const long double bottom = bottom_left + (bottom_right - bottom_left) * wx;
    value = top + (bottom - top) * wy;
  }
  // u8 使用四舍五入（half away from zero），再经 Store 做饱和转换。
  if (input.dtype() == DataType::kUInt8) value = std::round(value);
  char* destination = static_cast<char*>(output->data());
  detail::StoreFromLongDouble(
      destination + static_cast<size_t>(output_linear) * output->ElementSize(),
      output->dtype(), value);
}

// 由输入形状解析输入 ImageInfo，再复制一份把 h/w 换成目标尺寸得到输出
// ImageInfo；两者 rank/n/c/layout 保持一致。
// 形状与布局组合非法时返回 InvalidArgument。复杂度 O(rank)。
inline Status PrepareResize(const Tensor& input, const ResizeParams& options,
                            ImageInfo* image, ImageInfo* output_info) {
  auto parsed = ParseImage(input.shape(), options.layout);
  if (!parsed.ok()) return parsed.status();
  *image = parsed.value();
  *output_info = parsed.value();
  output_info->h = options.height;
  output_info->w = options.width;
  return Status::Ok();
}

// naive 实现：单线程按输出扁平下标 [0, Numel()) 依次调用 ResizeElement。
// 不依赖任何并行库，作为其余 CPU level 的参考实现。
// 复杂度 O(N*H_out*W_out*C)。
inline Status ResizeNaiveImpl(const Tensor& input, const ResizeParams& options,
                              Tensor* output) {
  ImageInfo image;
  ImageInfo output_info;
  MYCV_RETURN_IF_ERROR(PrepareResize(input, options, &image, &output_info));
  for (int64_t i = 0; i < output->Numel(); ++i) {
    ResizeElement(input, image, output_info, options, i, output);
  }
  return Status::Ok();
}

// OpenMP 实现：parallel for 直接切分扁平输出下标，粒度为单元素，未写
// schedule 子句，调度方式与线程数均由 OpenMP 运行时决定（通常为
// static 均分，线程数受 OMP_NUM_THREADS 等影响）。每个 i 只写自己的
// 输出元素、无跨迭代依赖，且与 naive 共用同一像素核，故两者逐位一致。
// 复杂度 O(N*H_out*W_out*C)，由线程并行分摊。
inline Status ResizeOpenMPImpl(const Tensor& input, const ResizeParams& options,
                               Tensor* output) {
  ImageInfo image;
  ImageInfo output_info;
  MYCV_RETURN_IF_ERROR(PrepareResize(input, options, &image, &output_info));
#pragma omp parallel for
  for (int64_t i = 0; i < output->Numel(); ++i) {
    ResizeElement(input, image, output_info, options, i, output);
  }
  return Status::Ok();
}

// TBB 实现：parallel_for(0, Numel(), body) 让 TBB 用默认 auto_partitioner
// 把半开区间递归二分（配合 work stealing 动态调度），body 对每个下标调用
// 同一像素核；迭代彼此独立且写法与 naive 相同，因此位精确一致。
// 复杂度 O(N*H_out*W_out*C)。
#ifdef MYCV_WITH_TBB
inline Status ResizeTbbImpl(const Tensor& input, const ResizeParams& options,
                            Tensor* output) {
  ImageInfo image;
  ImageInfo output_info;
  MYCV_RETURN_IF_ERROR(PrepareResize(input, options, &image, &output_info));
  tbb::parallel_for(int64_t{0}, output->Numel(), [&](int64_t i) {
    ResizeElement(input, image, output_info, options, i, output);
  });
  return Status::Ok();
}
#endif

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_RESIZE_RESIZE_CPU_H_
