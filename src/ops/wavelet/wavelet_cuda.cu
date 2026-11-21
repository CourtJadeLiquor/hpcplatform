// wavelet 算子的 CUDA 实现（docs/architecture.md §6.1/§6.2）。
//
// 单层是可分离的两趟滤波器组，边界为周期：
//   rows:  x[y, :]         -> low_rows[y, :], high_rows[y, :]  (W -> W/2)
//   cols:  low_rows[:, x]  -> LL[:, x], LH[:, x]
//          high_rows[:, x] -> HL[:, x], HH[:, x]              (H -> H/2)
// 正向内核顺序：LoadPlane -> AnalyzeRows -> AnalyzeCols（直接写四带）。
// 逆向内核顺序：LoadBands -> SynthesizeCols -> SynthesizeRows -> StorePlane。
// 每个输出样本由一个线程独立计算，没有跨样本依赖，因此块内不需要同步。
//
// 精度策略：所有累加用 double，滤波器来自与 CPU 共用的 host 函数，只在
// 写回时窄化。CUDA 合成由 2k+j = y (mod N) 反解系数下标、按 j 迭代，而
// CPU 按 k 累加，两者加法顺序不同；加上设备端 FMA 收缩，跨后端对拍使用
// 文档约定的 1e-4 相对容差。

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "backends/cuda/cuda_device_utils.h"
#include "mycv/ops/image.h"
#include "wavelet_internal.h"

namespace mycv {
namespace ops_internal {
namespace {

// 按值携带到设备的滤波器组：taps 只有 2（Haar）或 4（Db4），最多 4 项，
// 远小于内核参数区上限。因此直接按值传参，无需单独的 device 常量表，
// 也避免每个线程去读全局内存。
struct WaveletFilter {
  double low[4];
  double high[4];
  int taps;
};

// 在主机端由共享的 WaveletLowPass/WaveletHighPass 生成当前调用的小滤波器，
// 复杂度 O(taps)；系数与 CPU 逐位相同。
WaveletFilter MakeDeviceFilter(WaveletKind kind) {
  const std::vector<double> low = WaveletLowPass(kind);
  const std::vector<double> high = WaveletHighPass(low);
  WaveletFilter filter = {};
  filter.taps = static_cast<int>(low.size());
  for (int i = 0; i < filter.taps && i < 4; ++i) {
    filter.low[i] = low[static_cast<size_t>(i)];
    filter.high[i] = high[static_cast<size_t>(i)];
  }
  return filter;
}

// 从设备输入按元素偏移读出 double，支持 f32/f64。
__device__ inline double LoadValue(const void* base, int64_t offset,
                                   DataType dtype) {
  if (dtype == DataType::kFloat32) {
    return static_cast<double>(static_cast<const float*>(base)[offset]);
  }
  return static_cast<const double*>(base)[offset];
}

// 把 double 写回设备张量，只在这里窄化到 f32 或保留 f64。
__device__ inline void StoreValue(void* base, int64_t offset, DataType dtype,
                                  double value) {
  if (dtype == DataType::kFloat32) {
    static_cast<float*>(base)[offset] = static_cast<float>(value);
    return;
  }
  static_cast<double*>(base)[offset] = value;
}

// 新分配输出张量的扁平下标（连续内存），按布局公式直接计算。
__device__ inline int64_t ImageLinearOffset(const DeviceImage& image, int64_t n,
                                            int64_t y, int64_t x, int64_t c) {
  if (image.layout == ImageLayout::kNHWC) {
    return ((n * image.h + y) * image.w + x) * image.c + c;
  }
  return ((n * image.c + c) * image.h + y) * image.w + x;
}

// 正向第 1 步：把一个 (n, c) 平面读入 double 暂存区，复杂度 O(H*W)。
// 扁平下标 index 按平面行优先布局解码：y = index / w，
// x = index - y*w；取数走 ImageElementOffset，因此输入可以是非连续视图。
// 每个线程处理若干元素，样本之间互不依赖，无需任何同步。
__global__ void WaveletLoadPlaneKernel(DeviceTensor source, DeviceImage image,
                                       int64_t n, int64_t c, double* plane,
                                       int64_t count, DataType dtype) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t y = index / image.w;
    const int64_t x = index - y * image.w;
    plane[index] = LoadValue(source.data,
                             ImageElementOffset(source, image, n, y, x, c),
                             dtype);
  }
}

// 正向第 2 步（行分析）：每个线程负责一个 (y, k)，独立算出该行的第 k 个
// 低通/高通系数：
//   low_rows[y,k]  = sum_j h[j] * x[y, (2k+j) mod W]
//   high_rows[y,k] = sum_j g[j] * x[y, (2k+j) mod W]
// 一个输出只由一个线程写，因此块内无需同步。周期边界本可用取模，但
// 2*k + taps - 1 在 W >= 2 时一定小于 2*W，所以一次减法即可回绕。
// count = H*(W/2)，复杂度 O(H*(W/2)*taps)。
__global__ void WaveletAnalyzeRowsKernel(const double* plane, double* low_rows,
                                         double* high_rows, int64_t w,
                                         int64_t half_w, WaveletFilter filter,
                                         int64_t count) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t y = index / half_w;
    const int64_t k = index - y * half_w;
    double low_value = 0.0;
    double high_value = 0.0;
    for (int j = 0; j < filter.taps; ++j) {
      int64_t source = 2 * k + j;
      if (source >= w) source -= w;
      const double value = plane[y * w + source];
      low_value += filter.low[j] * value;
      high_value += filter.high[j] * value;
    }
    low_rows[index] = low_value;
    high_rows[index] = high_value;
  }
}

// 正向第 3 步（列分析）：每个线程负责一个 (yy, xx)，对两条行缓冲做列方向
// 滤波，并直接写入四个子带张量，省去一次单独的回写：
//   LL[yy,xx] = sum_j h[j] * low_rows[(2yy+j) mod H, xx]
//   LH[yy,xx] = sum_j g[j] * low_rows[...]（行低通 + 列高通）
//   HL[yy,xx] = sum_j h[j] * high_rows[...]（行高通 + 列低通）
//   HH[yy,xx] = sum_j g[j] * high_rows[...]
// 周期边界同样用一次减法；每个线程独立写四个输出，无需同步。
// count = (H/2)*(W/2)，复杂度 O((H/2)*(W/2)*taps)。
__global__ void WaveletAnalyzeColsKernel(
    const double* low_rows, const double* high_rows, DeviceTensor band,
    DeviceImage band_image, void* ll_dst, void* lh_dst, void* hl_dst,
    void* hh_dst, int64_t n, int64_t c, int64_t h, int64_t half_w,
    WaveletFilter filter, int64_t count, DataType dtype) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t yy = index / half_w;
    const int64_t xx = index - yy * half_w;
    double ll = 0.0;
    double lh = 0.0;
    double hl = 0.0;
    double hh = 0.0;
    for (int j = 0; j < filter.taps; ++j) {
      int64_t source = 2 * yy + j;
      if (source >= h) source -= h;
      const int64_t offset = source * half_w + xx;
      ll += filter.low[j] * low_rows[offset];
      lh += filter.high[j] * low_rows[offset];
      hl += filter.low[j] * high_rows[offset];
      hh += filter.high[j] * high_rows[offset];
    }
    const int64_t destination =
        ImageElementOffset(band, band_image, n, yy, xx, c);
    StoreValue(ll_dst, destination, dtype, ll);
    StoreValue(lh_dst, destination, dtype, lh);
    StoreValue(hl_dst, destination, dtype, hl);
    StoreValue(hh_dst, destination, dtype, hh);
  }
}

// 逆向第 1 步：把 LL/LH/HL/HH 四个子带读入连续的 double 暂存区。
// 每个线程处理一个 (yy, xx)，四次读取共用同一元素偏移；输入子带可以是
// 非连续视图。count = (H/2)*(W/2)，复杂度 O(H*W/4)。
__global__ void WaveletLoadBandsKernel(
    DeviceTensor band, DeviceImage band_image, const void* ll_src,
    const void* lh_src, const void* hl_src, const void* hh_src, double* ll,
    double* lh, double* hl, double* hh, int64_t n, int64_t c, int64_t half_w,
    int64_t count, DataType dtype) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t yy = index / half_w;
    const int64_t xx = index - yy * half_w;
    const int64_t offset = ImageElementOffset(band, band_image, n, yy, xx, c);
    ll[index] = LoadValue(ll_src, offset, dtype);
    lh[index] = LoadValue(lh_src, offset, dtype);
    hl[index] = LoadValue(hl_src, offset, dtype);
    hh[index] = LoadValue(hh_src, offset, dtype);
  }
}

// 逆向第 2 步（列合成）：由四带重建两条行缓冲（H/2 -> H）。
// 合成公式为 out[y] = sum_{k,j} h[j]*low[k] + g[j]*high[k]，
// 其中 2k + j = y (mod h)。反过来固定 y 与 j 解 k：
//   k = (y - j) / 2 (mod h/2)，
// 因为 N 为偶数，只有 (y-j) 为偶数时才存在整数 k，且唯一；于是每个输出
// 只需迭代 taps 次，而不是 h/2 次。y-j 为负时先加 h 回绕。
// 每个线程负责一个 (y, x)，各写 low_rows/high_rows 的一个元素，无需同步。
// count = H*(W/2)，复杂度 O(H*(W/2)*taps)。
__global__ void WaveletSynthesizeColsKernel(
    const double* ll, const double* lh, const double* hl, const double* hh,
    double* low_rows, double* high_rows, int64_t h, int64_t half_w,
    WaveletFilter filter, int64_t count) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t y = index / half_w;
    const int64_t x = index - y * half_w;
    double low_value = 0.0;
    double high_value = 0.0;
    for (int j = 0; j < filter.taps; ++j) {
      int64_t difference = (y - j) % h;
      if (difference < 0) difference += h;
      if ((difference & 1) != 0) continue;
      const int64_t offset = (difference >> 1) * half_w + x;
      low_value += filter.low[j] * ll[offset] + filter.high[j] * lh[offset];
      high_value += filter.low[j] * hl[offset] + filter.high[j] * hh[offset];
    }
    low_rows[index] = low_value;
    high_rows[index] = high_value;
  }
}

// 逆向第 3 步（行合成）：把两条行缓冲（W/2 -> W）合成为图像平面。
// 与列合成同理，对每个输出 x 反解 k = (x-j)/2 (mod w/2)，仅当 x-j 为偶数
// 时有效；每个线程独立写 plane 的一个样本，无需同步。
// 复杂度 O(H*W*taps)。这里按 j 迭代，与 CPU 按 k 累加的顺序不同，属于
// 文档允许的 1e-4 相对容差范围。
__global__ void WaveletSynthesizeRowsKernel(const double* low_rows,
                                            const double* high_rows,
                                            double* plane, int64_t w,
                                            int64_t half_w,
                                            WaveletFilter filter,
                                            int64_t count) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t y = index / w;
    const int64_t x = index - y * w;
    double value = 0.0;
    for (int j = 0; j < filter.taps; ++j) {
      int64_t difference = (x - j) % w;
      if (difference < 0) difference += w;
      if ((difference & 1) != 0) continue;
      const int64_t offset = y * half_w + (difference >> 1);
      value +=
          filter.low[j] * low_rows[offset] + filter.high[j] * high_rows[offset];
    }
    plane[index] = value;
  }
}

// 逆向最后一步：把 double 合成平面写回输出张量的对应 (n, c) 平面。
// 输出是连续张量，用 ImageLinearOffset 直接算偏移；只在这里窄化 dtype。
// 复杂度 O(H*W)。
__global__ void WaveletStorePlaneKernel(const double* plane, void* destination,
                                        DeviceImage image, int64_t n, int64_t c,
                                        int64_t count, DataType dtype) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t y = index / image.w;
    const int64_t x = index - y * image.w;
    StoreValue(destination, ImageLinearOffset(image, n, y, x, c), dtype,
               plane[index]);
  }
}

}  // namespace

// CUDA 流水线编排：主机侧只发射内核，不做逐元素计算。
// 步骤：
//  1. 取 stream 并按 layout 解析；正向由图像得到 H/W，逆向由子带乘 2；
//  2. 申请一块 2*H*W 的 f64 暂存区，按方向复用划分（见下方注释）；
//  3. 由共享的 host 系数生成按值传参的 WaveletFilter；
//  4. 逐平面串行发射正向 3 个或逆向 4 个内核，全部进入同一 stream；
//  5. 最后统一 FinishLaunch 检查错误并同步。
// 逆向不使用入口的 `other`（四带走 extra_in），故该参数匿名。
Status WaveletCuda(const OpContext& context, const Tensor& input,
                   const Tensor*, const void* params, Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  cudaStream_t stream = stream_result.value()->handle();
  const auto& options = *static_cast<const WaveletParams*>(params);
  auto parsed = ParseImage(input.shape(), options.layout);
  if (!parsed.ok()) return parsed.status();

  // 正向按图像尺寸解析，逆向按子带尺寸解析并把两个方向都乘 2。
  ImageInfo image_info = parsed.value();
  ImageInfo band_info = parsed.value();
  if (options.inverse) {
    image_info.h *= 2;
    image_info.w *= 2;
  } else {
    band_info.h /= 2;
    band_info.w /= 2;
  }
  const int64_t h = image_info.h;
  const int64_t w = image_info.w;
  const int64_t half_h = h / 2;
  const int64_t half_w = w / 2;
  const int64_t plane_size = h * w;
  const int64_t band_elems = half_h * half_w;
  const int64_t row_elems = h * half_w;

  // 一块 f64 暂存区容纳当前方向的所有中间平面：
  //   正向：plane (H*W) + low_rows/high_rows（各 H*W/2）
  //   逆向：LL/LH/HL/HH（各 H*W/4）+ low_rows/high_rows（各 H*W/2）
  // 两组视图起点相同（plane 与 ll 都指向 scratch_base），因为两个方向
  // 不会同时使用。
  auto scratch = context.context->AllocateTensor(Shape{2 * plane_size},
                                                 DataType::kFloat64);
  if (!scratch.ok()) return scratch.status();
  double* scratch_base = static_cast<double*>(scratch.value().data());
  double* plane = scratch_base;
  double* low_rows = scratch_base + plane_size;
  double* high_rows = low_rows + row_elems;
  double* ll = scratch_base;
  double* lh = ll + band_elems;
  double* hl = lh + band_elems;
  double* hh = hl + band_elems;

  const WaveletFilter filter = MakeDeviceFilter(options.kind);
  const int64_t planes = image_info.n * image_info.c;
  for (int64_t p = 0; p < planes; ++p) {
    const int64_t n = p / image_info.c;
    const int64_t c = p % image_info.c;
    if (!options.inverse) {
      WaveletLoadPlaneKernel<<<BlockCount(plane_size), kThreadsPerBlock, 0,
                               stream>>>(Describe(input), Describe(image_info),
                                         n, c, plane, plane_size,
                                         input.dtype());
      WaveletAnalyzeRowsKernel<<<BlockCount(row_elems), kThreadsPerBlock, 0,
                                 stream>>>(
          plane, low_rows, high_rows, w, half_w, filter, row_elems);
      WaveletAnalyzeColsKernel<<<BlockCount(band_elems), kThreadsPerBlock, 0,
                                 stream>>>(
          low_rows, high_rows, Describe(*output), Describe(band_info),
          output->data(), options.extra_out[0]->data(),
          options.extra_out[1]->data(), options.extra_out[2]->data(), n, c, h,
          half_w, filter, band_elems, output->dtype());
    } else {
      WaveletLoadBandsKernel<<<BlockCount(band_elems), kThreadsPerBlock, 0,
                               stream>>>(
          Describe(input), Describe(band_info), input.data(),
          options.extra_in[0]->data(), options.extra_in[1]->data(),
          options.extra_in[2]->data(), ll, lh, hl, hh, n, c, half_w,
          band_elems, input.dtype());
      WaveletSynthesizeColsKernel<<<BlockCount(row_elems), kThreadsPerBlock, 0,
                                    stream>>>(
          ll, lh, hl, hh, low_rows, high_rows, h, half_w, filter, row_elems);
      WaveletSynthesizeRowsKernel<<<BlockCount(plane_size), kThreadsPerBlock, 0,
                                    stream>>>(
          low_rows, high_rows, plane, w, half_w, filter, plane_size);
      WaveletStorePlaneKernel<<<BlockCount(plane_size), kThreadsPerBlock, 0,
                                stream>>>(
          plane, output->data(), Describe(image_info), n, c, plane_size,
          output->dtype());
    }
  }
  return FinishLaunch(stream, "Wavelet kernel launch");
}

}  // namespace ops_internal
}  // namespace mycv
