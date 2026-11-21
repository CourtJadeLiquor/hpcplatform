#ifndef MYCV_SRC_OPS_WAVELET_WAVELET_CPU_H_
#define MYCV_SRC_OPS_WAVELET_WAVELET_CPU_H_

// wavelet 算子的共享 CPU 骨架（docs/architecture.md §6.1/§7.2）。
// 单层是可分离的两趟滤波器组，边界为周期（wrap-around）：
//   rows:   x[y, :]         -> low_rows[y, :], high_rows[y, :]  (W -> W/2)
//   cols:   low_rows[:, x]  -> LL[:, x], LH[:, x]
//           high_rows[:, x] -> HL[:, x], HH[:, x]              (H -> H/2)
// 一维分析：low[k]  = sum_j h[j] * x[(2k+j) mod N]
//           high[k] = sum_j g[j] * x[(2k+j) mod N]
// 一维合成：out[(2k+j) mod N] += h[j]*low[k] + g[j]*high[k]
// 其中 h 为低通（Haar 或 Db4），g[k] = (-1)^k * h[taps-1-k]。
//
// Runner 策略模板把互相独立的行/列分给不同并行后端；每条线内部按
// “k 升序、j 升序”固定累加，与线程无关，因此三级 CPU 输出位精确。
// 所有累加在 double 中完成，只在写回时窄化到张量 dtype。

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef MYCV_WITH_TBB
#include <tbb/parallel_for.h>
#endif

#include "wavelet_internal.h"

namespace mycv {
namespace ops_internal {

// 执行策略：每个翻译单元只选用一个，共享的平面驱动因此保持单一模板。
// For(begin, end, fn) 的每个下标代表一条独立的行或列，fn 不修改共享状态。
struct WaveletSequentialRunner {
  template <typename Fn>
  static void For(int64_t begin, int64_t end, Fn&& fn) {
    for (int64_t i = begin; i < end; ++i) fn(i);
  }
};

#ifdef MYCV_WITH_OPENMP
// OpenMP 策略：把独立行/列摊给 omp parallel for，默认静态划分。
struct WaveletOpenMPRunner {
  template <typename Fn>
  static void For(int64_t begin, int64_t end, Fn&& fn) {
#pragma omp parallel for
    for (int64_t i = begin; i < end; ++i) fn(i);
  }
};
#endif

#ifdef MYCV_WITH_TBB
// TBB 策略：parallel_for 自动分块并做负载均衡。
struct WaveletTbbRunner {
  template <typename Fn>
  static void For(int64_t begin, int64_t end, Fn&& fn) {
    tbb::parallel_for(begin, end, [&](int64_t i) { fn(i); });
  }
};
#endif

// 分析低通/高通系数由 wavelet_internal.h 的 WaveletLowPass /
// WaveletHighPass 提供，CUDA 翻译单元复用同一份定义，保证系数一致。

// 对 `n` 个样本做一次一维分析（分解），输出 n/2 个低通与高通系数：
//   low[k]  = sum_j h[j] * x[(2k+j) mod n]
//   high[k] = sum_j g[j] * x[(2k+j) mod n]
// `src_stride` / `low_stride` / `high_stride` 以样本为单位，使同一函数既能
// 处理连续行（stride 1）也能处理跨步行缓冲（stride W/2）。周期边界用
// 取模实现；k 与 j 都按升序累加，保证与线程无关。复杂度 O(n*taps)。
inline void AnalyzeSequence(const double* src, int64_t src_stride, int64_t n,
                            const std::vector<double>& low_filter,
                            const std::vector<double>& high_filter,
                            double* low, int64_t low_stride, double* high,
                            int64_t high_stride) {
  const int64_t half = n / 2;
  const int taps = static_cast<int>(low_filter.size());
  for (int64_t k = 0; k < half; ++k) {
    double low_value = 0.0;
    double high_value = 0.0;
    for (int j = 0; j < taps; ++j) {
      const int64_t index = ((2 * k + j) % n) * src_stride;
      low_value += low_filter[static_cast<size_t>(j)] * src[index];
      high_value += high_filter[static_cast<size_t>(j)] * src[index];
    }
    low[k * low_stride] = low_value;
    high[k * high_stride] = high_value;
  }
}

// 一维合成（重建）：先把输出清零，再把每个系数上采样后叠加：
//   out[(2k+j) mod n] += h[j]*low[k] + g[j]*high[k]
// 按 k 升序、j 升序累加，与分析的固定顺序一致；边界为周期。因为多个
// (k, j) 会落到同一输出样本，累加顺序会影响末位，所以顺序必须固定。
// 复杂度 O(n*taps)。
inline void SynthesizeSequence(const double* low, int64_t low_stride,
                               const double* high, int64_t high_stride,
                               int64_t n,
                               const std::vector<double>& low_filter,
                               const std::vector<double>& high_filter,
                               double* output, int64_t output_stride) {
  for (int64_t i = 0; i < n; ++i) output[i * output_stride] = 0.0;
  const int64_t half = n / 2;
  const int taps = static_cast<int>(low_filter.size());
  for (int64_t k = 0; k < half; ++k) {
    const double low_value = low[k * low_stride];
    const double high_value = high[k * high_stride];
    for (int j = 0; j < taps; ++j) {
      const int64_t index = (2 * k + j) % n;
      output[index * output_stride] +=
          low_filter[static_cast<size_t>(j)] * low_value +
          high_filter[static_cast<size_t>(j)] * high_value;
    }
  }
}

// 一次二维分析：先行、后列。
//   行：x[y,:]        -> low_rows[y,:], high_rows[y,:]（W -> W/2）
//   列：low_rows[:,x] -> LL[:,x], LH[:,x]
//       high_rows[:,x]-> HL[:,x], HH[:,x]（H -> H/2）
// 行与行、列与列互不依赖，Runner 只决定哪条线交给哪个线程，因此三级
// CPU 位精确。中间缓冲 low_rows/high_rows 各 H*(W/2) 个 double。
// 复杂度 O(H*W*taps)。
template <typename Runner>
inline void AnalyzePlane(const double* plane, int64_t h, int64_t w,
                         const std::vector<double>& low_filter,
                         const std::vector<double>& high_filter, double* ll,
                         double* lh, double* hl, double* hh) {
  const int64_t half_w = w / 2;
  std::vector<double> low_rows(static_cast<size_t>(h * half_w));
  std::vector<double> high_rows(static_cast<size_t>(h * half_w));
  Runner::For(0, h, [&](int64_t y) {
    AnalyzeSequence(plane + y * w, 1, w, low_filter, high_filter,
                    low_rows.data() + y * half_w, 1,
                    high_rows.data() + y * half_w, 1);
  });
  Runner::For(0, half_w, [&](int64_t x) {
    AnalyzeSequence(low_rows.data() + x, half_w, h, low_filter, high_filter,
                    ll + x, half_w, lh + x, half_w);
    AnalyzeSequence(high_rows.data() + x, half_w, h, low_filter, high_filter,
                    hl + x, half_w, hh + x, half_w);
  });
}

// 二维合成：AnalyzePlane 的严格逆序，先列后行。
//   列：LL/LH -> low_rows，HL/HH -> high_rows（H/2 -> H）
//   行：low_rows/high_rows -> plane（W/2 -> W）
// 中间缓冲与 AnalyzePlane 同尺寸；复杂度 O(H*W*taps)。
template <typename Runner>
inline void SynthesizePlane(const double* ll, const double* lh,
                            const double* hl, const double* hh, int64_t h,
                            int64_t w,
                            const std::vector<double>& low_filter,
                            const std::vector<double>& high_filter,
                            double* plane) {
  const int64_t half_w = w / 2;
  std::vector<double> low_rows(static_cast<size_t>(h * half_w));
  std::vector<double> high_rows(static_cast<size_t>(h * half_w));
  Runner::For(0, half_w, [&](int64_t x) {
    SynthesizeSequence(ll + x, half_w, lh + x, half_w, h, low_filter,
                       high_filter, low_rows.data() + x, half_w);
    SynthesizeSequence(hl + x, half_w, hh + x, half_w, h, low_filter,
                       high_filter, high_rows.data() + x, half_w);
  });
  Runner::For(0, h, [&](int64_t y) {
    SynthesizeSequence(low_rows.data() + y * half_w, 1,
                       high_rows.data() + y * half_w, 1, w, low_filter,
                       high_filter, plane + y * w, 1);
  });
}

// 从输入视图按字节偏移读出 double 值，支持 f32/f64。
inline double LoadWaveletValue(const char* source, DataType dtype) {
  if (dtype == DataType::kFloat32) {
    return *reinterpret_cast<const float*>(source);
  }
  return *reinterpret_cast<const double*>(source);
}

// 把 double 写回输出张量，只在这里窄化到 f32 或保留 f64。
inline void StoreWaveletValue(char* destination, DataType dtype, double value) {
  if (dtype == DataType::kFloat32) {
    *reinterpret_cast<float*>(destination) = static_cast<float>(value);
    return;
  }
  *reinterpret_cast<double*>(destination) = value;
}

// 把一个 (n, c) 平面读入连续的 double 暂存区，复杂度 O(H*W)。
// 读取走 ImageByteOffset，因此源张量可以是非连续视图。
inline void LoadWaveletPlane(const Tensor& source, const ImageInfo& info,
                             int64_t n, int64_t c, std::vector<double>* plane) {
  plane->resize(static_cast<size_t>(info.h * info.w));
  const char* base = static_cast<const char*>(source.data());
  const DataType dtype = source.dtype();
  for (int64_t y = 0; y < info.h; ++y) {
    for (int64_t x = 0; x < info.w; ++x) {
      (*plane)[static_cast<size_t>(y * info.w + x)] = LoadWaveletValue(
          base + ImageByteOffset(source, info, n, y, x, c), dtype);
    }
  }
}

// 把一个平面写回输出张量，复杂度 O(H*W)。输出由入口分配，是连续内存。
inline void StoreWaveletPlane(const std::vector<double>& plane,
                              const ImageInfo& info, int64_t n, int64_t c,
                              Tensor* destination) {
  char* base = static_cast<char*>(destination->data());
  const DataType dtype = destination->dtype();
  for (int64_t y = 0; y < info.h; ++y) {
    for (int64_t x = 0; x < info.w; ++x) {
      StoreWaveletValue(
          base + ImageByteOffset(*destination, info, n, y, x, c), dtype,
          plane[static_cast<size_t>(y * info.w + x)]);
    }
  }
}

// 所有 CPU 级别共用的驱动。`input` 在正向是图像、逆向是 LL，因此两个
// 方向的平面尺寸不同：图像/子带的 ImageInfo 由方向推导，而不是只看张量。
// 每个 (n, c) 平面独立处理：正向 load -> AnalyzePlane -> 写四个子带；
// 逆向 load 四个子带 -> SynthesizePlane -> 写图像。
// 复杂度 O(H*W*taps)。
template <typename Runner>
inline Status WaveletImpl(const Tensor& input, const void* params,
                          Tensor* output) {
  const WaveletParams& options = *static_cast<const WaveletParams*>(params);
  auto parsed = ParseImage(input.shape(), options.layout);
  if (!parsed.ok()) return parsed.status();
  const ImageInfo parsed_info = parsed.value();

  const std::vector<double> low_filter = WaveletLowPass(options.kind);
  const std::vector<double> high_filter = WaveletHighPass(low_filter);

  ImageInfo image_info = parsed_info;
  ImageInfo band_info = parsed_info;
  if (options.inverse) {
    image_info.h = parsed_info.h * 2;
    image_info.w = parsed_info.w * 2;
  } else {
    band_info.h = parsed_info.h / 2;
    band_info.w = parsed_info.w / 2;
  }
  const int64_t band_elems = band_info.h * band_info.w;

  std::vector<double> plane;
  std::vector<double> ll;
  std::vector<double> lh;
  std::vector<double> hl;
  std::vector<double> hh;
  for (int64_t n = 0; n < image_info.n; ++n) {
    for (int64_t c = 0; c < image_info.c; ++c) {
      if (!options.inverse) {
        LoadWaveletPlane(input, image_info, n, c, &plane);
        ll.resize(static_cast<size_t>(band_elems));
        lh.resize(static_cast<size_t>(band_elems));
        hl.resize(static_cast<size_t>(band_elems));
        hh.resize(static_cast<size_t>(band_elems));
        AnalyzePlane<Runner>(plane.data(), image_info.h, image_info.w,
                             low_filter, high_filter, ll.data(), lh.data(),
                             hl.data(), hh.data());
        StoreWaveletPlane(ll, band_info, n, c, output);
        StoreWaveletPlane(lh, band_info, n, c, options.extra_out[0]);
        StoreWaveletPlane(hl, band_info, n, c, options.extra_out[1]);
        StoreWaveletPlane(hh, band_info, n, c, options.extra_out[2]);
      } else {
        LoadWaveletPlane(input, band_info, n, c, &ll);
        LoadWaveletPlane(*options.extra_in[0], band_info, n, c, &lh);
        LoadWaveletPlane(*options.extra_in[1], band_info, n, c, &hl);
        LoadWaveletPlane(*options.extra_in[2], band_info, n, c, &hh);
        plane.resize(static_cast<size_t>(image_info.h * image_info.w));
        SynthesizePlane<Runner>(ll.data(), lh.data(), hl.data(), hh.data(),
                                image_info.h, image_info.w, low_filter,
                                high_filter, plane.data());
        StoreWaveletPlane(plane, image_info, n, c, output);
      }
    }
  }
  return Status::Ok();
}

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_WAVELET_WAVELET_CPU_H_
