#ifndef MYCV_SRC_OPS_FOURIER_FOURIER_CPU_H_
#define MYCV_SRC_OPS_FOURIER_FOURIER_CPU_H_

// fourier 算子的共享 CPU 骨架（docs/architecture.md §6.1/§7.2）。
// 2-D 变换可分离：每个 (n, c) 平面先对每行做长度为 W 的 1-D 变换，再对
// 每列做长度为 H 的 1-D 变换，即 X[k] = sum_n x[n]*exp(-i*2*pi*k*n/N)
// 的二维形式。
//
// Runner 策略模板把“逐行/逐列”的独立任务分给不同并行后端：Sequential、
// OpenMP、TBB 三个 Runner 都只提供 For(begin, end, fn)，差别仅在循环的
// 调度方式；平面驱动与一维内核是同一份模板代码，不需要三份拷贝。
//
// 精度策略：所有累加在 double 中完成，只在写回时窄化到张量 dtype（与
// gaussian_blur 一致）。每一行/列的运算顺序（bit-reversal、按 len 升序
// 的蝶形、DFT 回退的 j 升序累加）与线程无关，因此三级 CPU 输出逐字节
// 相同（位精确）。

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef MYCV_WITH_TBB
#include <tbb/parallel_for.h>
#endif

#include "fourier_internal.h"

namespace mycv {
namespace ops_internal {

// 执行策略：每个翻译单元只选用一个，共享的平面驱动因此保持单一模板。
// For(begin, end, fn) 的每个下标代表一条独立的行或列，fn 不修改共享状态。
struct SequentialRunner {
  template <typename Fn>
  static void For(int64_t begin, int64_t end, Fn&& fn) {
    for (int64_t i = begin; i < end; ++i)
      fn(i);
  }
};

#ifdef MYCV_WITH_OPENMP
// OpenMP 策略：把独立行/列摊给 omp parallel for，默认静态划分。
struct OpenMPRunner {
  template <typename Fn>
  static void For(int64_t begin, int64_t end, Fn&& fn) {
#pragma omp parallel for
    for (int64_t i = begin; i < end; ++i)
      fn(i);
  }
};
#endif

#ifdef MYCV_WITH_TBB
// TBB 策略：parallel_for 自动分块并做负载均衡。
struct TbbRunner {
  template <typename Fn>
  static void For(int64_t begin, int64_t end, Fn&& fn) {
    tbb::parallel_for(begin, end, [&](int64_t i) { fn(i); });
  }
};
#endif

// 长度为 n 的一维 twiddle 表。表里存的是 -2*pi*k/n 的 cos/sin，即
// twiddle = cos(-2*pi*k/N) + i*sin(-2*pi*k/N)；逆向只需把正弦取反，
// 因此一张表同时服务正、逆两个方向，并被所有行/列共享。
struct FftPlan1D {
  int64_t n = 0;
  bool radix2 = false;
  std::vector<double> cos_table;
  std::vector<double> sin_table;
};

// 是否为 2 的幂；只有这种长度才走 O(n log n) 的 radix-2。
inline bool IsPowerOfTwo(int64_t n) {
  return n > 0 && (n & (n - 1)) == 0;
}

// 预计算长度为 n 的 twiddle 表，复杂度 O(n)。表只与 n 有关，每个平面
// 构造一次后所有行/列复用，避免重复计算三角函数。
inline FftPlan1D MakeFftPlan(int64_t n) {
  constexpr double kTwoPi = 6.283185307179586476925286766559;
  FftPlan1D plan;
  plan.n = n;
  plan.radix2 = IsPowerOfTwo(n);
  plan.cos_table.resize(static_cast<size_t>(n));
  plan.sin_table.resize(static_cast<size_t>(n));
  for (int64_t k = 0; k < n; ++k) {
    const double angle =
        -kTwoPi * static_cast<double>(k) / static_cast<double>(n);
    plan.cos_table[static_cast<size_t>(k)] = std::cos(angle);
    plan.sin_table[static_cast<size_t>(k)] = std::sin(angle);
  }
  return plan;
}

// 对 `n` 个复数样本做就地一维变换；`stride` 以复数样本为单位（行内为 1，
// 列方向为 W）。变换核为 exp(-i*2*pi*k*n/N)，逆向把正弦取反，归一化
// 1/(H*W) 留给调用方在写回时一次完成。
//
// 2 的幂：先做 bit-reversal 置换，再按 len = 2,4,...,n 迭代蝶形
//   a' = a + w*b、b' = a - w*b，复杂度 O(n log n)。
// 其他长度：退化为 out-of-place 的直接 DFT，按 j 升序累加，复杂度 O(n^2)；
//   j 升序是刻意的，用来与 CUDA 的 DFT 回退内核保持相同的加法顺序。
inline void Transform1D(double* re, double* im, int64_t stride,
                        const FftPlan1D& plan, bool inverse) {
  const int64_t n = plan.n;
  const double sign = inverse ? -1.0 : 1.0;

  if (plan.radix2) {
    // bit-reversal 置换：把下标 i 与位序反转后的下标对调。
    for (int64_t i = 1, j = 0; i < n; ++i) {
      int64_t bit = n >> 1;
      for (; (j & bit) != 0; bit >>= 1)
        j ^= bit;
      j ^= bit;
      if (i < j) {
        std::swap(re[i * stride], re[j * stride]);
        std::swap(im[i * stride], im[j * stride]);
      }
    }
    // 迭代 Cooley-Tukey 蝶形：len 逐级倍增，级内按输出下标升序推进。
    for (int64_t len = 2; len <= n; len <<= 1) {
      const int64_t half = len >> 1;
      const int64_t step = n / len;
      for (int64_t base = 0; base < n; base += len) {
        for (int64_t j = 0; j < half; ++j) {
          const double wr = plan.cos_table[static_cast<size_t>(j * step)];
          const double wi =
              sign * plan.sin_table[static_cast<size_t>(j * step)];
          const int64_t a = (base + j) * stride;
          const int64_t b = (base + j + half) * stride;
          const double vr = re[b] * wr - im[b] * wi;
          const double vi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - vr;
          im[b] = im[a] - vi;
          re[a] += vr;
          im[a] += vi;
        }
      }
    }
    return;
  }

  // 非 2 的幂：直接按定义做 O(n^2) DFT，对任意 n 都正确。
  std::vector<double> out_re(static_cast<size_t>(n));
  std::vector<double> out_im(static_cast<size_t>(n));
  for (int64_t k = 0; k < n; ++k) {
    double sum_re = 0.0;
    double sum_im = 0.0;
    int64_t index = 0;  // (j*k) mod n，每取一个样本自增 k 并减法回绕
    for (int64_t j = 0; j < n; ++j) {
      const double wr = plan.cos_table[static_cast<size_t>(index)];
      const double wi = sign * plan.sin_table[static_cast<size_t>(index)];
      const double xr = re[j * stride];
      const double xi = im[j * stride];
      sum_re += xr * wr - xi * wi;
      sum_im += xr * wi + xi * wr;
      index += k;
      if (index >= n) index -= n;
    }
    out_re[static_cast<size_t>(k)] = sum_re;
    out_im[static_cast<size_t>(k)] = sum_im;
  }
  for (int64_t k = 0; k < n; ++k) {
    re[k * stride] = out_re[static_cast<size_t>(k)];
    im[k * stride] = out_im[static_cast<size_t>(k)];
  }
}

// 先行后列：行用 stride 1、长度 W，列用 stride W、长度 H。
// 行与行、列与列互不依赖，策略只决定“哪条线交给哪个线程”，不改变任何
// 一条线内部的运算顺序 —— 这正是三级 CPU 位精确的前提。
template <typename Runner>
inline void TransformPlane(double* re, double* im, int64_t h, int64_t w,
                           const FftPlan1D& plan_h, const FftPlan1D& plan_w,
                           bool inverse) {
  Runner::For(0, h, [&](int64_t y) {
    Transform1D(re + y * w, im + y * w, 1, plan_w, inverse);
  });
  Runner::For(0, w, [&](int64_t x) {
    Transform1D(re + x, im + x, w, plan_h, inverse);
  });
}

// 从可能非连续的输入视图按字节偏移读出 double 值，支持 f32/f64。
inline double LoadFourierValue(const char* source, DataType dtype) {
  if (dtype == DataType::kFloat32) {
    return *reinterpret_cast<const float*>(source);
  }
  return *reinterpret_cast<const double*>(source);
}

// 把 double 写回输出张量，只在这里窄化到 f32 或保留 f64。
inline void StoreFourierValue(char* destination, DataType dtype, double value) {
  if (dtype == DataType::kFloat32) {
    *reinterpret_cast<float*>(destination) = static_cast<float>(value);
    return;
  }
  *reinterpret_cast<double*>(destination) = value;
}

// 把一个 (n, c) 平面读入连续的复数暂存区（实部 + 虚部），复杂度 O(H*W)。
// `imag_source` 为 nullptr 表示正向输入是实值，虚部统一填 0；读取走
// ImageByteOffset，因此输入可以是非连续的张量视图。
inline void LoadPlane(const Tensor& real_source, const Tensor* imag_source,
                      const ImageInfo& info, int64_t n, int64_t c,
                      std::vector<double>* re, std::vector<double>* im) {
  const int64_t plane_size = info.h * info.w;
  re->resize(static_cast<size_t>(plane_size));
  im->resize(static_cast<size_t>(plane_size));
  const char* real_base = static_cast<const char*>(real_source.data());
  const char* imag_base = imag_source == nullptr
                              ? nullptr
                              : static_cast<const char*>(imag_source->data());
  const DataType real_dtype = real_source.dtype();
  const DataType imag_dtype =
      imag_source == nullptr ? DataType::kFloat32 : imag_source->dtype();
  for (int64_t y = 0; y < info.h; ++y) {
    for (int64_t x = 0; x < info.w; ++x) {
      const int64_t linear = y * info.w + x;
      (*re)[static_cast<size_t>(linear)] = LoadFourierValue(
          real_base + ImageByteOffset(real_source, info, n, y, x, c),
          real_dtype);
      if (imag_base != nullptr) {
        (*im)[static_cast<size_t>(linear)] = LoadFourierValue(
            imag_base + ImageByteOffset(*imag_source, info, n, y, x, c),
            imag_dtype);
      } else {
        (*im)[static_cast<size_t>(linear)] = 0.0;
      }
    }
  }
}

// 把一个平面写回输出张量，复杂度 O(H*W)。`scale` 是逆向归一化因子
// 1/(H*W)（正向为 1）；`imag_output` 为 nullptr 时跳过虚部。输出张量由
// 入口分配，是连续内存。
inline void StorePlane(const std::vector<double>& re,
                       const std::vector<double>& im, const ImageInfo& info,
                       int64_t n, int64_t c, double scale, Tensor* real_output,
                       Tensor* imag_output) {
  char* real_base = static_cast<char*>(real_output->data());
  char* imag_base = imag_output == nullptr
                        ? nullptr
                        : static_cast<char*>(imag_output->data());
  const DataType real_dtype = real_output->dtype();
  const DataType imag_dtype =
      imag_output == nullptr ? DataType::kFloat32 : imag_output->dtype();
  for (int64_t y = 0; y < info.h; ++y) {
    for (int64_t x = 0; x < info.w; ++x) {
      const int64_t linear = y * info.w + x;
      StoreFourierValue(
          real_base + ImageByteOffset(*real_output, info, n, y, x, c),
          real_dtype, re[static_cast<size_t>(linear)] * scale);
      if (imag_base != nullptr) {
        StoreFourierValue(
            imag_base + ImageByteOffset(*imag_output, info, n, y, x, c),
            imag_dtype, im[static_cast<size_t>(linear)] * scale);
      }
    }
  }
}

// 所有 CPU 级别共用的驱动：构造一次行/列 twiddle 表，逐 (n, c) 平面执行
// load -> TransformPlane -> store。`imag_input` 是入口的 `other`（仅逆向
// 使用），`options.imag` 是正向的虚部输出槽位。
//
// 复杂度：radix-2 时每平面 O(H*W*(log H + log W))；任一维非 2 的幂时
// 回退为 O(H*W*(H + W))。
template <typename Runner>
inline Status FourierImpl(const Tensor& input, const Tensor* imag_input,
                          const FourierParams& options, Tensor* real_output) {
  auto parsed = ParseImage(input.shape(), options.layout);
  if (!parsed.ok()) return parsed.status();
  const ImageInfo& info = parsed.value();

  const FftPlan1D plan_h = MakeFftPlan(info.h);
  const FftPlan1D plan_w = MakeFftPlan(info.w);
  const double scale =
      options.inverse
          ? 1.0 / (static_cast<double>(info.h) * static_cast<double>(info.w))
          : 1.0;

  std::vector<double> re;
  std::vector<double> im;
  for (int64_t n = 0; n < info.n; ++n) {
    for (int64_t c = 0; c < info.c; ++c) {
      LoadPlane(input, imag_input, info, n, c, &re, &im);
      TransformPlane<Runner>(re.data(), im.data(), info.h, info.w, plan_h,
                             plan_w, options.inverse);
      StorePlane(re, im, info, n, c, scale, real_output, options.imag);
    }
  }
  return Status::Ok();
}

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_FOURIER_FOURIER_CPU_H_
