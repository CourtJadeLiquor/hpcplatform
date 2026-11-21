#include <immintrin.h>

#include <cstring>

#include "gaussian_blur_cpu.h"

namespace mycv {
namespace ops_internal {

namespace {

// AVX2 后端：可分离两趟一维卷积（水平后垂直），每趟 O(H*W*K)，
// 把每像素 O(K^2) 降到 O(2K)。优化点是把「通道」方向装进 256 位
// 寄存器：以 __m256d 为累加器，每轮同时处理同一像素的 4 个连续
// 通道（256 位按 double 计 4 lane，按 float 计相当于 8 lane）。
// 前提：输入连续、layout 为 NHWC（HWC 解析后同为 NHWC）且 c >= 4；
// 其余情况（HW/CHW/c < 4）整体回退到共享标量骨架，结果同字节。
// tap 顺序与 naive 相同，且乘加写成独立的 mul + add、不生成 FMA，
// 因此与 naive 位精确一致；c % 4 的通道尾部走标量路径。
constexpr int kLanes = 4;  // 一个 __m256d 覆盖的通道数（4 个 double）

// 判断能否走向量路径：至少三维、NHWC 排布且通道数够填满一个寄存器。
bool Avx2Applicable(const ImageInfo& info) {
  return info.rank >= 3 && info.layout == ImageLayout::kNHWC &&
         info.c >= kLanes;
}

// 从 base 的 first_linear 处取 4 个连续通道，统一转成 4 个 double lane。
// u8：memcpy 4 字节 -> _mm_cvtepu8_epi32 零扩展成 4 个 int32
//     -> _mm256_cvtepi32_pd；零扩展即 0..255 直接转 double，无符号问题。
// f32：_mm_loadu_ps 读 4 个 float -> _mm256_cvtps_pd。
// f64：_mm256_loadu_pd 直接读 4 个 double。
// 输入输出：返回一个 __m256d，不修改内存；O(1)。
__m256d LoadFourPixels(const char* base, DataType dtype, int64_t first_linear) {
  if (dtype == DataType::kUInt8) {
    uint32_t four_bytes = 0;
    std::memcpy(&four_bytes, base + first_linear, sizeof(four_bytes));
    const __m128i bytes = _mm_cvtsi32_si128(static_cast<int>(four_bytes));
    return _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(bytes));
  }
  if (dtype == DataType::kFloat32) {
    return _mm256_cvtps_pd(
        _mm_loadu_ps(reinterpret_cast<const float*>(base + first_linear * 4)));
  }
  return _mm256_loadu_pd(
      reinterpret_cast<const double*>(base + first_linear * 8));
}

// 水平 pass 的一个 4 通道向量块：计算 (n,y,x) 处 channel_base 起的
// 4 个通道，写入 tmp 的对应位置。步骤：
// 1) acc = 0（__m256d，4 个 double lane 对应 4 个通道）；
// 2) 对每个 tap i：x 偏移按 replicate 夹取得到 xx，用
//    _mm256_broadcast_sd 把核系数 k[i] 复制到 4 lane，加载 4 个像素
//    后做 mul 再 add（顺序与 naive 相同，不收缩成 FMA）；
// 3) _mm256_storeu_pd 把 4 个结果写回 tmp。
// 单块 O(K)；整个水平 pass O(H*W*K)。
void HorizontalBlock(const Tensor& input, const ImageInfo& info,
                     const double* kernel, int radius, int64_t n, int64_t y,
                     int64_t x, int64_t channel_base, double* tmp) {
  const char* base = static_cast<const char*>(input.data());
  const int64_t tap_count = 2 * radius + 1;
  const int64_t block_base = ImageLinearIndex(info, n, y, x, channel_base);
  __m256d acc = _mm256_setzero_pd();
  for (int i = 0; i < tap_count; ++i) {
    const int64_t xx = ClampIndex(x + i - radius, info.w);
    const int64_t source = ImageLinearIndex(info, n, y, xx, channel_base);
    const __m256d k = _mm256_broadcast_sd(&kernel[i]);
    acc = _mm256_add_pd(
        acc, _mm256_mul_pd(k, LoadFourPixels(base, input.dtype(), source)));
  }
  _mm256_storeu_pd(tmp + block_base, acc);
}

// 垂直 pass 的一个 4 通道向量块：对 (n,y,x) 处 4 个通道沿 y 累加。
// 步骤与水平块相同，只是夹取方向换成 y，数据源换成 double 的 tmp；
// 累加结束后按输出 dtype 写出：
// - u8：先把 4 个 lane 存到栈上，逐个 std::round 再窄化为 uint8。
//   系数为正且已归一化，结果天然在 [0,255]，无需硬件饱和。
// - f32：_mm256_cvtpd_ps 压成 4 个 float，_mm_storeu_ps 写出。
// - f64：_mm256_storeu_pd 直接写 4 个 double。
// 单块 O(K)；整个垂直 pass O(H*W*K)。
void VerticalBlock(const double* tmp, const ImageInfo& info,
                   const double* kernel, int radius, int64_t n, int64_t y,
                   int64_t x, int64_t channel_base, Tensor* output) {
  const int64_t tap_count = 2 * radius + 1;
  __m256d acc = _mm256_setzero_pd();
  for (int i = 0; i < tap_count; ++i) {
    const int64_t yy = ClampIndex(y + i - radius, info.h);
    const int64_t source = ImageLinearIndex(info, n, yy, x, channel_base);
    const __m256d k = _mm256_broadcast_sd(&kernel[i]);
    acc = _mm256_add_pd(acc, _mm256_mul_pd(k, _mm256_loadu_pd(tmp + source)));
  }
  char* base = static_cast<char*>(output->data());
  if (output->dtype() == DataType::kUInt8) {
    double lanes[kLanes];
    _mm256_storeu_pd(lanes, acc);
    uint8_t* bytes = reinterpret_cast<uint8_t*>(base);
    const int64_t first = ImageLinearIndex(info, n, y, x, channel_base);
    for (int j = 0; j < kLanes; ++j) {
      bytes[first + j] = static_cast<uint8_t>(std::round(lanes[j]));
    }
  } else if (output->dtype() == DataType::kFloat32) {
    _mm_storeu_ps(reinterpret_cast<float*>(
                      base + ImageLinearIndex(info, n, y, x, channel_base) * 4),
                  _mm256_cvtpd_ps(acc));
  } else {
    _mm256_storeu_pd(
        reinterpret_cast<double*>(
            base + ImageLinearIndex(info, n, y, x, channel_base) * 8),
        acc);
  }
}

// AVX2 两趟实现。步骤：
// 1) 拒绝非连续张量；解析图像；不满足 Avx2Applicable 时整体回退到
//    BlurNaiveImpl（累加顺序相同，输出逐字节一致）；
// 2) vectorized = c - c % 4，即按 4 通道对齐的部分；
// 3) 水平 pass：遍历 (n,y,x)，每个位置先按 cb 以 4 通道为步长调用
//    HorizontalBlock，再把 [vectorized, c) 的尾部通道交给标量
//    BlurHorizontalElement；结果写入 double 的 tmp；
// 4) 垂直 pass：同样的遍历顺序与尾部分工，从 tmp 累加后写 output。
// 每 pass 复杂度 O(H*W*K)，额外内存 O(H*W*C) 个 double。
Status BlurAvx2Impl(const Tensor& input, const GaussianBlurParams& options,
                    Tensor* output) {
  if (!input.IsContiguous()) {
    return Status::Unsupported(
        "GaussianBlur AVX2 level requires contiguous tensors");
  }
  ImageInfo info;
  MYCV_RETURN_IF_ERROR(PrepareBlur(input, options, &info));
  if (!Avx2Applicable(info)) {
    return BlurNaiveImpl(input, options, output);
  }
  const std::vector<double> kernel =
      BuildGaussianKernel(options.kernel_size, options.sigma);
  const int radius = options.kernel_size / 2;
  const int64_t vectorized = info.c - info.c % kLanes;
  std::vector<double> tmp(static_cast<size_t>(output->Numel()));
  for (int64_t n = 0; n < info.n; ++n) {
    for (int64_t y = 0; y < info.h; ++y) {
      for (int64_t x = 0; x < info.w; ++x) {
        for (int64_t cb = 0; cb < vectorized; cb += kLanes) {
          HorizontalBlock(input, info, kernel.data(), radius, n, y, x, cb,
                          tmp.data());
        }
        for (int64_t c = vectorized; c < info.c; ++c) {
          BlurHorizontalElement(input, info, kernel.data(), radius,
                                ImageLinearIndex(info, n, y, x, c), tmp.data());
        }
      }
    }
  }
  for (int64_t n = 0; n < info.n; ++n) {
    for (int64_t y = 0; y < info.h; ++y) {
      for (int64_t x = 0; x < info.w; ++x) {
        for (int64_t cb = 0; cb < vectorized; cb += kLanes) {
          VerticalBlock(tmp.data(), info, kernel.data(), radius, n, y, x, cb,
                        output);
        }
        for (int64_t c = vectorized; c < info.c; ++c) {
          BlurVerticalElement(tmp.data(), info, kernel.data(), radius,
                              ImageLinearIndex(info, n, y, x, c), output);
        }
      }
    }
  }
  return Status::Ok();
}

}  // namespace

// 入口：把参数转成 GaussianBlurParams 并执行 AVX2 实现；
// 不适用或非连续时由 BlurAvx2Impl 回退到 naive，语义不变。
Status GaussianBlurCpuAvx2(const OpContext&, const Tensor& input, const Tensor*,
                           const void* params, Tensor* output) {
  return BlurAvx2Impl(input, *static_cast<const GaussianBlurParams*>(params),
                      output);
}

}  // namespace ops_internal
}  // namespace mycv
