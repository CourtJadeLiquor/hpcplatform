#ifndef MYCV_SRC_OPS_WAVELET_WAVELET_INTERNAL_H_
#define MYCV_SRC_OPS_WAVELET_WAVELET_INTERNAL_H_

// wavelet 算子目录的内部契约（docs/architecture.md §6.1）。
// 入口负责全部分配；后端只读源平面，并按本参数结构写目标：
//
//   正向  input = 图像   output = LL   extra_out = {LH, HL, HH}
//   逆向  input = LL     output = 图像  extra_in = {LH, HL, HH}
//
// 单层二维 DWT 可分离：先逐行、再逐列做一维滤波器组，每个方向降采样 2，
// 于是 H x W 变成四个 H/2 x W/2 子带：
//   LL = 行低通 + 列低通（近似分量，可继续分解）
//   LH = 行低通 + 列高通（水平细节）
//   HL = 行高通 + 列低通（垂直细节）
//   HH = 行高通 + 列高通（对角细节）
// 边界统一按周期（wrap-around）处理。

#include <cmath>
#include <vector>

#include "framework/image_utils.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"
#include "mycv/ops/wavelet.h"

namespace mycv {
namespace ops_internal {

struct WaveletParams {
  WaveletKind kind = WaveletKind::kHaar;    // 小波基：Haar 或 Db4
  ImageLayout layout = ImageLayout::kNHWC;  // 输入/输出布局
  bool inverse = false;  // false 分析（分解），true 合成
  // 正向：LH/HL/HH 三个输出槽位；逆向：对应的三个输入子带。
  // LL 始终走 output（正向）或 input（逆向），因此不在这里重复。
  const Tensor* extra_in[3] = {nullptr, nullptr, nullptr};
  Tensor* extra_out[3] = {nullptr, nullptr, nullptr};
};

// 分析（分解）低通滤波器系数：
//   Haar: h = [1/sqrt2, 1/sqrt2]
//   Db4 : h = [(1+sqrt3)/(4*sqrt2), (3+sqrt3)/(4*sqrt2),
//              (3-sqrt3)/(4*sqrt2), (1-sqrt3)/(4*sqrt2)]
// 高通由 g[k] = (-1)^k * h[taps-1-k] 导出。系数在此只算一次，CPU 与
// CUDA 共用，保证各后端看到逐位相同的滤波器。
inline std::vector<double> WaveletLowPass(WaveletKind kind) {
  const double kSqrt2 = std::sqrt(2.0);
  if (kind == WaveletKind::kHaar) {
    return {1.0 / kSqrt2, 1.0 / kSqrt2};
  }
  const double kSqrt3 = std::sqrt(3.0);
  return {(1.0 + kSqrt3) / (4.0 * kSqrt2), (3.0 + kSqrt3) / (4.0 * kSqrt2),
          (3.0 - kSqrt3) / (4.0 * kSqrt2), (1.0 - kSqrt3) / (4.0 * kSqrt2)};
}

// 由低通导出正交镜像高通：g[k] = (-1)^k * h[taps-1-k]，复杂度 O(taps)。
// 分析与合成复用同一对系数（paraunitary），因此重建误差只来自浮点舍入。
inline std::vector<double> WaveletHighPass(const std::vector<double>& low) {
  const size_t taps = low.size();
  std::vector<double> high(taps);
  for (size_t k = 0; k < taps; ++k) {
    const double sign = (k % 2 == 0) ? 1.0 : -1.0;
    high[k] = sign * low[taps - 1 - k];
  }
  return high;
}

// 各后端入口签名统一为 (context, input, other, params, output)。
// 张量由入口统一分配；正向写 output 与 extra_out，逆向读 extra_in。
// `other` 未被本算子使用，保留它是为了与其他算子共用内核签名。

// CPU 参考实现：顺序执行所有行、列，是其他级别的逐字节基线。
Status WaveletCpuNaive(const OpContext& context, const Tensor& input,
                       const Tensor* other, const void* params, Tensor* output);
#ifdef MYCV_WITH_OPENMP
// CPU OpenMP 级别：只把互相独立的行/列分给线程，累加顺序不变。
Status WaveletCpuOpenMP(const OpContext& context, const Tensor& input,
                        const Tensor* other, const void* params,
                        Tensor* output);
#endif
#ifdef MYCV_WITH_TBB
// CPU TBB 级别：同一骨架，用 oneTBB 的 parallel_for 调度行/列。
Status WaveletCpuTbb(const OpContext& context, const Tensor& input,
                     const Tensor* other, const void* params, Tensor* output);
#endif
#ifdef MYCV_WITH_CUDA
// CUDA 后端：每个输出样本一个线程，整条流水线由设备内核完成。
Status WaveletCuda(const OpContext& context, const Tensor& input,
                   const Tensor* other, const void* params, Tensor* output);
#endif

// 把本二进制内编译进来的实现注册到算子注册表。OpenCV 没有对应变换，
// 查找时返回 kNotImplemented，而不是悄悄用主机代码跑设备内存。
void RegisterWaveletKernels();

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_WAVELET_WAVELET_INTERNAL_H_
