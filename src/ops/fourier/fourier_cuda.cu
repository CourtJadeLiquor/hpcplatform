// fourier 算子的 CUDA 实现（docs/architecture.md §6.1/§6.2）。
//
// 二维变换可分离，每个 (n, c) 平面拆成两次独立的一维变换：先逐行
// （stride 1、长度 W），再逐列（stride W、长度 H）。一次一维变换固定由
// 一个 block 负责：长度为 2 的幂时在 block 内就地做 radix-2 的
// Cooley-Tukey 蝶形，级与级之间用 __syncthreads() 同步；其他长度回退到
// out-of-place 的 O(n^2) DFT，写入乒乓缓冲，累加保持 j 升序以与 CPU 一致。
//
// 精度策略与 CPU 相同：所有算术在 double 中完成，twiddle 表使用相同的
// -2*pi*k/n 角度，仅在最后写回时窄化到张量 dtype。设备端仍可能做 FMA
// 收缩，与主机结果并非逐位相同，因此跨后端对拍使用文档约定的 1e-4
// 相对容差。

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "backends/cuda/cuda_device_utils.h"
#include "fourier_internal.h"
#include "mycv/ops/image.h"

namespace mycv {
namespace ops_internal {
namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// 是否为 2 的幂；只有这种长度才走 radix-2 内核。
bool IsPowerOfTwo(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

// 每行/列一个 block：grid 大小取线数，但封顶到 kMaxBlocks；内核内部用
// grid-stride 循环兜住超出上限的超高/超宽平面，避免超过最大 grid 维度。
unsigned int LineBlocks(int64_t lines) {
  const int64_t capped = lines < kMaxBlocks ? lines : kMaxBlocks;
  return static_cast<unsigned int>(capped > 0 ? capped : 1);
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

// 输出张量是新分配的连续内存，因此扁平元素下标可直接按布局公式计算
// （与 gaussian_blur 的假设相同），无需走 strides 的逐元素寻址。
__device__ inline int64_t ImageLinearOffset(const DeviceImage& image, int64_t n,
                                            int64_t y, int64_t x, int64_t c) {
  if (image.layout == ImageLayout::kNHWC) {
    return ((n * image.h + y) * image.w + x) * image.c + c;
  }
  return ((n * image.c + c) * image.h + y) * image.w + x;
}

// 流水线第 1 步：把当前 (n, c) 平面从输入张量搬到 double 暂存区。
// 扁平下标 index 解码为 (y, x)，用 ImageElementOffset 支持非连续的输入
// 视图；re[index] 存实部，has_imag 为 false（正向实值输入）时 im[index]
// 直接填 0。每个线程通过 grid-stride 处理若干元素，元素之间没有依赖，
// 因此不需要任何同步。复杂度 O(H*W)。
__global__ void FourierLoadPlaneKernel(DeviceTensor real_source,
                                       DeviceImage image,
                                       DeviceTensor imag_source, bool has_imag,
                                       double* re, double* im, int64_t n,
                                       int64_t c, int64_t count,
                                       DataType real_dtype,
                                       DataType imag_dtype) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t y = index / image.w;
    const int64_t x = index - y * image.w;
    re[index] = LoadValue(
        real_source.data,
        ImageElementOffset(real_source, image, n, y, x, c), real_dtype);
    im[index] = has_imag
                    ? LoadValue(imag_source.data,
                                ImageElementOffset(imag_source, image, n, y, x, c),
                                imag_dtype)
                    : 0.0;
  }
}

// 流水线最后一步：把变换后的复数平面写回输出张量，复杂度 O(H*W)。
// 输出是连续张量，用 ImageLinearOffset 直接算扁平偏移；写回时统一乘
// `scale`（逆向为 1/(H*W)，正向为 1）。has_imag 为 false（逆向）时只写
// 实部，正向则把虚部同样缩放后写入 params.imag。
__global__ void FourierStorePlaneKernel(const double* re, const double* im,
                                        void* real_output, void* imag_output,
                                        bool has_imag, DeviceImage image,
                                        int64_t n, int64_t c, int64_t count,
                                        double scale, DataType real_dtype,
                                        DataType imag_dtype) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t y = index / image.w;
    const int64_t x = index - y * image.w;
    const int64_t offset = ImageLinearOffset(image, n, y, x, c);
    StoreValue(real_output, offset, real_dtype, re[index] * scale);
    if (has_imag) {
      StoreValue(imag_output, offset, imag_dtype, im[index] * scale);
    }
  }
}

// 流水线第 2 步（长度为 2 的幂）：对 `line_count` 条线各做一次就地
// radix-2 变换。每行/每列固定由一个 block 负责，块内线程按 threadIdx.x
// 分摊样本。`stride` 是相邻复数样本的距离（行 1、列 W）；`line_offset`
// 是相邻两条线首样本的距离（行 W、列 1）。于是第 line 条线的第 i 个样本
// 位于 origin + i*stride，其中 origin = line*line_offset。
//
// 先做 bit-reversal 交换，再按 len = 2,4,...,n 逐级蝶形：
//   a' = a + w*b、b' = a - w*b。
// 第 t 个蝶形属于第 t/half 组，组起点 group = (t/half)*len，组内下标
// j = t - (t/half)*half；twiddle 取表项 j*step（step = n/len）。同一级
// 的蝶形只读上一级结果，所以级与级之间必须 __syncthreads()；bit-reversal
// 之后也要同步一次。每线 O(n log n)，每平面 O(H*W*log)。
__global__ void FourierRadix2Kernel(double* re, double* im, int64_t n,
                                    int64_t stride, int64_t line_offset,
                                    int64_t line_count, const double* cos_table,
                                    const double* sin_table, double sign) {
  for (int64_t line = blockIdx.x; line < line_count; line += gridDim.x) {
    const int64_t origin = line * line_offset;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
      int64_t reversed = 0;
      for (int64_t value = i, bit = n >> 1; bit > 0; value >>= 1, bit >>= 1) {
        if (value & 1) reversed |= bit;
      }
      if (i < reversed) {
        const int64_t a = origin + i * stride;
        const int64_t b = origin + reversed * stride;
        const double tr = re[a];
        re[a] = re[b];
        re[b] = tr;
        const double ti = im[a];
        im[a] = im[b];
        im[b] = ti;
      }
    }
    __syncthreads();
    for (int64_t len = 2; len <= n; len <<= 1) {
      const int64_t half = len >> 1;
      const int64_t step = n / len;
      for (int64_t t = threadIdx.x; t < (n >> 1); t += blockDim.x) {
        const int64_t group = (t / half) * len;
        const int64_t j = t - (t / half) * half;
        const double wr = cos_table[j * step];
        const double wi = sign * sin_table[j * step];
        const int64_t a = origin + (group + j) * stride;
        const int64_t b = origin + (group + j + half) * stride;
        const double vr = re[b] * wr - im[b] * wi;
        const double vi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - vr;
        im[b] = im[a] - vi;
        re[a] += vr;
        im[a] += vi;
      }
      __syncthreads();
    }
  }
}

// 流水线第 2 步的回退（长度不是 2 的幂）：out-of-place 的直接 DFT。
// 每个线程负责一个输出频率 k，内层按 j = 0..n-1 升序累加
//   X[k] = sum_j x[j] * exp(-i*2*pi*k*j/N)，
// 与 CPU 回退的加法顺序完全一致，因此跨后端对拍稳定。结果写入
// out_re/out_im（乒乓缓冲），源数据不会被覆盖；index 表示 (j*k) mod n，
// 每取一个样本自增 k 并用一次减法回绕。每线 O(n^2)、每平面
// O(H*W*(H+W))；同一线的各输出互不依赖，不需要块内同步。
__global__ void FourierDftKernel(const double* in_re, const double* in_im,
                                 double* out_re, double* out_im, int64_t n,
                                 int64_t stride, int64_t line_offset,
                                 int64_t line_count, const double* cos_table,
                                 const double* sin_table, double sign) {
  for (int64_t line = blockIdx.x; line < line_count; line += gridDim.x) {
    const int64_t origin = line * line_offset;
    for (int64_t k = threadIdx.x; k < n; k += blockDim.x) {
      double sum_re = 0.0;
      double sum_im = 0.0;
      int64_t index = 0;  // (j*k) mod n，每取一个样本自增 k 并减法回绕
      for (int64_t j = 0; j < n; ++j) {
        const double wr = cos_table[index];
        const double wi = sign * sin_table[index];
        const double xr = in_re[origin + j * stride];
        const double xi = in_im[origin + j * stride];
        sum_re += xr * wr - xi * wi;
        sum_im += xr * wi + xi * wr;
        index += k;
        if (index >= n) index -= n;
      }
      out_re[origin + k * stride] = sum_re;
      out_im[origin + k * stride] = sum_im;
    }
  }
}

}  // namespace

// CUDA 流水线编排：主机侧只生成 twiddle 表并发射内核，不做逐元素计算。
// 步骤：
//  1. 取 stream，按 layout 解析 H/W/n/c，判断行、列是否 radix-2；
//  2. 在主机生成 [cos_h, sin_h, cos_w, sin_w] 四段 twiddle 表，角度为
//     -2*pi*k/n，一次 cudaMemcpyAsync 上传，整张图共享；
//  3. 申请 2*H*W 个 f64（实部/虚部）暂存区；只要任一方向需要 DFT 回退，
//     再追加一对 2*H*W 的乒乓缓冲（buffers = 1 或 2）；
//  4. 逐平面串行发射 load -> rows -> cols -> store：行用 stride 1、长度 W，
//     列用 stride W、长度 H；DFT 回退后 cur 指向乒乓的另一半，否则就地；
//  5. 所有发射进入同一 stream，最后统一 FinishLaunch 检查错误并同步。
// 逆向归一化 1/(H*W) 只由 store 内核乘一次。
Status FourierCuda(const OpContext& context, const Tensor& input,
                   const Tensor* other, const void* params, Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  cudaStream_t stream = stream_result.value()->handle();
  const auto& options = *static_cast<const FourierParams*>(params);
  auto parsed = ParseImage(input.shape(), options.layout);
  if (!parsed.ok()) return parsed.status();
  const ImageInfo& info = parsed.value();

  const int64_t plane_size = info.h * info.w;
  const bool radix_h = IsPowerOfTwo(info.h);
  const bool radix_w = IsPowerOfTwo(info.w);
  const double sign = options.inverse ? -1.0 : 1.0;
  const double scale =
      options.inverse
          ? 1.0 / (static_cast<double>(info.h) * static_cast<double>(info.w))
          : 1.0;

  // twiddle 表布局：[cos_h (h), sin_h (h), cos_w (w), sin_w (w)]。
  const int64_t w_base = 2 * info.h;
  std::vector<double> host_table(static_cast<size_t>(2 * (info.h + info.w)));
  for (int64_t k = 0; k < info.h; ++k) {
    const double angle =
        -kTwoPi * static_cast<double>(k) / static_cast<double>(info.h);
    host_table[static_cast<size_t>(k)] = std::cos(angle);
    host_table[static_cast<size_t>(info.h + k)] = std::sin(angle);
  }
  for (int64_t k = 0; k < info.w; ++k) {
    const double angle =
        -kTwoPi * static_cast<double>(k) / static_cast<double>(info.w);
    host_table[static_cast<size_t>(w_base + k)] = std::cos(angle);
    host_table[static_cast<size_t>(w_base + info.w + k)] = std::sin(angle);
  }

  double* table = nullptr;
  if (cudaMalloc(&table, sizeof(double) * host_table.size()) != cudaSuccess) {
    return CudaStatus(cudaGetLastError(), "cudaMalloc for FFT twiddle table");
  }
  cudaMemcpyAsync(table, host_table.data(),
                  sizeof(double) * host_table.size(), cudaMemcpyHostToDevice,
                  stream);
  const double* cos_h = table;
  const double* sin_h = table + info.h;
  const double* cos_w = table + w_base;
  const double* sin_w = table + w_base + info.w;

  // 两个方向都是 radix-2 时就地变换即可，只需一组复数平面（buffers = 1）；
  // 任一方向要走 DFT 回退时，读、写必须分离，于是再申请一组乒乓缓冲。
  const int64_t buffers = (radix_h && radix_w) ? 1 : 2;
  auto scratch = context.context->AllocateTensor(
      Shape{buffers * 2 * plane_size}, DataType::kFloat64);
  if (!scratch.ok()) {
    cudaFree(table);
    return scratch.status();
  }
  double* a_re = static_cast<double*>(scratch.value().data());
  double* a_im = a_re + plane_size;
  double* b_re = nullptr;
  double* b_im = nullptr;
  if (buffers == 2) {
    b_re = a_re + 2 * plane_size;
    b_im = b_re + plane_size;
  }

  const unsigned int plane_blocks = BlockCount(plane_size);
  const int64_t planes = info.n * info.c;
  for (int64_t p = 0; p < planes; ++p) {
    const int64_t n = p / info.c;
    const int64_t c = p % info.c;
    FourierLoadPlaneKernel<<<plane_blocks, kThreadsPerBlock, 0, stream>>>(
        Describe(input), Describe(info),
        other == nullptr ? DeviceTensor{} : Describe(*other), other != nullptr,
        a_re, a_im, n, c, plane_size, input.dtype(),
        other == nullptr ? input.dtype() : other->dtype());

    double* cur_re = a_re;
    double* cur_im = a_im;
    // 行：每条线是一行，stride 1，长度 W。
    if (radix_w) {
      FourierRadix2Kernel<<<LineBlocks(info.h), kThreadsPerBlock, 0, stream>>>(
          cur_re, cur_im, info.w, 1, info.w, info.h, cos_w, sin_w, sign);
    } else {
      FourierDftKernel<<<LineBlocks(info.h), kThreadsPerBlock, 0, stream>>>(
          cur_re, cur_im, b_re, b_im, info.w, 1, info.w, info.h, cos_w, sin_w,
          sign);
      cur_re = b_re;
      cur_im = b_im;
    }
    // 列：每条线是一列，stride W，长度 H。只有行方向的 DFT 已经写入 b 时
    // 才把目标换成 a，避免同一缓冲同时读、写。
    if (radix_h) {
      FourierRadix2Kernel<<<LineBlocks(info.w), kThreadsPerBlock, 0, stream>>>(
          cur_re, cur_im, info.h, info.w, 1, info.w, cos_h, sin_h, sign);
    } else {
      double* dst_re = (cur_re == a_re) ? b_re : a_re;
      double* dst_im = (cur_im == a_im) ? b_im : a_im;
      FourierDftKernel<<<LineBlocks(info.w), kThreadsPerBlock, 0, stream>>>(
          cur_re, cur_im, dst_re, dst_im, info.h, info.w, 1, info.w, cos_h,
          sin_h, sign);
      cur_re = dst_re;
      cur_im = dst_im;
    }
    FourierStorePlaneKernel<<<plane_blocks, kThreadsPerBlock, 0, stream>>>(
        cur_re, cur_im, output->data(),
        options.imag == nullptr ? nullptr : options.imag->data(),
        options.imag != nullptr, Describe(info), n, c, plane_size, scale,
        output->dtype(),
        options.imag == nullptr ? output->dtype() : options.imag->dtype());
  }

  const Status status = FinishLaunch(stream, "Fourier kernel launch");
  cudaFree(table);
  return status;
}

}  // namespace ops_internal
}  // namespace mycv
