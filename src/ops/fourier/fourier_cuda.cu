// CUDA implementation of the fourier operator
// (docs/architecture.md §6.1/§6.2).
//
// The 2-D transform is separable: each (n, c) plane is split into two
// independent 1-D passes, first by row (stride 1, length W), then by column
// (stride W, length H). One block handles each 1-D transform. Power-of-two
// lengths use an in-place radix-2 Cooley-Tukey butterfly within the block, with
// __syncthreads() between stages. Other lengths fall back to an out-of-place
// O(n^2) DFT using ping-pong buffers, accumulating in ascending j order to match
// the CPU.
//
// The precision policy matches the CPU: all arithmetic uses double, twiddle
// tables use the same -2*pi*k/n angles, and values are narrowed to the tensor
// dtype only during final write-back. Device-side FMA contraction can still
// prevent bit-identical host results, so cross-backend comparisons use the
// documented 1e-4 relative tolerance.

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

// Whether n is a power of two; only such lengths use the radix-2 kernel.
bool IsPowerOfTwo(int64_t n) {
  return n > 0 && (n & (n - 1)) == 0;
}

// One block per row/column: grid size is the line count capped at kMaxBlocks.
// Grid-stride loops inside the kernel cover taller/wider planes beyond that cap
// without exceeding the maximum grid dimension.
unsigned int LineBlocks(int64_t lines) {
  const int64_t capped = lines < kMaxBlocks ? lines : kMaxBlocks;
  return static_cast<unsigned int>(capped > 0 ? capped : 1);
}

// Reads a double from device input using an element offset; supports f32/f64.
__device__ inline double LoadValue(const void* base, int64_t offset,
                                   DataType dtype) {
  if (dtype == DataType::kFloat32) {
    return static_cast<double>(static_cast<const float*>(base)[offset]);
  }
  return static_cast<const double*>(base)[offset];
}

// Writes a double to a device tensor, narrowing to f32 or retaining f64 only here.
__device__ inline void StoreValue(void* base, int64_t offset, DataType dtype,
                                  double value) {
  if (dtype == DataType::kFloat32) {
    static_cast<float*>(base)[offset] = static_cast<float>(value);
    return;
  }
  static_cast<double*>(base)[offset] = value;
}

// Output tensors are newly allocated contiguous memory, so flat element indices
// can be computed directly from the layout formula (as in gaussian_blur)
// without per-element strides-based addressing.
__device__ inline int64_t ImageLinearOffset(const DeviceImage& image, int64_t n,
                                            int64_t y, int64_t x, int64_t c) {
  if (image.layout == ImageLayout::kNHWC) {
    return ((n * image.h + y) * image.w + x) * image.c + c;
  }
  return ((n * image.c + c) * image.h + y) * image.w + x;
}

// Pipeline step 1: copies the current (n, c) plane from the input tensor into
// double scratch space. The flat index is decoded as (y, x), and
// ImageElementOffset supports non-contiguous input views. re[index] stores the
// real part; im[index] is zero when has_imag is false (real-valued forward
// input). Each thread processes multiple elements with a grid-stride loop;
// elements are independent, so no synchronization is needed. Complexity O(H*W).
__global__ void FourierLoadPlaneKernel(
    DeviceTensor real_source, DeviceImage image, DeviceTensor imag_source,
    bool has_imag, double* re, double* im, int64_t n, int64_t c, int64_t count,
    DataType real_dtype, DataType imag_dtype) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t y = index / image.w;
    const int64_t x = index - y * image.w;
    re[index] = LoadValue(real_source.data,
                          ImageElementOffset(real_source, image, n, y, x, c),
                          real_dtype);
    im[index] =
        has_imag ? LoadValue(imag_source.data,
                             ImageElementOffset(imag_source, image, n, y, x, c),
                             imag_dtype)
                 : 0.0;
  }
}

// Final pipeline step: writes the transformed complex plane to the output
// tensor in O(H*W). Outputs are contiguous, so ImageLinearOffset computes flat
// offsets directly. Write-back uniformly applies `scale` (1/(H*W) for inverse,
// 1 for forward). When has_imag is false (inverse), only the real part is
// written; forward also scales the imaginary part and writes it to params.imag.
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

// Pipeline step 2 for power-of-two lengths: performs an in-place radix-2
// transform on each of `line_count` lines. One block handles each row/column,
// and threads divide samples by threadIdx.x. `stride` is the distance between
// adjacent complex samples (1 for rows, W for columns); `line_offset` is the
// distance between the first samples of adjacent lines (W for rows, 1 for
// columns). Sample i of line `line` is at origin + i*stride, where
// origin = line*line_offset.
//
// First perform bit-reversal swaps, then butterflies for len = 2,4,...,n:
//   a' = a + w*b, b' = a - w*b.
// Butterfly t belongs to group t/half, whose start is group = (t/half)*len;
// its within-group index is j = t - (t/half)*half, using twiddle table entry
// j*step (step = n/len). Butterflies in one stage read only the previous stage,
// so __syncthreads() is required between stages and once after bit-reversal.
// Complexity is O(n log n) per line and O(H*W*log) per plane.
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

// Pipeline step 2 fallback for non-power-of-two lengths: out-of-place direct
// DFT. Each thread handles one output frequency k, with the inner loop
// accumulating in ascending j = 0..n-1 order:
//   X[k] = sum_j x[j] * exp(-i*2*pi*k*j/N).
// This exactly matches the CPU fallback's addition order for stable
// cross-backend comparisons. Results go to out_re/out_im (ping-pong buffers)
// without overwriting the source. index represents (j*k) mod n, incremented by
// k per sample and wrapped with one subtraction. Complexity is O(n^2) per line
// and O(H*W*(H+W)) per plane. Outputs within a line are independent, so no
// block synchronization is needed.
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
      int64_t index = 0;  // (j*k) mod n; add k per sample and wrap by subtraction
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

// CUDA pipeline orchestration: the host only generates twiddle tables and
// launches kernels, performing no element-wise computation.
// Steps:
//  1. Obtain the stream, parse H/W/n/c from layout, and determine whether row
//     and column transforms use radix-2.
//  2. Generate four host twiddle-table segments [cos_h, sin_h, cos_w, sin_w]
//     with angle -2*pi*k/n, upload them in one cudaMemcpyAsync, and share them
//     across the entire image.
//  3. Allocate scratch space for 2*H*W f64 values (real/imaginary). If either
//     direction needs the DFT fallback, append another 2*H*W pair as ping-pong
//     buffers (buffers = 1 or 2).
//  4. Launch load -> rows -> cols -> store serially per plane: rows use stride
//     1 and length W; columns use stride W and length H. After a DFT fallback,
//     cur points to the other ping-pong half; otherwise transforms are in place.
//  5. Enqueue all launches on the same stream, then use FinishLaunch once to
//     check errors and synchronize.
// Only the store kernel applies inverse normalization by 1/(H*W), exactly once.
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

  // Twiddle table layout: [cos_h (h), sin_h (h), cos_w (w), sin_w (w)].
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
  cudaMemcpyAsync(table, host_table.data(), sizeof(double) * host_table.size(),
                  cudaMemcpyHostToDevice, stream);
  const double* cos_h = table;
  const double* sin_h = table + info.h;
  const double* cos_w = table + w_base;
  const double* sin_w = table + w_base + info.w;

  // If both directions use radix-2, transforms are in place and need one complex
  // plane (buffers = 1). If either uses the DFT fallback, reads and writes must
  // be separate, so allocate another pair as ping-pong buffers.
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
    // Rows: each line is one row, with stride 1 and length W.
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
    // Columns: each line is one column, with stride W and length H. Switch the
    // destination to a only when the row DFT has already written b, avoiding
    // simultaneous reads and writes to the same buffer.
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
