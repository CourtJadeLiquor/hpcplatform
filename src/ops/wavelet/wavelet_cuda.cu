// CUDA implementation of the wavelet operator (docs/architecture.md §6.1/§6.2).
//
// A single level uses a separable two-pass filter bank with periodic boundaries:
//   rows:  x[y, :]         -> low_rows[y, :], high_rows[y, :]  (W -> W/2)
//   cols:  low_rows[:, x]  -> LL[:, x], LH[:, x]
//          high_rows[:, x] -> HL[:, x], HH[:, x]              (H -> H/2)
// Forward kernel order: LoadPlane -> AnalyzeRows -> AnalyzeCols (writes four
// bands directly).
// Inverse kernel order: LoadBands -> SynthesizeCols -> SynthesizeRows ->
// StorePlane. Each output sample is computed independently by one thread, with
// no cross-sample dependencies and therefore no intra-block synchronization.
//
// Precision policy: accumulate in double using filters from host functions shared
// with the CPU, narrowing only on writeback. CUDA synthesis solves 2k+j = y
// (mod N) for coefficient indices and iterates over j, whereas the CPU accumulates
// over k. These different addition orders and device FMA contraction require the
// documented 1e-4 relative tolerance for cross-backend comparisons.

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "backends/cuda/cuda_device_utils.h"
#include "hpcplatform/ops/image.h"
#include "wavelet_internal.h"

namespace hpcplatform {
namespace ops_internal {
namespace {

// Filter bank passed to the device by value: taps is 2 (Haar) or 4 (Db4), with at
// most four entries, well below the kernel parameter limit. Passing by value
// avoids a separate device constant table and per-thread global-memory reads.
struct WaveletFilter {
  double low[4];
  double high[4];
  int taps;
};

// Build the small filter for the current call on the host using the shared
// WaveletLowPass/WaveletHighPass functions in O(taps) time; coefficients are
// bitwise-identical to those used by the CPU.
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

// Read a double from device input at an element offset; supports f32/f64.
__device__ inline double LoadValue(const void* base, int64_t offset,
                                   DataType dtype) {
  if (dtype == DataType::kFloat32) {
    return static_cast<double>(static_cast<const float*>(base)[offset]);
  }
  return static_cast<const double*>(base)[offset];
}

// Write a double back to a device tensor, narrowing to f32 or retaining f64 only
// here.
__device__ inline void StoreValue(void* base, int64_t offset, DataType dtype,
                                  double value) {
  if (dtype == DataType::kFloat32) {
    static_cast<float*>(base)[offset] = static_cast<float>(value);
    return;
  }
  static_cast<double*>(base)[offset] = value;
}

// Compute the flat index for a newly allocated contiguous output tensor directly
// from the layout formula.
__device__ inline int64_t ImageLinearOffset(const DeviceImage& image, int64_t n,
                                            int64_t y, int64_t x, int64_t c) {
  if (image.layout == ImageLayout::kNHWC) {
    return ((n * image.h + y) * image.w + x) * image.c + c;
  }
  return ((n * image.c + c) * image.h + y) * image.w + x;
}

// Forward step 1: load an (n, c) plane into a double scratch buffer in O(H*W)
// time. Decode flat index in row-major plane order as y = index / w and
// x = index - y*w; reads use ImageElementOffset, so the input may be a
// non-contiguous view. Each thread handles several elements independently, with
// no synchronization required.
__global__ void WaveletLoadPlaneKernel(DeviceTensor source, DeviceImage image,
                                       int64_t n, int64_t c, double* plane,
                                       int64_t count, DataType dtype) {
  for (int64_t index = ThreadIndex(); index < count; index += ThreadStride()) {
    const int64_t y = index / image.w;
    const int64_t x = index - y * image.w;
    plane[index] = LoadValue(
        source.data, ImageElementOffset(source, image, n, y, x, c), dtype);
  }
}

// Forward step 2 (row analysis): each thread handles one (y, k) and independently
// computes the kth low-pass/high-pass coefficients for that row:
//   low_rows[y,k]  = sum_j h[j] * x[y, (2k+j) mod W]
//   high_rows[y,k] = sum_j g[j] * x[y, (2k+j) mod W]
// Each output is written by only one thread, so no intra-block synchronization is
// needed. Periodic boundaries could use modulo, but 2*k + taps - 1 < 2*W whenever
// W >= 2, so a single subtraction suffices for wrap-around.
// count = H*(W/2); complexity: O(H*(W/2)*taps).
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

// Forward step 3 (column analysis): each thread handles one (yy, xx), filters both
// row buffers along columns, and writes the four subband tensors directly,
// avoiding a separate writeback:
//   LL[yy,xx] = sum_j h[j] * low_rows[(2yy+j) mod H, xx]
//   LH[yy,xx] = sum_j g[j] * low_rows[...] (row low-pass + column high-pass)
//   HL[yy,xx] = sum_j h[j] * high_rows[...] (row high-pass + column low-pass)
//   HH[yy,xx] = sum_j g[j] * high_rows[...]
// Periodic boundaries likewise use one subtraction; each thread independently
// writes four outputs without synchronization.
// count = (H/2)*(W/2); complexity: O((H/2)*(W/2)*taps).
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

// Inverse step 1: load the LL/LH/HL/HH subbands into contiguous double scratch
// buffers. Each thread handles one (yy, xx), with all four reads sharing the same
// element offset; input subbands may be non-contiguous views.
// count = (H/2)*(W/2); complexity: O(H*W/4).
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

// Inverse step 2 (column synthesis): reconstruct both row buffers from the four
// bands (H/2 -> H). Synthesis is
// out[y] = sum_{k,j} h[j]*low[k] + g[j]*high[k], where 2k + j = y (mod h).
// Instead, fix y and j and solve for k:
//   k = (y - j) / 2 (mod h/2).
// Because N is even, an integer k exists and is unique only when (y-j) is even;
// each output therefore needs taps iterations rather than h/2. Add h first when
// y-j is negative to wrap around. Each thread handles one (y, x) and writes one
// low_rows/high_rows element without synchronization.
// count = H*(W/2); complexity: O(H*(W/2)*taps).
__global__ void WaveletSynthesizeColsKernel(const double* ll, const double* lh,
                                            const double* hl, const double* hh,
                                            double* low_rows, double* high_rows,
                                            int64_t h, int64_t half_w,
                                            WaveletFilter filter,
                                            int64_t count) {
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

// Inverse step 3 (row synthesis): combine the two row buffers (W/2 -> W) into an
// image plane. As in column synthesis, solve k = (x-j)/2 (mod w/2) for each output
// x; the solution is valid only when x-j is even. Each thread independently writes
// one plane sample without synchronization.
// Complexity: O(H*W*taps). Iterating over j differs from the CPU's accumulation
// over k and falls within the documented 1e-4 relative tolerance.
__global__ void WaveletSynthesizeRowsKernel(
    const double* low_rows, const double* high_rows, double* plane, int64_t w,
    int64_t half_w, WaveletFilter filter, int64_t count) {
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

// Final inverse step: write the synthesized double plane to the corresponding
// (n, c) plane of the output tensor. The output is contiguous, so compute offsets
// directly with ImageLinearOffset; narrow dtype only here.
// Complexity: O(H*W).
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

// CUDA pipeline orchestration: the host only launches kernels and performs no
// element-wise computation.
// Steps:
//  1. Get the stream and parse layout; forward obtains H/W from the image, while
//     inverse doubles the subband dimensions;
//  2. Allocate one 2*H*W f64 scratch buffer and partition it by direction (see
//     the comments below);
//  3. Build a by-value WaveletFilter from the shared host coefficients;
//  4. Launch three forward or four inverse kernels serially for each plane, all on
//     the same stream;
//  5. Call FinishLaunch once at the end to check errors and synchronize.
// Inverse does not use the entry point's `other` (the four bands use extra_in), so
// that parameter is unnamed.
Status WaveletCuda(const OpContext& context, const Tensor& input, const Tensor*,
                   const void* params, Tensor* output) {
  auto stream_result = GetCudaStream(context);
  if (!stream_result.ok()) return stream_result.status();
  cudaStream_t stream = stream_result.value()->handle();
  const auto& options = *static_cast<const WaveletParams*>(params);
  auto parsed = ParseImage(input.shape(), options.layout);
  if (!parsed.ok()) return parsed.status();

  // Forward parses image dimensions; inverse parses subband dimensions and
  // doubles both directions.
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

  // One f64 scratch buffer holds all intermediate planes for the current direction:
  //   Forward: plane (H*W) + low_rows/high_rows (H*W/2 each)
  //   Inverse: LL/LH/HL/HH (H*W/4 each) + low_rows/high_rows (H*W/2 each)
  // Both view groups start at the same address (plane and ll both point to
  // scratch_base) because the two directions are never used simultaneously.
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
                                 stream>>>(plane, low_rows, high_rows, w,
                                           half_w, filter, row_elems);
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
          options.extra_in[2]->data(), ll, lh, hl, hh, n, c, half_w, band_elems,
          input.dtype());
      WaveletSynthesizeColsKernel<<<BlockCount(row_elems), kThreadsPerBlock, 0,
                                    stream>>>(
          ll, lh, hl, hh, low_rows, high_rows, h, half_w, filter, row_elems);
      WaveletSynthesizeRowsKernel<<<BlockCount(plane_size), kThreadsPerBlock, 0,
                                    stream>>>(low_rows, high_rows, plane, w,
                                              half_w, filter, plane_size);
      WaveletStorePlaneKernel<<<BlockCount(plane_size), kThreadsPerBlock, 0,
                                stream>>>(plane, output->data(),
                                          Describe(image_info), n, c,
                                          plane_size, output->dtype());
    }
  }
  return FinishLaunch(stream, "Wavelet kernel launch");
}

}  // namespace ops_internal
}  // namespace hpcplatform
