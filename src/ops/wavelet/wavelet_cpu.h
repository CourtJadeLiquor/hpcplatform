#ifndef hpcplatform_SRC_OPS_WAVELET_WAVELET_CPU_H_
#define hpcplatform_SRC_OPS_WAVELET_WAVELET_CPU_H_

// Shared CPU skeleton for the wavelet operator (docs/architecture.md §6.1/§7.2).
// A single level uses a separable two-pass filter bank with periodic
// (wrap-around) boundaries:
//   rows:   x[y, :]         -> low_rows[y, :], high_rows[y, :]  (W -> W/2)
//   cols:   low_rows[:, x]  -> LL[:, x], LH[:, x]
//           high_rows[:, x] -> HL[:, x], HH[:, x]              (H -> H/2)
// 1D analysis: low[k]  = sum_j h[j] * x[(2k+j) mod N]
//              high[k] = sum_j g[j] * x[(2k+j) mod N]
// 1D synthesis: out[(2k+j) mod N] += h[j]*low[k] + g[j]*high[k]
// Here h is the low-pass filter (Haar or Db4), and
// g[k] = (-1)^k * h[taps-1-k].
//
// The Runner policy template distributes independent rows/columns across
// parallel backends. Each line accumulates in fixed ascending k and j order,
// independent of threads, so all three CPU levels produce bit-exact output.
// All accumulation uses double and narrows to the tensor dtype only on writeback.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef hpcplatform_WITH_TBB
#include <tbb/parallel_for.h>
#endif

#include "wavelet_internal.h"

namespace hpcplatform {
namespace ops_internal {

// Execution policy: each translation unit selects exactly one, keeping the
// shared plane driver as a single template. Each index passed to
// For(begin, end, fn) represents an independent row or column; fn does not
// modify shared state.
struct WaveletSequentialRunner {
  template <typename Fn>
  static void For(int64_t begin, int64_t end, Fn&& fn) {
    for (int64_t i = begin; i < end; ++i)
      fn(i);
  }
};

#ifdef hpcplatform_WITH_OPENMP
// OpenMP policy: distributes independent rows/columns with omp parallel for,
// using the default static scheduling.
struct WaveletOpenMPRunner {
  template <typename Fn>
  static void For(int64_t begin, int64_t end, Fn&& fn) {
#pragma omp parallel for
    for (int64_t i = begin; i < end; ++i)
      fn(i);
  }
};
#endif

#ifdef hpcplatform_WITH_TBB
// TBB policy: parallel_for automatically partitions work and balances load.
struct WaveletTbbRunner {
  template <typename Fn>
  static void For(int64_t begin, int64_t end, Fn&& fn) {
    tbb::parallel_for(begin, end, [&](int64_t i) { fn(i); });
  }
};
#endif

// Analysis low-pass/high-pass coefficients come from WaveletLowPass and
// WaveletHighPass in wavelet_internal.h. The CUDA translation unit reuses the
// same definitions to ensure identical coefficients.

// Perform 1D analysis (decomposition) on `n` samples, producing n/2 low-pass and
// high-pass coefficients:
//   low[k]  = sum_j h[j] * x[(2k+j) mod n]
//   high[k] = sum_j g[j] * x[(2k+j) mod n]
// `src_stride`, `low_stride`, and `high_stride` are measured in samples, allowing
// the same function to handle contiguous rows (stride 1) and strided row buffers
// (stride W/2). Modulo indexing implements periodic boundaries; ascending k and j
// accumulation keeps results thread-independent. Complexity: O(n*taps).
inline void AnalyzeSequence(const double* src, int64_t src_stride, int64_t n,
                            const std::vector<double>& low_filter,
                            const std::vector<double>& high_filter, double* low,
                            int64_t low_stride, double* high,
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

// 1D synthesis (reconstruction): clear the output, then upsample and accumulate
// each coefficient:
//   out[(2k+j) mod n] += h[j]*low[k] + g[j]*high[k]
// Accumulate in ascending k and j order, matching the fixed analysis order;
// boundaries are periodic. Multiple (k, j) pairs can map to the same output
// sample, so accumulation order affects the low bits and must remain fixed.
// Complexity: O(n*taps).
inline void SynthesizeSequence(const double* low, int64_t low_stride,
                               const double* high, int64_t high_stride,
                               int64_t n, const std::vector<double>& low_filter,
                               const std::vector<double>& high_filter,
                               double* output, int64_t output_stride) {
  for (int64_t i = 0; i < n; ++i)
    output[i * output_stride] = 0.0;
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

// One 2D analysis pass: rows first, then columns.
//   Rows: x[y,:]         -> low_rows[y,:], high_rows[y,:] (W -> W/2)
//   Cols: low_rows[:,x]  -> LL[:,x], LH[:,x]
//         high_rows[:,x] -> HL[:,x], HH[:,x] (H -> H/2)
// Rows and columns are mutually independent; Runner only decides which thread
// handles each line, so all three CPU levels are bit-exact. The intermediate
// low_rows/high_rows buffers each contain H*(W/2) doubles.
// Complexity: O(H*W*taps).
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

// 2D synthesis: the exact reverse of AnalyzePlane, columns first and then rows.
//   Cols: LL/LH -> low_rows, HL/HH -> high_rows (H/2 -> H)
//   Rows: low_rows/high_rows -> plane (W/2 -> W)
// Intermediate buffers have the same sizes as in AnalyzePlane.
// Complexity: O(H*W*taps).
template <typename Runner>
inline void SynthesizePlane(const double* ll, const double* lh,
                            const double* hl, const double* hh, int64_t h,
                            int64_t w, const std::vector<double>& low_filter,
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

// Read a double value from an input view at a byte offset; supports f32/f64.
inline double LoadWaveletValue(const char* source, DataType dtype) {
  if (dtype == DataType::kFloat32) {
    return *reinterpret_cast<const float*>(source);
  }
  return *reinterpret_cast<const double*>(source);
}

// Write a double back to the output tensor, narrowing to f32 or retaining f64
// only here.
inline void StoreWaveletValue(char* destination, DataType dtype, double value) {
  if (dtype == DataType::kFloat32) {
    *reinterpret_cast<float*>(destination) = static_cast<float>(value);
    return;
  }
  *reinterpret_cast<double*>(destination) = value;
}

// Load an (n, c) plane into a contiguous double scratch buffer in O(H*W) time.
// Reads use ImageByteOffset, so the source tensor may be a non-contiguous view.
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

// Write a plane back to the output tensor in O(H*W) time. The entry point
// allocates the output as contiguous memory.
inline void StoreWaveletPlane(const std::vector<double>& plane,
                              const ImageInfo& info, int64_t n, int64_t c,
                              Tensor* destination) {
  char* base = static_cast<char*>(destination->data());
  const DataType dtype = destination->dtype();
  for (int64_t y = 0; y < info.h; ++y) {
    for (int64_t x = 0; x < info.w; ++x) {
      StoreWaveletValue(base + ImageByteOffset(*destination, info, n, y, x, c),
                        dtype, plane[static_cast<size_t>(y * info.w + x)]);
    }
  }
}

// Driver shared by all CPU levels. `input` is the image in the forward direction
// and LL in the inverse direction, so plane dimensions differ: derive ImageInfo
// for the image/subbands from the direction rather than the tensor alone.
// Each (n, c) plane is processed independently: forward performs load ->
// AnalyzePlane -> write four subbands; inverse performs load four subbands ->
// SynthesizePlane -> write image.
// Complexity: O(H*W*taps).
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
}  // namespace hpcplatform

#endif  // hpcplatform_SRC_OPS_WAVELET_WAVELET_CPU_H_
