#ifndef MYCV_SRC_OPS_WAVELET_WAVELET_INTERNAL_H_
#define MYCV_SRC_OPS_WAVELET_WAVELET_INTERNAL_H_

// Internal contract for the wavelet operator directory (docs/architecture.md §6.1).
// The entry point performs all allocation; backends only read source planes and
// write destinations according to this parameter structure:
//
//   Forward  input = image  output = LL    extra_out = {LH, HL, HH}
//   Inverse  input = LL     output = image extra_in = {LH, HL, HH}
//
// A single-level 2D DWT is separable: apply 1D filter banks row-wise and then
// column-wise, downsampling each direction by 2. H x W therefore becomes four
// H/2 x W/2 subbands:
//   LL = row low-pass + column low-pass (approximation; can be decomposed further)
//   LH = row low-pass + column high-pass (horizontal detail)
//   HL = row high-pass + column low-pass (vertical detail)
//   HH = row high-pass + column high-pass (diagonal detail)
// All boundaries use periodic (wrap-around) extension.

#include <cmath>
#include <vector>

#include "framework/image_utils.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"
#include "mycv/ops/wavelet.h"

namespace mycv {
namespace ops_internal {

struct WaveletParams {
  WaveletKind kind = WaveletKind::kHaar;    // Wavelet basis: Haar or Db4
  ImageLayout layout = ImageLayout::kNHWC;  // Input/output layout
  bool inverse = false;  // false: analysis (decomposition); true: synthesis
  // Forward: three output slots for LH/HL/HH; inverse: the corresponding inputs.
  // LL always uses output (forward) or input (inverse), so it is not repeated here.
  const Tensor* extra_in[3] = {nullptr, nullptr, nullptr};
  Tensor* extra_out[3] = {nullptr, nullptr, nullptr};
};

// Analysis (decomposition) low-pass filter coefficients:
//   Haar: h = [1/sqrt2, 1/sqrt2]
//   Db4 : h = [(1+sqrt3)/(4*sqrt2), (3+sqrt3)/(4*sqrt2),
//              (3-sqrt3)/(4*sqrt2), (1-sqrt3)/(4*sqrt2)]
// The high-pass filter is derived as g[k] = (-1)^k * h[taps-1-k]. Coefficients
// are computed once here and shared by CPU and CUDA so every backend uses
// bitwise-identical filters.
inline std::vector<double> WaveletLowPass(WaveletKind kind) {
  const double kSqrt2 = std::sqrt(2.0);
  if (kind == WaveletKind::kHaar) {
    return {1.0 / kSqrt2, 1.0 / kSqrt2};
  }
  const double kSqrt3 = std::sqrt(3.0);
  return {(1.0 + kSqrt3) / (4.0 * kSqrt2), (3.0 + kSqrt3) / (4.0 * kSqrt2),
          (3.0 - kSqrt3) / (4.0 * kSqrt2), (1.0 - kSqrt3) / (4.0 * kSqrt2)};
}

// Derive the quadrature-mirror high-pass filter from the low-pass filter:
// g[k] = (-1)^k * h[taps-1-k], in O(taps) time. Analysis and synthesis reuse
// the same paraunitary coefficient pair, so reconstruction error comes only
// from floating-point rounding.
inline std::vector<double> WaveletHighPass(const std::vector<double>& low) {
  const size_t taps = low.size();
  std::vector<double> high(taps);
  for (size_t k = 0; k < taps; ++k) {
    const double sign = (k % 2 == 0) ? 1.0 : -1.0;
    high[k] = sign * low[taps - 1 - k];
  }
  return high;
}

// All backend entry points use (context, input, other, params, output).
// The entry point allocates all tensors; forward writes output and extra_out,
// while inverse reads extra_in. `other` is unused by this operator but retained
// to share the kernel signature with other operators.

// CPU reference implementation: processes all rows and columns sequentially and
// serves as the byte-exact baseline for the other levels.
Status WaveletCpuNaive(const OpContext& context, const Tensor& input,
                       const Tensor* other, const void* params, Tensor* output);
#ifdef MYCV_WITH_OPENMP
// CPU OpenMP level: distributes only independent rows/columns across threads,
// preserving the accumulation order.
Status WaveletCpuOpenMP(const OpContext& context, const Tensor& input,
                        const Tensor* other, const void* params,
                        Tensor* output);
#endif
#ifdef MYCV_WITH_TBB
// CPU TBB level: uses the same skeleton, with oneTBB parallel_for scheduling
// rows and columns.
Status WaveletCpuTbb(const OpContext& context, const Tensor& input,
                     const Tensor* other, const void* params, Tensor* output);
#endif
#ifdef MYCV_WITH_CUDA
// CUDA backend: one thread per output sample, with the entire pipeline executed
// by device kernels.
Status WaveletCuda(const OpContext& context, const Tensor& input,
                   const Tensor* other, const void* params, Tensor* output);
#endif

// Register implementations compiled into this binary in the operator registry.
// OpenCV has no corresponding transform; lookup returns kNotImplemented rather
// than silently running host code on device memory.
void RegisterWaveletKernels();

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_WAVELET_WAVELET_INTERNAL_H_
