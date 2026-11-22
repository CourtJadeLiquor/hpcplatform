#ifndef MYCV_OPS_WAVELET_H_
#define MYCV_OPS_WAVELET_H_

#include <cstdint>
#include <memory>
#include <string>

#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
#include "mycv/operator.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

namespace mycv {

// Orthonormal single-level 2-D discrete wavelet transform (periodic boundary).
//
// One level splits an H x W plane (H and W must be even) into four H/2 x W/2
// bands, with the naming used throughout this project:
//     LL  low-pass rows  + low-pass columns   (approximation)
//     LH  low-pass rows  + high-pass columns  (horizontal detail)
//     HL  high-pass rows + low-pass columns   (vertical detail)
//     HH  high-pass rows + high-pass columns  (diagonal detail)
//
// Filters (analysis and synthesis share the same coefficients, so the pair is
// paraunitary and the inverse is exact up to floating-point rounding):
//     Haar   h = {1/sqrt2, 1/sqrt2}
//     Db4    h = {(1+sqrt3)/(4*sqrt2), (3+sqrt3)/(4*sqrt2),
//                 (3-sqrt3)/(4*sqrt2), (1-sqrt3)/(4*sqrt2)}
//     high   g[k] = (-1)^k * h[taps-1-k]
//
// u8 is rejected (detail bands are signed). All four bands keep the input
// dtype, layout and backend.
// Wavelet basis: Haar or Daubechies-4.
enum class WaveletKind : uint8_t {
  kHaar = 0,
  kDaubechies4 = 1,
};

// Returns the stable short name of a wavelet basis.
inline const char* WaveletKindName(WaveletKind kind) {
  switch (kind) {
    case WaveletKind::kHaar:
      return "haar";
    case WaveletKind::kDaubechies4:
      return "db4";
  }
  return "unknown";
}

// Parses a stable wavelet name back to its enum ("haar" / "db4"); an unknown
// name returns kInvalidArgument.
inline Result<WaveletKind> WaveletKindFromName(const std::string& name) {
  if (name == "haar") return WaveletKind::kHaar;
  if (name == "db4") return WaveletKind::kDaubechies4;
  return Status::InvalidArgument("unknown wavelet kind: " + name);
}

// Four subbands: LL approximation, LH horizontal detail, HL vertical detail,
// and HH diagonal detail.
struct WaveletBands {
  Tensor ll;
  Tensor lh;
  Tensor hl;
  Tensor hh;
};

// Single-level 2D discrete wavelet decomposition; H/W must be even; u8 is
// rejected.
// `kind` selects the analysis filter pair; `layout` follows the image API
// (rank-2 HW, rank-3 HWC/CHW, rank-4 NHWC/NCHW).
Result<WaveletBands> WaveletTransform(const OpContext& context,
                                      const Tensor& input,
                                      WaveletKind kind = WaveletKind::kHaar,
                                      ImageLayout layout = ImageLayout::kNHWC);

// Reconstructs an image from four subbands; all subbands must have the same
// shape, dtype, and backend, and kind must match the decomposition.
Result<Tensor> InverseWaveletTransform(const OpContext& context,
                                       const WaveletBands& bands,
                                       WaveletKind kind = WaveletKind::kHaar,
                                       ImageLayout layout = ImageLayout::kNHWC);

// A configured single-level wavelet decomposition holding a bound basis and
// layout. Built by LoadOperators from a config entry; call Run with an input
// allocated on the bound backend/device. See docs/operator_config.md.
class WaveletTransformOperator : public Operator {
 public:
  WaveletTransformOperator(std::string name, std::shared_ptr<Context> context,
                           CpuKernelLevel cpu_level, WaveletKind kind,
                           ImageLayout layout);

  const std::string& name() const override;
  OpKind kind() const override;

  // Decomposes `input` with the bound basis; bit-identical to calling
  // mycv::WaveletTransform directly.
  Result<WaveletBands> Run(const Tensor& input) const;

 private:
  const std::string name_;
  const WaveletKind kind_;
  const ImageLayout layout_;
};

}  // namespace mycv

#endif  // MYCV_OPS_WAVELET_H_
