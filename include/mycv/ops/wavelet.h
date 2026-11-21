#ifndef MYCV_OPS_WAVELET_H_
#define MYCV_OPS_WAVELET_H_

#include <cstdint>

#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
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
// 小波基：Haar 或 Daubechies-4。
enum class WaveletKind : uint8_t {
  kHaar = 0,
  kDaubechies4 = 1,
};

// 返回小波基的稳定短名。
inline const char* WaveletKindName(WaveletKind kind) {
  switch (kind) {
    case WaveletKind::kHaar:
      return "haar";
    case WaveletKind::kDaubechies4:
      return "db4";
  }
  return "unknown";
}

// 四个子带：LL 近似、LH 水平细节、HL 垂直细节、HH 对角细节。
struct WaveletBands {
  Tensor ll;
  Tensor lh;
  Tensor hl;
  Tensor hh;
};

// 单层二维离散小波分解；H/W 必须为偶数，拒绝 u8。
// `kind` selects the analysis filter pair; `layout` follows the image API
// (rank-2 HW, rank-3 HWC/CHW, rank-4 NHWC/NCHW).
Result<WaveletBands> WaveletTransform(const OpContext& context,
                                      const Tensor& input,
                                      WaveletKind kind = WaveletKind::kHaar,
                                      ImageLayout layout = ImageLayout::kNHWC);

// 由四个子带重建图像；所有子带必须同形状/类型/后端，kind 与分解时一致。
Result<Tensor> InverseWaveletTransform(const OpContext& context,
                                       const WaveletBands& bands,
                                       WaveletKind kind = WaveletKind::kHaar,
                                       ImageLayout layout = ImageLayout::kNHWC);

}  // namespace mycv

#endif  // MYCV_OPS_WAVELET_H_
