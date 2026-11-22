#ifndef MYCV_OPS_LAPLACIAN_PYRAMID_H_
#define MYCV_OPS_LAPLACIAN_PYRAMID_H_

#include <vector>

#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

namespace mycv {

// Gaussian/Laplacian pyramid built on top of GaussianBlur and Resize.
//
// One stage of the Gaussian pyramid is
//     G_{i+1} = ResizeNearest(GaussianBlur(G_i))       (ceil-halved size)
// and the matching Laplacian band is
//     L_i = G_i - ResizeBilinear(G_{i+1})              (same size as G_i)
// The coarsest Gaussian level is returned as the last element, so a pyramid of
// `levels` band-limited images has `levels + 1` tensors:
//     levels[0..levels-1]  band-pass L_i, finest first
//     levels[levels]       coarse residual G_levels
//
// Because the bands are signed differences, only f32/f64 inputs are accepted
// (u8 cannot represent a negative residual without a separate signed dtype).
// Padding inside GaussianBlur is REPLICATE; `layout` follows the image API.
// Laplacian pyramid result: levels[0..N-1] are band-pass levels; levels[N] is
// the coarsest residual.
struct LaplacianPyramid {
  std::vector<Tensor> levels;
};

// Builds the pyramid using GaussianBlur + Resize; accepts only f32/f64 inputs.
//
// `levels` must be >= 1 and small enough that every stage stays >= 2x2.
// `kernel_size` must be an odd integer >= 3; `sigma` must be finite and > 0.
Result<LaplacianPyramid> BuildLaplacianPyramid(
    const OpContext& context, const Tensor& input, int levels,
    int kernel_size = 5, double sigma = 1.0,
    ImageLayout layout = ImageLayout::kNHWC);

}  // namespace mycv

#endif  // MYCV_OPS_LAPLACIAN_PYRAMID_H_
