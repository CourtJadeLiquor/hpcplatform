#ifndef hpcplatform_OPS_LAPLACIAN_PYRAMID_H_
#define hpcplatform_OPS_LAPLACIAN_PYRAMID_H_

#include <memory>
#include <string>
#include <vector>

#include "hpcplatform/core/error.h"
#include "hpcplatform/core/tensor.h"
#include "hpcplatform/operator.h"
#include "hpcplatform/ops/image.h"
#include "hpcplatform/ops/op.h"

namespace hpcplatform {

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

// A configured Laplacian pyramid holding a bound level count, blur kernel,
// sigma, and layout. Built by LoadOperators from a config entry; call Run with
// an input allocated on the bound backend/device. See docs/operator_config.md.
class LaplacianPyramidOperator : public Operator {
 public:
  LaplacianPyramidOperator(std::string name, std::shared_ptr<Context> context,
                           CpuKernelLevel cpu_level, int levels,
                           int kernel_size, double sigma, ImageLayout layout);

  const std::string& name() const override;
  OpKind kind() const override;

  // Builds the pyramid with the bound parameters; bit-identical to calling
  // hpcplatform::BuildLaplacianPyramid directly.
  Result<LaplacianPyramid> Run(const Tensor& input) const;

 private:
  const std::string name_;
  const int levels_;
  const int kernel_size_;
  const double sigma_;
  const ImageLayout layout_;
};

}  // namespace hpcplatform

#endif  // hpcplatform_OPS_LAPLACIAN_PYRAMID_H_
