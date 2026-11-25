#ifndef hpcplatform_OPS_GAUSSIAN_BLUR_H_
#define hpcplatform_OPS_GAUSSIAN_BLUR_H_

#include <cstdint>
#include <memory>
#include <string>

#include "hpcplatform/core/error.h"
#include "hpcplatform/core/tensor.h"
#include "hpcplatform/ops/image.h"
#include "hpcplatform/ops/op.h"
#include "hpcplatform/ops/operator.h"

namespace hpcplatform {

// Separable Gaussian smoothing: horizontal then vertical passes; output shape
// matches the input shape; supports u8/f32/f64.
//
// Separable Gaussian smoothing (two passes: horizontal then vertical).
//
// - `kernel_size` must be an odd integer >= 3; `sigma` must be finite and > 0.
// - Border handling is REPLICATE (edge pixels repeat), aligned with
//   cv::BORDER_REPLICATE so the OpenCV backend is directly comparable.
// - Output shape equals the input shape; dtype supports u8/f32/f64.
// - u8 output: both passes accumulate in double precision and the final value
//   is rounded back to u8 (half away from zero).
//
// `layout` selects the interpretation of rank-3/rank-4 tensors (HW/HWC/CHW/
// NHWC/NCHW, see hpcplatform/ops/image.h); OpenCV accepts HW/HWC/NHWC only.
Result<Tensor> GaussianBlur(const OpContext& context, const Tensor& input,
                            int kernel_size, double sigma,
                            ImageLayout layout = ImageLayout::kNHWC);

// A configured Gaussian blur holding a bound kernel size, sigma, and layout.
// Built by LoadOperators from a config entry; call Run with an input tensor
// allocated on the bound backend/device. See docs/operator_config.md.
class GaussianBlurOperator : public Operator {
 public:
  GaussianBlurOperator(std::string name, std::shared_ptr<Context> context,
                       CpuKernelLevel cpu_level, int kernel_size, double sigma,
                       ImageLayout layout);

  const std::string& name() const override;
  OpKind kind() const override;

  // Blurs `input` with the bound parameters. Output shape equals the input
  // shape; result is bit-identical to calling hpcplatform::GaussianBlur directly.
  Result<Tensor> Run(const Tensor& input) const;

 private:
  const std::string name_;
  const int kernel_size_;
  const double sigma_;
  const ImageLayout layout_;
};

}  // namespace hpcplatform

#endif  // hpcplatform_OPS_GAUSSIAN_BLUR_H_
