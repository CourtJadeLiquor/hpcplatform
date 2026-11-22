#ifndef MYCV_OPS_RESIZE_H_
#define MYCV_OPS_RESIZE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"
#include "mycv/operator.h"

namespace mycv {

// Resizes an image to output_height x output_width.
//
// - Bilinear uses align_corners=false and half-pixel centers;
// - Input supports HW/HWC/CHW/NHWC/NCHW layouts and u8/f32/f64;
// - On success, returns a new tensor with the input's dtype, backend, and layout.
//
// Bilinear uses align_corners=false and half-pixel centers.
Result<Tensor> Resize(
    const OpContext& context, const Tensor& input, int64_t output_height,
    int64_t output_width,
    InterpolationMode interpolation = InterpolationMode::kBilinear,
    ImageLayout layout = ImageLayout::kNHWC);

// A configured resize operator holding bound output dimensions, interpolation,
// and layout. Built by LoadOperators from a config entry; call Run with an input
// tensor allocated on the bound backend/device. See docs/operator_config.md.
class ResizeOperator : public Operator {
 public:
  ResizeOperator(std::string name, std::shared_ptr<Context> context,
                 CpuKernelLevel cpu_level, int64_t output_height,
                 int64_t output_width, InterpolationMode interpolation,
                 ImageLayout layout);

  const std::string& name() const override;
  OpKind kind() const override;

  // Resizes `input` with the bound parameters. Result is bit-identical to
  // calling the free function mycv::Resize with the same arguments.
  Result<Tensor> Run(const Tensor& input) const;

 private:
  const std::string name_;
  const int64_t output_height_;
  const int64_t output_width_;
  const InterpolationMode interpolation_;
  const ImageLayout layout_;
};

}  // namespace mycv

#endif  // MYCV_OPS_RESIZE_H_
