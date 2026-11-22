#ifndef MYCV_OPS_RESIZE_H_
#define MYCV_OPS_RESIZE_H_

#include <cstdint>

#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

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

}  // namespace mycv

#endif  // MYCV_OPS_RESIZE_H_
