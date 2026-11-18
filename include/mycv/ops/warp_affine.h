#ifndef MYCV_OPS_WARP_AFFINE_H_
#define MYCV_OPS_WARP_AFFINE_H_

#include <array>
#include <cstdint>

#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

namespace mycv {

// matrix is a forward source-to-destination 2x3 transform in row-major order.
Result<Tensor> WarpAffine(
    const OpContext& context, const Tensor& input,
    const std::array<double, 6>& matrix, int64_t output_height,
    int64_t output_width,
    InterpolationMode interpolation = InterpolationMode::kBilinear,
    double border_value = 0.0, ImageLayout layout = ImageLayout::kNHWC);

}  // namespace mycv

#endif  // MYCV_OPS_WARP_AFFINE_H_
