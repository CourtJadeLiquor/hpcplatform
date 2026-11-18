#ifndef MYCV_OPS_CROP_H_
#define MYCV_OPS_CROP_H_

#include <cstdint>

#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

namespace mycv {

// Copies an image ROI into a contiguous output tensor.
Result<Tensor> Crop(const OpContext& context, const Tensor& input, int64_t top,
                    int64_t left, int64_t height, int64_t width,
                    ImageLayout layout = ImageLayout::kNHWC);

}  // namespace mycv

#endif  // MYCV_OPS_CROP_H_
