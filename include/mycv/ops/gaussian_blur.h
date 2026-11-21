#ifndef MYCV_OPS_GAUSSIAN_BLUR_H_
#define MYCV_OPS_GAUSSIAN_BLUR_H_

#include <cstdint>

#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

namespace mycv {

// 可分离高斯平滑：先水平后垂直两趟，输出与输入同形状，支持 u8/f32/f64。
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
// NHWC/NCHW, see mycv/ops/image.h); OpenCV accepts HW/HWC/NHWC only.
Result<Tensor> GaussianBlur(const OpContext& context, const Tensor& input,
                            int kernel_size, double sigma,
                            ImageLayout layout = ImageLayout::kNHWC);

}  // namespace mycv

#endif  // MYCV_OPS_GAUSSIAN_BLUR_H_
