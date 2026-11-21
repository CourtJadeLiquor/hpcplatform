#ifndef MYCV_OPS_RESIZE_H_
#define MYCV_OPS_RESIZE_H_

#include <cstdint>

#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

namespace mycv {

// 缩放图像到 output_height x output_width。
//
// - 双线性使用 align_corners=false 与半像素中心；
// - 输入支持 HW/HWC/CHW/NHWC/NCHW 与 u8/f32/f64；
// - 成功时返回与输入同 dtype、同后端、同布局的新张量。
//
// Bilinear uses align_corners=false and half-pixel centers.
Result<Tensor> Resize(
    const OpContext& context, const Tensor& input, int64_t output_height,
    int64_t output_width,
    InterpolationMode interpolation = InterpolationMode::kBilinear,
    ImageLayout layout = ImageLayout::kNHWC);

}  // namespace mycv

#endif  // MYCV_OPS_RESIZE_H_
