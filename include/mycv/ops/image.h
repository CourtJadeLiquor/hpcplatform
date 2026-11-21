#ifndef MYCV_OPS_IMAGE_H_
#define MYCV_OPS_IMAGE_H_

#include <cstdint>

namespace mycv {

// 插值方式：最近邻或双线性。
enum class InterpolationMode : uint8_t {
  kNearest = 0,
  kBilinear = 1,
};

// 图像布局：rank 2 为 HW；rank 3 为 HWC 或 CHW；rank 4 为 NHWC 或 NCHW。
// 布局由调用方显式指定，不做隐式转置。
enum class ImageLayout : uint8_t {
  kNHWC = 0,
  kNCHW = 1,
};

// 返回布局的稳定短名。
inline const char* ImageLayoutName(ImageLayout layout) {
  return layout == ImageLayout::kNCHW ? "nchw" : "nhwc";
}

}  // namespace mycv

#endif  // MYCV_OPS_IMAGE_H_
