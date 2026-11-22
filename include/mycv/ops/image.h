#ifndef MYCV_OPS_IMAGE_H_
#define MYCV_OPS_IMAGE_H_

#include <cstdint>

namespace mycv {

// Interpolation mode: nearest neighbor or bilinear.
enum class InterpolationMode : uint8_t {
  kNearest = 0,
  kBilinear = 1,
};

// Image layouts: rank 2 is HW; rank 3 is HWC or CHW; rank 4 is NHWC or NCHW.
// The caller specifies the layout explicitly; no implicit transposition occurs.
enum class ImageLayout : uint8_t {
  kNHWC = 0,
  kNCHW = 1,
};

// Returns the stable short name of a layout.
inline const char* ImageLayoutName(ImageLayout layout) {
  return layout == ImageLayout::kNCHW ? "nchw" : "nhwc";
}

}  // namespace mycv

#endif  // MYCV_OPS_IMAGE_H_
