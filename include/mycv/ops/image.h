#ifndef MYCV_OPS_IMAGE_H_
#define MYCV_OPS_IMAGE_H_

#include <cstdint>

namespace mycv {

enum class InterpolationMode : uint8_t {
  kNearest = 0,
  kBilinear = 1,
};

// Rank 2 is HW. Rank 3 is HWC (NHWC) or CHW (NCHW with N=1). Rank 4 is
// NHWC or NCHW. Layout is explicit; no implicit transpose is performed.
enum class ImageLayout : uint8_t {
  kNHWC = 0,
  kNCHW = 1,
};

inline const char* ImageLayoutName(ImageLayout layout) {
  return layout == ImageLayout::kNCHW ? "nchw" : "nhwc";
}

}  // namespace mycv

#endif  // MYCV_OPS_IMAGE_H_
