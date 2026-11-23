#ifndef hpcplatform_OPS_IMAGE_H_
#define hpcplatform_OPS_IMAGE_H_

#include <cstdint>
#include <string>

#include "hpcplatform/core/error.h"

namespace hpcplatform {

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

// Returns the interpolation mode's stable short name.
inline const char* InterpolationModeName(InterpolationMode mode) {
  switch (mode) {
    case InterpolationMode::kNearest:
      return "nearest";
    case InterpolationMode::kBilinear:
      return "bilinear";
  }
  return "unknown";
}

// Parses a stable interpolation name back to its enum.
inline Result<InterpolationMode> InterpolationModeFromName(const std::string& name) {
  if (name == "nearest") return InterpolationMode::kNearest;
  if (name == "bilinear") return InterpolationMode::kBilinear;
  return Status::InvalidArgument("unknown interpolation name: " + name);
}

// Returns the stable short name of a layout.
inline const char* ImageLayoutName(ImageLayout layout) {
  return layout == ImageLayout::kNCHW ? "nchw" : "nhwc";
}

// Parses a stable layout name back to its enum.
inline Result<ImageLayout> ImageLayoutFromName(const std::string& name) {
  if (name == "nhwc") return ImageLayout::kNHWC;
  if (name == "nchw") return ImageLayout::kNCHW;
  return Status::InvalidArgument("unknown layout name: " + name);
}

}  // namespace hpcplatform

#endif  // hpcplatform_OPS_IMAGE_H_
