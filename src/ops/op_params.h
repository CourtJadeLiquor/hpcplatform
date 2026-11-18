#ifndef MYCV_SRC_OPS_OP_PARAMS_H_
#define MYCV_SRC_OPS_OP_PARAMS_H_

#include <array>
#include <cstdint>
#include <vector>

#include "mycv/ops/image.h"

namespace mycv {
namespace ops_internal {

struct NormalizeParams {
  const std::vector<double>* mean;
  const std::vector<double>* std;
  double epsilon;
};

struct ResizeParams {
  int64_t height;
  int64_t width;
  InterpolationMode interpolation;
  ImageLayout layout;
};

struct CropParams {
  int64_t top;
  int64_t left;
  int64_t height;
  int64_t width;
  ImageLayout layout;
};

struct TransposeParams {
  const std::vector<int>* order;
};

struct WarpAffineParams {
  const std::array<double, 6>* matrix;
  int64_t height;
  int64_t width;
  InterpolationMode interpolation;
  double border_value;
  ImageLayout layout;
};

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_OP_PARAMS_H_
