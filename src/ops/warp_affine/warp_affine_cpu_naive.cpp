#include <cmath>

#include "core/value_cast.h"
#include "ops/image_utils.h"
#include "ops/op_common.h"
#include "ops/op_params.h"

namespace mycv {
namespace ops_internal {

Status WarpAffineCpuNaive(const OpContext& context, const Tensor& input,
                          const Tensor* other, const void* params, Tensor* output) {
  (void)context;
  (void)other;
  const auto& options = *static_cast<const WarpAffineParams*>(params);
  auto image_result = ParseImage(input.shape(), options.layout);
  if (!image_result.ok()) return image_result.status();
  const ImageInfo& image = image_result.value();
  ImageInfo output_info = image;
  output_info.h = options.height;
  output_info.w = options.width;
  const auto& m = *options.matrix;
  const long double determinant = m[0] * m[4] - m[1] * m[3];

  const char* source = static_cast<const char*>(input.data());
  char* destination = static_cast<char*>(output->data());
  const auto sample = [&](int64_t n, int64_t y, int64_t x, int64_t c) {
    if (y < 0 || y >= image.h || x < 0 || x >= image.w) {
      return static_cast<long double>(options.border_value);
    }
    return detail::LoadAsLongDouble(
        source + ImageByteOffset(input, image, n, y, x, c), input.dtype());
  };

  for (int64_t n = 0; n < image.n; ++n) {
    for (int64_t y = 0; y < options.height; ++y) {
      for (int64_t x = 0; x < options.width; ++x) {
        const long double dx = x - m[2];
        const long double dy = y - m[5];
        const long double source_x = (m[4] * dx - m[1] * dy) / determinant;
        const long double source_y = (-m[3] * dx + m[0] * dy) / determinant;
        const int64_t x0 = static_cast<int64_t>(std::floor(source_x));
        const int64_t y0 = static_cast<int64_t>(std::floor(source_y));
        const long double wx = source_x - x0;
        const long double wy = source_y - y0;

        for (int64_t c = 0; c < image.c; ++c) {
          long double value = options.border_value;
          if (options.interpolation == InterpolationMode::kNearest) {
            value = sample(n, static_cast<int64_t>(std::lround(source_y)),
                           static_cast<int64_t>(std::lround(source_x)), c);
          } else {
            const long double top_left = sample(n, y0, x0, c);
            const long double top_right = sample(n, y0, x0 + 1, c);
            const long double bottom_left = sample(n, y0 + 1, x0, c);
            const long double bottom_right = sample(n, y0 + 1, x0 + 1, c);
            const long double top = top_left + (top_right - top_left) * wx;
            const long double bottom = bottom_left + (bottom_right - bottom_left) * wx;
            value = top + (bottom - top) * wy;
          }
          if (input.dtype() == DataType::kUInt8) value = std::round(value);
          const int64_t output_index = ImageLinearIndex(output_info, n, y, x, c);
          detail::StoreFromLongDouble(
              destination + static_cast<size_t>(output_index) * output->ElementSize(),
              output->dtype(), value);
        }
      }
    }
  }
  return Status::Ok();
}

}  // namespace ops_internal
}  // namespace mycv
