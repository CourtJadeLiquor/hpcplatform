#include <cstring>

#include "ops/image_utils.h"
#include "ops/op_common.h"
#include "ops/op_params.h"

namespace mycv {
namespace ops_internal {

Status CropCpuNaive(const OpContext& context, const Tensor& input,
                    const Tensor* other, const void* params, Tensor* output) {
  (void)context;
  (void)other;
  const auto& options = *static_cast<const CropParams*>(params);
  auto image_result = ParseImage(input.shape(), options.layout);
  if (!image_result.ok()) return image_result.status();
  ImageInfo output_info = image_result.value();
  output_info.h = options.height;
  output_info.w = options.width;

  const char* source = static_cast<const char*>(input.data());
  char* destination = static_cast<char*>(output->data());
  const size_t element_size = input.ElementSize();
  for (int64_t n = 0; n < image_result.value().n; ++n) {
    for (int64_t y = 0; y < options.height; ++y) {
      for (int64_t x = 0; x < options.width; ++x) {
        for (int64_t c = 0; c < image_result.value().c; ++c) {
          const size_t source_offset = ImageByteOffset(
              input, image_result.value(), n, options.top + y, options.left + x, c);
          const int64_t output_index = ImageLinearIndex(output_info, n, y, x, c);
          std::memcpy(destination + static_cast<size_t>(output_index) * element_size,
                      source + source_offset, element_size);
        }
      }
    }
  }
  return Status::Ok();
}

}  // namespace ops_internal
}  // namespace mycv
