#include "core/value_cast.h"
#include "ops/op_common.h"

namespace mycv {
namespace ops_internal {

Status CastCpuNaive(const OpContext& context, const Tensor& input,
                    const Tensor* other, const void* params, Tensor* output) {
  (void)context;
  (void)other;
  (void)params;
  const char* source = static_cast<const char*>(input.data());
  char* destination = static_cast<char*>(output->data());
  for (int64_t i = 0; i < input.Numel(); ++i) {
    detail::CastElement(source + StridedByteOffset(input, i), input.dtype(),
                        destination + static_cast<size_t>(i) * output->ElementSize(),
                        output->dtype());
  }
  return Status::Ok();
}

}  // namespace ops_internal
}  // namespace mycv
