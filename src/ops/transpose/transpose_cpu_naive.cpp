#include <cstring>

#include "mycv/core/tensor_iterator.h"
#include "ops/op_common.h"
#include "ops/op_params.h"

namespace mycv {
namespace ops_internal {

Status TransposeCpuNaive(const OpContext& context, const Tensor& input,
                         const Tensor* other, const void* params, Tensor* output) {
  (void)context;
  (void)other;
  const auto& options = *static_cast<const TransposeParams*>(params);
  auto view = input.Permute(*options.order);
  if (!view.ok()) return view.status();

  TensorIterator source(view.value());
  char* destination = static_cast<char*>(output->data());
  const size_t element_size = input.ElementSize();
  while (source.valid()) {
    std::memcpy(destination + static_cast<size_t>(source.linear_index()) * element_size,
                source.data(), element_size);
    source.Next();
  }
  return Status::Ok();
}

}  // namespace ops_internal
}  // namespace mycv
