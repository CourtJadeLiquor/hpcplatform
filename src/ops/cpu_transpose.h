#ifndef MYCV_SRC_OPS_CPU_TRANSPOSE_H_
#define MYCV_SRC_OPS_CPU_TRANSPOSE_H_

#include <cstring>

#ifdef MYCV_WITH_TBB
#include <tbb/parallel_for.h>
#endif

#include "ops/op_common.h"
#include "ops/op_params.h"

namespace mycv {
namespace ops_internal {

inline void TransposeElement(const Tensor& view, int64_t i, Tensor* output) {
  std::memcpy(static_cast<char*>(output->data()) +
                  static_cast<size_t>(i) * output->ElementSize(),
              static_cast<const char*>(view.data()) + StridedByteOffset(view, i),
              view.ElementSize());
}

inline Status TransposeOpenMPImpl(const Tensor& input,
                                  const TransposeParams& options, Tensor* output) {
  auto view = input.Permute(*options.order);
  if (!view.ok()) return view.status();
#pragma omp parallel for
  for (int64_t i = 0; i < output->Numel(); ++i) {
    TransposeElement(view.value(), i, output);
  }
  return Status::Ok();
}

#ifdef MYCV_WITH_TBB
inline Status TransposeTbbImpl(const Tensor& input, const TransposeParams& options,
                               Tensor* output) {
  auto view = input.Permute(*options.order);
  if (!view.ok()) return view.status();
  tbb::parallel_for(int64_t{0}, output->Numel(), [&](int64_t i) {
    TransposeElement(view.value(), i, output);
  });
  return Status::Ok();
}
#endif

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_CPU_TRANSPOSE_H_
