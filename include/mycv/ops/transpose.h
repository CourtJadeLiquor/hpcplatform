#ifndef MYCV_OPS_TRANSPOSE_H_
#define MYCV_OPS_TRANSPOSE_H_

#include <vector>

#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/op.h"

namespace mycv {

// Materializes a permutation into a contiguous output tensor.
Result<Tensor> Transpose(const OpContext& context, const Tensor& input,
                         const std::vector<int>& order);

}  // namespace mycv

#endif  // MYCV_OPS_TRANSPOSE_H_
