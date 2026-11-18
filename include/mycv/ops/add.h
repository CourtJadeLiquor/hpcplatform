#ifndef MYCV_OPS_ADD_H_
#define MYCV_OPS_ADD_H_

#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/op.h"

namespace mycv {

// Elementwise addition. 'other' must have the same shape or be a scalar.
Result<Tensor> Add(const OpContext& context, const Tensor& input,
                   const Tensor& other);

}  // namespace mycv

#endif  // MYCV_OPS_ADD_H_
