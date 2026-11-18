#ifndef MYCV_OPS_CAST_H_
#define MYCV_OPS_CAST_H_

#include "mycv/core/dtype.h"
#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/op.h"

namespace mycv {

Result<Tensor> Cast(const OpContext& context, const Tensor& input,
                    DataType dst_dtype);

}  // namespace mycv

#endif  // MYCV_OPS_CAST_H_
