#ifndef MYCV_OPS_NORMALIZE_H_
#define MYCV_OPS_NORMALIZE_H_

#include <vector>

#include "mycv/core/error.h"
#include "mycv/core/tensor.h"
#include "mycv/ops/op.h"

namespace mycv {

// Channel-last normalization: output = (input - mean[c]) / (std[c] + epsilon).
// Integral inputs produce f32; f32/f64 inputs preserve their dtype.
Result<Tensor> Normalize(const OpContext& context, const Tensor& input,
                         const std::vector<double>& mean,
                         const std::vector<double>& std,
                         double epsilon = 0.0);

}  // namespace mycv

#endif  // MYCV_OPS_NORMALIZE_H_
