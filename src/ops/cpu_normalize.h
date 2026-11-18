#ifndef MYCV_SRC_OPS_CPU_NORMALIZE_H_
#define MYCV_SRC_OPS_CPU_NORMALIZE_H_

#ifdef MYCV_WITH_TBB
#include <tbb/parallel_for.h>
#endif

#include "core/value_cast.h"
#include "ops/cpu_dispatch.h"
#include "ops/op_common.h"
#include "ops/op_params.h"

namespace mycv {
namespace ops_internal {

template <typename Out>
Out NormalizeElement(const Tensor& input, const NormalizeParams& options,
                     int64_t i, bool contiguous) {
  const int64_t channels = input.shape().Dim(input.shape().Rank() - 1);
  const size_t channel = static_cast<size_t>(i % channels);
  const char* source = static_cast<const char*>(input.data());
  const size_t offset = contiguous ? static_cast<size_t>(i) * input.ElementSize()
                                   : StridedByteOffset(input, i);
  const Out value = static_cast<Out>(
      detail::LoadAsLongDouble(source + offset, input.dtype()));
  return (value - static_cast<Out>((*options.mean)[channel])) /
         static_cast<Out>((*options.std)[channel] + options.epsilon);
}

template <typename Out>
Status NormalizeLoop(const Tensor& input, const NormalizeParams& options,
                     Tensor* output) {
  const bool contiguous = input.IsContiguous();
  for (int64_t i = 0; i < input.Numel(); ++i) {
    output->Data<Out>()[i] = NormalizeElement<Out>(input, options, i, contiguous);
  }
  return Status::Ok();
}

template <typename Out>
Status NormalizeOpenMPLoop(const Tensor& input, const NormalizeParams& options,
                           Tensor* output) {
  const bool contiguous = input.IsContiguous();
#pragma omp parallel for
  for (int64_t i = 0; i < input.Numel(); ++i) {
    output->Data<Out>()[i] = NormalizeElement<Out>(input, options, i, contiguous);
  }
  return Status::Ok();
}

#ifdef MYCV_WITH_TBB
template <typename Out>
Status NormalizeTbbLoop(const Tensor& input, const NormalizeParams& options,
                        Tensor* output) {
  const bool contiguous = input.IsContiguous();
  tbb::parallel_for(int64_t{0}, input.Numel(), [&](int64_t i) {
    output->Data<Out>()[i] = NormalizeElement<Out>(input, options, i, contiguous);
  });
  return Status::Ok();
}
#endif

template <typename In, typename Out>
Status NormalizeSimdTyped(const Tensor& input, const NormalizeParams& options,
                          Tensor* output) {
  if (!input.IsContiguous()) {
    return Status::Unsupported("SIMD Normalize requires a contiguous input");
  }
  const In* __restrict source = input.Data<In>();
  Out* __restrict destination = output->Data<Out>();
  const int64_t channels = input.shape().Dim(input.shape().Rank() - 1);
  MYCV_CPU_IVDEP
  for (int64_t i = 0; i < input.Numel(); ++i) {
    const size_t channel = static_cast<size_t>(i % channels);
    destination[i] =
        (static_cast<Out>(source[i]) - static_cast<Out>((*options.mean)[channel])) /
        static_cast<Out>((*options.std)[channel] + options.epsilon);
  }
  return Status::Ok();
}

inline Status NormalizeNaiveImpl(const Tensor& input, const NormalizeParams& options,
                                 Tensor* output) {
  return output->dtype() == DataType::kFloat32
             ? NormalizeLoop<float>(input, options, output)
             : NormalizeLoop<double>(input, options, output);
}

inline Status NormalizeOpenMPImpl(const Tensor& input,
                                  const NormalizeParams& options, Tensor* output) {
  return output->dtype() == DataType::kFloat32
             ? NormalizeOpenMPLoop<float>(input, options, output)
             : NormalizeOpenMPLoop<double>(input, options, output);
}

#ifdef MYCV_WITH_TBB
inline Status NormalizeTbbImpl(const Tensor& input, const NormalizeParams& options,
                               Tensor* output) {
  return output->dtype() == DataType::kFloat32
             ? NormalizeTbbLoop<float>(input, options, output)
             : NormalizeTbbLoop<double>(input, options, output);
}
#endif

inline Status NormalizeSimdImpl(const Tensor& input, const NormalizeParams& options,
                                Tensor* output) {
  return DispatchArithmeticDType(input.dtype(), [&](auto tag) -> Status {
    using In = typename decltype(tag)::type;
    return output->dtype() == DataType::kFloat32
               ? NormalizeSimdTyped<In, float>(input, options, output)
               : NormalizeSimdTyped<In, double>(input, options, output);
  });
}

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_CPU_NORMALIZE_H_
