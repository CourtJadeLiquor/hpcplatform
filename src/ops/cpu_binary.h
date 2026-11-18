#ifndef MYCV_SRC_OPS_CPU_BINARY_H_
#define MYCV_SRC_OPS_CPU_BINARY_H_

#include <cstdint>
#include <type_traits>

#ifdef MYCV_WITH_TBB
#include <tbb/parallel_for.h>
#endif

#include "core/value_cast.h"
#include "ops/cpu_dispatch.h"
#include "ops/op_common.h"

namespace mycv {
namespace ops_internal {

enum class BinaryOperation { kAdd, kMul };

template <typename T>
T BinaryValue(T lhs, T rhs, BinaryOperation operation) {
  if constexpr (std::is_floating_point<T>::value) {
    return operation == BinaryOperation::kAdd ? lhs + rhs : lhs * rhs;
  } else {
    const long double value = operation == BinaryOperation::kAdd
                                  ? static_cast<long double>(lhs) + rhs
                                  : static_cast<long double>(lhs) * rhs;
    return detail::SaturatingCast<T>(value);
  }
}

template <typename T>
void BinaryElement(const Tensor& input, const Tensor& other,
                   BinaryOperation operation, Tensor* output, int64_t i,
                   bool contiguous_input, bool contiguous_other) {
  const bool scalar_rhs = other.Numel() == 1;
  const char* input_data = static_cast<const char*>(input.data());
  const char* other_data = static_cast<const char*>(other.data());
  const size_t input_offset = contiguous_input ? static_cast<size_t>(i) * sizeof(T)
                                               : StridedByteOffset(input, i);
  const size_t other_offset = scalar_rhs ? 0
      : (contiguous_other ? static_cast<size_t>(i) * sizeof(T)
                          : StridedByteOffset(other, i));
  const T lhs = *reinterpret_cast<const T*>(input_data + input_offset);
  const T rhs = *reinterpret_cast<const T*>(other_data + other_offset);
  output->Data<T>()[i] = BinaryValue(lhs, rhs, operation);
}

template <typename T>
Status BinaryNaiveTyped(const Tensor& input, const Tensor& other,
                        BinaryOperation operation, Tensor* output) {
  const bool contiguous_input = input.IsContiguous();
  const bool contiguous_other = other.IsContiguous();
  for (int64_t i = 0; i < input.Numel(); ++i) {
    BinaryElement<T>(input, other, operation, output, i, contiguous_input,
                     contiguous_other);
  }
  return Status::Ok();
}

template <typename T>
Status BinaryOpenMPTyped(const Tensor& input, const Tensor& other,
                         BinaryOperation operation, Tensor* output) {
  const bool contiguous_input = input.IsContiguous();
  const bool contiguous_other = other.IsContiguous();
#pragma omp parallel for
  for (int64_t i = 0; i < input.Numel(); ++i) {
    BinaryElement<T>(input, other, operation, output, i, contiguous_input,
                     contiguous_other);
  }
  return Status::Ok();
}

#ifdef MYCV_WITH_TBB
template <typename T>
Status BinaryTbbTyped(const Tensor& input, const Tensor& other,
                      BinaryOperation operation, Tensor* output) {
  const bool contiguous_input = input.IsContiguous();
  const bool contiguous_other = other.IsContiguous();
  tbb::parallel_for(int64_t{0}, input.Numel(), [&](int64_t i) {
    BinaryElement<T>(input, other, operation, output, i, contiguous_input,
                     contiguous_other);
  });
  return Status::Ok();
}
#endif

template <typename T>
Status BinarySimdTyped(const Tensor& input, const Tensor& other,
                       BinaryOperation operation, Tensor* output) {
  const bool scalar_rhs = other.Numel() == 1;
  if (input.IsContiguous() && (scalar_rhs || other.IsContiguous())) {
    const T* __restrict input_data = input.Data<T>();
    const T* __restrict other_data = other.Data<T>();
    T* __restrict output_data = output->Data<T>();
    if (scalar_rhs) {
      const T scalar = other_data[0];
      MYCV_CPU_IVDEP
      for (int64_t i = 0; i < input.Numel(); ++i) {
        output_data[i] = BinaryValue(input_data[i], scalar, operation);
      }
    } else {
      MYCV_CPU_IVDEP
      for (int64_t i = 0; i < input.Numel(); ++i) {
        output_data[i] = BinaryValue(input_data[i], other_data[i], operation);
      }
    }
    return Status::Ok();
  }
  return Status::Unsupported(
      "portable SIMD binary kernels require contiguous inputs");
}

inline Status BinaryNaive(const Tensor& input, const Tensor& other,
                          BinaryOperation operation, Tensor* output) {
  return DispatchArithmeticDType(input.dtype(), [&](auto tag) -> Status {
    using T = typename decltype(tag)::type;
    return BinaryNaiveTyped<T>(input, other, operation, output);
  });
}

inline Status BinaryOpenMP(const Tensor& input, const Tensor& other,
                           BinaryOperation operation, Tensor* output) {
  return DispatchArithmeticDType(input.dtype(), [&](auto tag) -> Status {
    using T = typename decltype(tag)::type;
    return BinaryOpenMPTyped<T>(input, other, operation, output);
  });
}

#ifdef MYCV_WITH_TBB
inline Status BinaryTbb(const Tensor& input, const Tensor& other,
                        BinaryOperation operation, Tensor* output) {
  return DispatchArithmeticDType(input.dtype(), [&](auto tag) -> Status {
    using T = typename decltype(tag)::type;
    return BinaryTbbTyped<T>(input, other, operation, output);
  });
}
#endif

inline Status BinarySimd(const Tensor& input, const Tensor& other,
                         BinaryOperation operation, Tensor* output) {
  return DispatchArithmeticDType(input.dtype(), [&](auto tag) -> Status {
    using T = typename decltype(tag)::type;
    return BinarySimdTyped<T>(input, other, operation, output);
  });
}

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_CPU_BINARY_H_
