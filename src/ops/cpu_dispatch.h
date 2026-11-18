#ifndef MYCV_SRC_OPS_CPU_DISPATCH_H_
#define MYCV_SRC_OPS_CPU_DISPATCH_H_

#include <cstdint>

#include "mycv/core/dtype.h"
#include "mycv/core/error.h"

#if defined(__GNUC__) && !defined(__clang__)
#define MYCV_CPU_IVDEP _Pragma("GCC ivdep")
#else
#define MYCV_CPU_IVDEP
#endif

namespace mycv {
namespace ops_internal {

template <typename T>
struct DTypeTag {
  using type = T;
};

template <typename Fn>
Status DispatchArithmeticDType(DataType dtype, Fn&& fn) {
  switch (dtype) {
    case DataType::kUInt8: return fn(DTypeTag<uint8_t>{});
    case DataType::kInt8: return fn(DTypeTag<int8_t>{});
    case DataType::kUInt16: return fn(DTypeTag<uint16_t>{});
    case DataType::kInt16: return fn(DTypeTag<int16_t>{});
    case DataType::kFloat32: return fn(DTypeTag<float>{});
    case DataType::kFloat64: return fn(DTypeTag<double>{});
    case DataType::kInt32: return fn(DTypeTag<int32_t>{});
    case DataType::kInt64: return fn(DTypeTag<int64_t>{});
    case DataType::kFloat16: break;
  }
  return Status::Unsupported("this CPU kernel does not support float16");
}

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_CPU_DISPATCH_H_
