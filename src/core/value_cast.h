#ifndef MYCV_SRC_CORE_VALUE_CAST_H_
#define MYCV_SRC_CORE_VALUE_CAST_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "mycv/core/dtype.h"

namespace mycv {
namespace detail {

inline float Float16BitsToFloat(uint16_t half) {
  const uint32_t sign = static_cast<uint32_t>(half & 0x8000U) << 16U;
  const uint32_t exponent = (half >> 10U) & 0x1fU;
  const uint32_t fraction = half & 0x03ffU;
  uint32_t bits = 0;
  if (exponent == 0) {
    if (fraction == 0) {
      bits = sign;
    } else {
      // Normalize a subnormal half into a normal float.
      uint32_t mantissa = fraction;
      int shift = 0;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1U;
        ++shift;
      }
      mantissa &= 0x03ffU;
      const uint32_t float_exponent = static_cast<uint32_t>(127 - 15 - shift);
      bits = sign | (float_exponent << 23U) | (mantissa << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (fraction << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
  }
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline uint16_t FloatToFloat16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint16_t sign = static_cast<uint16_t>((bits >> 16U) & 0x8000U);
  const uint32_t exponent = (bits >> 23U) & 0xffU;
  const uint32_t fraction = bits & 0x007fffffU;

  if (exponent == 0xffU) {
    return static_cast<uint16_t>(sign | 0x7c00U |
                                 (fraction == 0 ? 0 : (fraction >> 13U) | 1U));
  }

  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00U);
  if (half_exponent <= 0) {
    if (half_exponent < -10) return sign;
    const uint32_t mantissa = fraction | 0x00800000U;
    const int shift = 14 - half_exponent;
    uint32_t rounded = mantissa >> shift;
    const uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const uint32_t halfway = 1U << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0)) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }

  uint32_t rounded = fraction >> 13U;
  const uint32_t remainder = fraction & 0x1fffU;
  bool increment =
      remainder > 0x1000U || (remainder == 0x1000U && (rounded & 1U) != 0);
  uint32_t result_exponent = static_cast<uint32_t>(half_exponent);
  if (increment) {
    ++rounded;
    if (rounded == 0x400U) {
      rounded = 0;
      ++result_exponent;
      if (result_exponent >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00U);
    }
  }
  return static_cast<uint16_t>(sign | (result_exponent << 10U) | rounded);
}

template <typename T>
T SaturatingCast(long double value) {
  if constexpr (std::is_floating_point<T>::value) {
    return static_cast<T>(value);
  } else {
    if (std::isnan(value)) return static_cast<T>(0);
    const long double low =
        static_cast<long double>(std::numeric_limits<T>::min());
    const long double high =
        static_cast<long double>(std::numeric_limits<T>::max());
    value = std::max(low, std::min(high, value));
    return static_cast<T>(value);
  }
}

inline long double LoadAsLongDouble(const void* data, DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8:
      return *static_cast<const uint8_t*>(data);
    case DataType::kInt8:
      return *static_cast<const int8_t*>(data);
    case DataType::kUInt16:
      return *static_cast<const uint16_t*>(data);
    case DataType::kInt16:
      return *static_cast<const int16_t*>(data);
    case DataType::kFloat16: {
      uint16_t bits = 0;
      std::memcpy(&bits, data, sizeof(bits));
      return Float16BitsToFloat(bits);
    }
    case DataType::kFloat32:
      return *static_cast<const float*>(data);
    case DataType::kFloat64:
      return *static_cast<const double*>(data);
    case DataType::kInt32:
      return *static_cast<const int32_t*>(data);
    case DataType::kInt64:
      return *static_cast<const int64_t*>(data);
  }
  return 0;
}

inline void StoreFromLongDouble(void* data, DataType dtype, long double value) {
  switch (dtype) {
    case DataType::kUInt8:
      *static_cast<uint8_t*>(data) = SaturatingCast<uint8_t>(value);
      break;
    case DataType::kInt8:
      *static_cast<int8_t*>(data) = SaturatingCast<int8_t>(value);
      break;
    case DataType::kUInt16:
      *static_cast<uint16_t*>(data) = SaturatingCast<uint16_t>(value);
      break;
    case DataType::kInt16:
      *static_cast<int16_t*>(data) = SaturatingCast<int16_t>(value);
      break;
    case DataType::kFloat16: {
      const uint16_t bits = FloatToFloat16Bits(static_cast<float>(value));
      std::memcpy(data, &bits, sizeof(bits));
      break;
    }
    case DataType::kFloat32:
      *static_cast<float*>(data) = static_cast<float>(value);
      break;
    case DataType::kFloat64:
      *static_cast<double*>(data) = static_cast<double>(value);
      break;
    case DataType::kInt32:
      *static_cast<int32_t*>(data) = SaturatingCast<int32_t>(value);
      break;
    case DataType::kInt64:
      *static_cast<int64_t*>(data) = SaturatingCast<int64_t>(value);
      break;
  }
}

inline void CastElement(const void* src, DataType src_dtype, void* dst,
                        DataType dst_dtype) {
  StoreFromLongDouble(dst, dst_dtype, LoadAsLongDouble(src, src_dtype));
}

}  // namespace detail
}  // namespace mycv

#endif  // MYCV_SRC_CORE_VALUE_CAST_H_
