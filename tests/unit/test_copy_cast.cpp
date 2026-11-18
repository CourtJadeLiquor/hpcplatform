#include "test_framework.h"

#include <cmath>
#include <cstdint>

#include "mycv/core/context.h"

namespace {

using namespace mycv;

MYCV_TEST(CopyHandlesNonContiguousSource) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto source = context->AllocateTensor(Shape{2, 4}, DataType::kInt32).value();
  for (int i = 0; i < 8; ++i) source.Data<int32_t>()[i] = i + 1;
  auto slice = source.Slice(1, 1, 3).value();
  auto destination = context->AllocateTensor(slice.shape(), DataType::kInt32).value();

  MYCV_EXPECT_OK(context->Copy(&destination, slice));
  const int32_t expected[] = {2, 3, 6, 7};
  for (int i = 0; i < 4; ++i) {
    MYCV_EXPECT_EQ(destination.Data<int32_t>()[i], expected[i]);
  }
}

MYCV_TEST(CopyRejectsMismatchedDestination) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto source = context->AllocateTensor(Shape{2}, DataType::kFloat32).value();
  auto wrong_shape = context->AllocateTensor(Shape{3}, DataType::kFloat32).value();
  auto wrong_dtype = context->AllocateTensor(Shape{2}, DataType::kFloat64).value();
  MYCV_EXPECT_EQ(context->Copy(&wrong_shape, source).code(),
                 StatusCode::kInvalidArgument);
  MYCV_EXPECT_EQ(context->Copy(&wrong_dtype, source).code(),
                 StatusCode::kInvalidArgument);
}

MYCV_TEST(CastSaturatesAndSupportsFloat16RoundTrip) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto source = context->AllocateTensor(Shape{5}, DataType::kFloat32).value();
  const float values[] = {-2.0f, 0.9f, 1.9f, 300.0f, 3.1415926f};
  for (int i = 0; i < 5; ++i) source.Data<float>()[i] = values[i];

  auto bytes = source.Cast(DataType::kUInt8).value();
  const uint8_t expected[] = {0, 0, 1, 255, 3};
  for (int i = 0; i < 5; ++i) MYCV_EXPECT_EQ(bytes.Data<uint8_t>()[i], expected[i]);

  auto half = source.Cast(DataType::kFloat16).value();
  MYCV_EXPECT_EQ(half.ElementSize(), static_cast<size_t>(2));
  auto restored = half.Cast(DataType::kFloat32).value();
  for (int i = 0; i < 5; ++i) {
    const float tolerance = std::max(0.002f, std::fabs(values[i]) * 0.001f);
    MYCV_EXPECT_TRUE(std::fabs(restored.Data<float>()[i] - values[i]) <= tolerance);
  }
}

MYCV_TEST(CastReadsStridedInput) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto source = context->AllocateTensor(Shape{2, 3}, DataType::kInt32).value();
  for (int i = 0; i < 6; ++i) source.Data<int32_t>()[i] = i;
  auto slice = source.Slice(1, 1, 3).value();
  auto cast = slice.Cast(DataType::kFloat64).value();
  const double expected[] = {1, 2, 4, 5};
  for (int i = 0; i < 4; ++i) MYCV_EXPECT_EQ(cast.Data<double>()[i], expected[i]);
}

}  // namespace

MYCV_TEST_MAIN()
