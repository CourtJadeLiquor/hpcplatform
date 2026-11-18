#include "test_framework.h"

#include <cmath>
#include <cstdint>

#include "mycv/mycv.h"

namespace {

using namespace mycv;

OpContext CpuOpContext(Context& context, CpuKernelLevel level = CpuKernelLevel::kNaive) {
  OpContext result;
  result.context = &context;
  result.stream = &context.stream();
  result.cpu_level = level;
  return result;
}

MYCV_TEST(AddMatchesHandComputedValues) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto lhs = context->AllocateTensor(Shape{2, 2}, DataType::kFloat32).value();
  auto rhs = context->AllocateTensor(Shape{2, 2}, DataType::kFloat32).value();
  for (int i = 0; i < 4; ++i) {
    lhs.Data<float>()[i] = static_cast<float>(i + 1);
    rhs.Data<float>()[i] = static_cast<float>(i + 5);
  }
  auto output = Add(CpuOpContext(*context), lhs, rhs).value();
  const float expected[] = {6, 8, 10, 12};
  for (int i = 0; i < 4; ++i) MYCV_EXPECT_EQ(output.Data<float>()[i], expected[i]);
}

MYCV_TEST(MulBroadcastsScalarAndReadsStridedInput) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto base = context->AllocateTensor(Shape{2, 3}, DataType::kFloat32).value();
  for (int i = 0; i < 6; ++i) base.Data<float>()[i] = static_cast<float>(i + 1);
  auto input = base.Slice(1, 1, 3).value();
  auto scalar = context->AllocateTensor(Shape{}, DataType::kFloat32).value();
  scalar.Data<float>()[0] = 2.0f;
  auto output = Mul(CpuOpContext(*context), input, scalar).value();
  const float expected[] = {4, 6, 10, 12};
  for (int i = 0; i < 4; ++i) MYCV_EXPECT_EQ(output.Data<float>()[i], expected[i]);
}

MYCV_TEST(IntegerAddSaturatesWithoutOverflow) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto lhs = context->AllocateTensor(Shape{2}, DataType::kInt8).value();
  auto rhs = context->AllocateTensor(Shape{2}, DataType::kInt8).value();
  lhs.Data<int8_t>()[0] = 120;
  lhs.Data<int8_t>()[1] = -120;
  rhs.Data<int8_t>()[0] = 10;
  rhs.Data<int8_t>()[1] = -10;
  auto output = Add(CpuOpContext(*context), lhs, rhs).value();
  MYCV_EXPECT_EQ(output.Data<int8_t>()[0], static_cast<int8_t>(127));
  MYCV_EXPECT_EQ(output.Data<int8_t>()[1], static_cast<int8_t>(-128));
}

MYCV_TEST(CastOpConvertsAndRejectsFloat16Arithmetic) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = context->AllocateTensor(Shape{3}, DataType::kFloat32).value();
  input.Data<float>()[0] = 1.9f;
  input.Data<float>()[1] = -1.0f;
  input.Data<float>()[2] = 256.0f;
  auto output = Cast(CpuOpContext(*context), input, DataType::kUInt8).value();
  MYCV_EXPECT_EQ(output.Data<uint8_t>()[0], static_cast<uint8_t>(1));
  MYCV_EXPECT_EQ(output.Data<uint8_t>()[1], static_cast<uint8_t>(0));
  MYCV_EXPECT_EQ(output.Data<uint8_t>()[2], static_cast<uint8_t>(255));

  auto half = context->AllocateTensor(Shape{1}, DataType::kFloat16).value();
  auto status = Add(CpuOpContext(*context), half, half);
  MYCV_EXPECT_EQ(status.status().code(), StatusCode::kUnsupported);
}

MYCV_TEST(NormalizeComputesPerChannelValues) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = context->AllocateTensor(Shape{1, 2, 2}, DataType::kFloat32).value();
  const float values[] = {3, 6, 5, 10};
  for (int i = 0; i < 4; ++i) input.Data<float>()[i] = values[i];
  auto output = Normalize(CpuOpContext(*context), input, {1, 2}, {2, 4}).value();
  const float expected[] = {1, 1, 2, 2};
  for (int i = 0; i < 4; ++i) MYCV_EXPECT_EQ(output.Data<float>()[i], expected[i]);
}

MYCV_TEST(NormalizeIntegerProducesFloatAndValidatesStd) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = context->AllocateTensor(Shape{1, 1, 2}, DataType::kUInt8).value();
  input.Data<uint8_t>()[0] = 255;
  input.Data<uint8_t>()[1] = 0;
  auto output = Normalize(CpuOpContext(*context), input, {0, 0}, {255, 1}).value();
  MYCV_EXPECT_EQ(output.dtype(), DataType::kFloat32);
  MYCV_EXPECT_EQ(output.Data<float>()[0], 1.0f);
  MYCV_EXPECT_EQ(output.Data<float>()[1], 0.0f);
  auto invalid = Normalize(CpuOpContext(*context), input, {0, 0}, {1, 0});
  MYCV_EXPECT_EQ(invalid.status().code(), StatusCode::kInvalidArgument);
}

MYCV_TEST(UnbuiltCpuLevelIsAnExplicitError) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = context->AllocateTensor(Shape{1}, DataType::kFloat32).value();
  auto result = Add(CpuOpContext(*context, CpuKernelLevel::kSimdNeon), input, input);
  MYCV_EXPECT_FALSE(result.ok());
  MYCV_EXPECT_EQ(result.status().code(), StatusCode::kUnsupported);
  MYCV_EXPECT_TRUE(result.status().message().find("simd_neon") != std::string::npos);
}

}  // namespace

MYCV_TEST_MAIN()
