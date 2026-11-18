#include "test_framework.h"

#include <array>
#include <cmath>
#include <cstdint>

#include "mycv/mycv.h"

namespace {

using namespace mycv;

OpContext CpuOpContext(Context& context) {
  OpContext result;
  result.context = &context;
  result.stream = &context.stream();
  return result;
}

MYCV_TEST(ResizeBilinearMatchesHandComputedValues) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = context->AllocateTensor(Shape{2, 2}, DataType::kFloat32).value();
  const float values[] = {0, 1, 2, 3};
  for (int i = 0; i < 4; ++i) input.Data<float>()[i] = values[i];
  auto output = Resize(CpuOpContext(*context), input, 4, 4).value();
  MYCV_EXPECT_EQ(output.Data<float>()[0], 0.0f);
  MYCV_EXPECT_EQ(output.Data<float>()[1], 0.25f);
  MYCV_EXPECT_EQ(output.Data<float>()[5], 0.75f);
  MYCV_EXPECT_EQ(output.Data<float>()[15], 3.0f);
}

MYCV_TEST(ResizeNearestAndNCHWAreExplicit) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = context->AllocateTensor(Shape{1, 2, 2, 2}, DataType::kFloat32).value();
  for (int i = 0; i < 8; ++i) input.Data<float>()[i] = static_cast<float>(i);
  auto output = Resize(CpuOpContext(*context), input, 1, 1,
                       InterpolationMode::kBilinear, ImageLayout::kNCHW).value();
  const Shape expected_shape{1, 2, 1, 1};
  MYCV_EXPECT_TRUE(output.shape() == expected_shape);
  MYCV_EXPECT_EQ(output.Data<float>()[0], 1.5f);
  MYCV_EXPECT_EQ(output.Data<float>()[1], 5.5f);

  auto nearest = Resize(CpuOpContext(*context), input, 4, 4,
                        InterpolationMode::kNearest, ImageLayout::kNCHW).value();
  MYCV_EXPECT_EQ(nearest.Data<float>()[0], 0.0f);
  MYCV_EXPECT_EQ(nearest.Data<float>()[2], 1.0f);
}

MYCV_TEST(CropCopiesFloatHwcRoiAndRejectsBadRoi) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = context->AllocateTensor(Shape{3, 4, 2}, DataType::kFloat32).value();
  for (int i = 0; i < 24; ++i) input.Data<float>()[i] = static_cast<float>(i);
  auto output = Crop(CpuOpContext(*context), input, 1, 1, 2, 2).value();
  const float expected[] = {10, 11, 12, 13, 18, 19, 20, 21};
  for (int i = 0; i < 8; ++i) MYCV_EXPECT_EQ(output.Data<float>()[i], expected[i]);
  auto invalid = Crop(CpuOpContext(*context), input, 2, 3, 2, 2);
  MYCV_EXPECT_EQ(invalid.status().code(), StatusCode::kInvalidArgument);
}

MYCV_TEST(TransposeMaterializesPermutation) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = context->AllocateTensor(Shape{2, 3}, DataType::kFloat32).value();
  for (int i = 0; i < 6; ++i) input.Data<float>()[i] = static_cast<float>(i);
  auto output = Transpose(CpuOpContext(*context), input, {1, 0}).value();
  const Shape expected_shape{3, 2};
  const float expected[] = {0, 3, 1, 4, 2, 5};
  MYCV_EXPECT_TRUE(output.shape() == expected_shape);
  MYCV_EXPECT_TRUE(output.IsContiguous());
  for (int i = 0; i < 6; ++i) MYCV_EXPECT_EQ(output.Data<float>()[i], expected[i]);
}

MYCV_TEST(WarpAffineIdentityAndTranslation) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = context->AllocateTensor(Shape{2, 2}, DataType::kFloat32).value();
  const float values[] = {0, 1, 2, 3};
  for (int i = 0; i < 4; ++i) input.Data<float>()[i] = values[i];
  const std::array<double, 6> identity{1, 0, 0, 0, 1, 0};
  auto same = WarpAffine(CpuOpContext(*context), input, identity, 2, 2).value();
  for (int i = 0; i < 4; ++i) MYCV_EXPECT_EQ(same.Data<float>()[i], values[i]);

  const std::array<double, 6> translate{1, 0, 1, 0, 1, 0};
  auto moved = WarpAffine(CpuOpContext(*context), input, translate, 2, 2,
                          InterpolationMode::kBilinear, -1.0).value();
  const float expected[] = {-1, 0, -1, 2};
  for (int i = 0; i < 4; ++i) MYCV_EXPECT_EQ(moved.Data<float>()[i], expected[i]);

  const std::array<double, 6> singular{1, 2, 0, 2, 4, 0};
  auto invalid = WarpAffine(CpuOpContext(*context), input, singular, 2, 2);
  MYCV_EXPECT_EQ(invalid.status().code(), StatusCode::kInvalidArgument);
}

MYCV_TEST(ImageOpsHandleSingleElementAndInvalidDimensions) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = context->AllocateTensor(Shape{1, 1, 1}, DataType::kUInt8).value();
  input.Data<uint8_t>()[0] = 200;
  auto output = Resize(CpuOpContext(*context), input, 3, 2).value();
  for (int i = 0; i < 6; ++i) MYCV_EXPECT_EQ(output.Data<uint8_t>()[i], static_cast<uint8_t>(200));
  auto invalid = Resize(CpuOpContext(*context), input, 0, 2);
  MYCV_EXPECT_EQ(invalid.status().code(), StatusCode::kInvalidArgument);
}

}  // namespace

MYCV_TEST_MAIN()
