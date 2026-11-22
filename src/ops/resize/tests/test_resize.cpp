// Tests resize interpolation modes, coordinate mapping, alignment, and invalid-argument error paths.
#include "test_framework.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "core/backend.h"
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
  for (int i = 0; i < 4; ++i)
    input.Data<float>()[i] = values[i];
  auto output = Resize(CpuOpContext(*context), input, 4, 4).value();
  MYCV_EXPECT_EQ(output.Data<float>()[0], 0.0f);
  MYCV_EXPECT_EQ(output.Data<float>()[1], 0.25f);
  MYCV_EXPECT_EQ(output.Data<float>()[5], 0.75f);
  MYCV_EXPECT_EQ(output.Data<float>()[15], 3.0f);
}

MYCV_TEST(ResizeNearestAndNCHWAreExplicit) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input =
      context->AllocateTensor(Shape{1, 2, 2, 2}, DataType::kFloat32).value();
  for (int i = 0; i < 8; ++i)
    input.Data<float>()[i] = static_cast<float>(i);
  auto output = Resize(CpuOpContext(*context), input, 1, 1,
                       InterpolationMode::kBilinear, ImageLayout::kNCHW)
                    .value();
  const Shape expected_shape{1, 2, 1, 1};
  MYCV_EXPECT_TRUE(output.shape() == expected_shape);
  MYCV_EXPECT_EQ(output.Data<float>()[0], 1.5f);
  MYCV_EXPECT_EQ(output.Data<float>()[1], 5.5f);

  auto nearest = Resize(CpuOpContext(*context), input, 4, 4,
                        InterpolationMode::kNearest, ImageLayout::kNCHW)
                     .value();
  MYCV_EXPECT_EQ(nearest.Data<float>()[0], 0.0f);
  MYCV_EXPECT_EQ(nearest.Data<float>()[2], 1.0f);
}

MYCV_TEST(ImageOpsHandleSingleElementAndInvalidDimensions) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input =
      context->AllocateTensor(Shape{1, 1, 1}, DataType::kUInt8).value();
  input.Data<uint8_t>()[0] = 200;
  auto output = Resize(CpuOpContext(*context), input, 3, 2).value();
  for (int i = 0; i < 6; ++i)
    MYCV_EXPECT_EQ(output.Data<uint8_t>()[i], static_cast<uint8_t>(200));
  auto invalid = Resize(CpuOpContext(*context), input, 0, 2);
  MYCV_EXPECT_EQ(invalid.status().code(), StatusCode::kInvalidArgument);
}

MYCV_TEST(ResizeCpuLevelsAreBitExact) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input =
      context->AllocateTensor(Shape{1, 33, 41, 3}, DataType::kFloat32).value();
  for (int64_t i = 0; i < input.Numel(); ++i) {
    input.Data<float>()[i] = static_cast<float>(i % 29) * 0.125f;
  }
  auto reference = Resize(CpuOpContext(*context), input, 17, 23,
                          InterpolationMode::kBilinear, ImageLayout::kNHWC)
                       .value();
  for (CpuKernelLevel level :
       {CpuKernelLevel::kOpenMP, CpuKernelLevel::kSimdGeneric,
        CpuKernelLevel::kSimdAvx2, CpuKernelLevel::kTbb}) {
    if (!context->backend_ptr()->SupportsKernelLevel(level)) continue;
    OpContext level_context = CpuOpContext(*context);
    level_context.cpu_level = level;
    auto result = Resize(level_context, input, 17, 23,
                         InterpolationMode::kBilinear, ImageLayout::kNHWC);
    // Backend-wide capability may include levels this operator never built
    // (resize has no SIMD kernels); treat that as "level unavailable".
    if (!result.ok() && result.status().code() == StatusCode::kUnsupported) {
      continue;
    }
    MYCV_EXPECT_TRUE(result.ok());
    if (!result.ok()) continue;
    MYCV_EXPECT_TRUE(std::memcmp(reference.data(), result.value().data(),
                                 reference.ByteSize()) == 0);
  }
}

MYCV_TEST(ResizeRank2Float64PreservesShapeAndPrecision) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = context->AllocateTensor(Shape{2, 2}, DataType::kFloat64).value();
  const double values[] = {0, 1, 2, 3};
  for (int i = 0; i < 4; ++i)
    input.Data<double>()[i] = values[i];

  // Rank-2 is HW. Half-pixel-centre mapping from 2x2 to 3x3:
  //   src = (dst + 0.5) * 2/3 - 0.5, clamped to [0, 1].
  auto output = Resize(CpuOpContext(*context), input, 3, 3).value();
  MYCV_EXPECT_TRUE(output.shape() == Shape({3, 3}));
  MYCV_EXPECT_EQ(output.dtype(), DataType::kFloat64);
  MYCV_EXPECT_EQ(output.Data<double>()[0], 0.0);
  MYCV_EXPECT_EQ(output.Data<double>()[1], 0.5);
  MYCV_EXPECT_EQ(output.Data<double>()[4], 1.5);
  MYCV_EXPECT_EQ(output.Data<double>()[8], 3.0);
}

MYCV_TEST(ResizeOutputShapeFollowsLayout) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  struct Case {
    Shape input;
    ImageLayout layout;
    Shape expected;
  };
  const Case cases[] = {
      {Shape{5, 7, 3}, ImageLayout::kNHWC, Shape{4, 6, 3}},
      {Shape{3, 5, 7}, ImageLayout::kNCHW, Shape{3, 4, 6}},
      {Shape{2, 5, 7, 3}, ImageLayout::kNHWC, Shape{2, 4, 6, 3}},
      {Shape{2, 3, 5, 7}, ImageLayout::kNCHW, Shape{2, 3, 4, 6}},
  };
  for (const Case& item : cases) {
    auto input =
        context->AllocateTensor(item.input, DataType::kFloat32).value();
    auto output = Resize(CpuOpContext(*context), input, 4, 6,
                         InterpolationMode::kBilinear, item.layout);
    MYCV_EXPECT_TRUE(output.ok());
    if (!output.ok()) continue;
    MYCV_EXPECT_TRUE(output.value().shape() == item.expected);
    MYCV_EXPECT_TRUE(output.value().IsContiguous());
    MYCV_EXPECT_EQ(output.value().dtype(), DataType::kFloat32);
  }
}

MYCV_TEST(ResizeRejectsUnsupportedInputs) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto half = context->AllocateTensor(Shape{4, 4, 3}, DataType::kFloat16);
  if (half.ok()) {
    auto result = Resize(CpuOpContext(*context), half.value(), 2, 2);
    MYCV_EXPECT_EQ(result.status().code(), StatusCode::kUnsupported);
  }
  auto rank5 =
      context->AllocateTensor(Shape{1, 1, 2, 2, 2}, DataType::kFloat32).value();
  auto bad_rank = Resize(CpuOpContext(*context), rank5, 2, 2);
  MYCV_EXPECT_EQ(bad_rank.status().code(), StatusCode::kInvalidArgument);

  auto valid =
      context->AllocateTensor(Shape{2, 2, 3}, DataType::kFloat32).value();
  auto null_context = Resize(OpContext{}, valid, 2, 2);
  MYCV_EXPECT_EQ(null_context.status().code(), StatusCode::kInvalidArgument);
  auto negative = Resize(CpuOpContext(*context), valid, -1, 2);
  MYCV_EXPECT_EQ(negative.status().code(), StatusCode::kInvalidArgument);
}

MYCV_TEST(ResizeU8LevelsAreBitExact) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input =
      context->AllocateTensor(Shape{1, 33, 41, 3}, DataType::kUInt8).value();
  for (int64_t i = 0; i < input.Numel(); ++i) {
    input.Data<uint8_t>()[i] = static_cast<uint8_t>((i * 37) & 0xFF);
  }
  auto reference = Resize(CpuOpContext(*context), input, 17, 23,
                          InterpolationMode::kBilinear, ImageLayout::kNHWC)
                       .value();
  for (CpuKernelLevel level :
       {CpuKernelLevel::kOpenMP, CpuKernelLevel::kSimdGeneric,
        CpuKernelLevel::kSimdAvx2, CpuKernelLevel::kTbb}) {
    if (!context->backend_ptr()->SupportsKernelLevel(level)) continue;
    OpContext level_context = CpuOpContext(*context);
    level_context.cpu_level = level;
    auto result = Resize(level_context, input, 17, 23,
                         InterpolationMode::kBilinear, ImageLayout::kNHWC);
    // Backend-wide capability may include levels this operator never built
    // (resize has no SIMD kernels); treat that as "level unavailable".
    if (!result.ok() && result.status().code() == StatusCode::kUnsupported) {
      continue;
    }
    MYCV_EXPECT_TRUE(result.ok());
    if (!result.ok()) continue;
    MYCV_EXPECT_TRUE(std::memcmp(reference.data(), result.value().data(),
                                 reference.ByteSize()) == 0);
  }
}

#ifdef MYCV_WITH_OPENCV
namespace {

std::shared_ptr<Context> GetOpenCvContext() {
  static auto result = Context::Create(BackendType::kOpenCV);
  static const bool reported = [] {
    if (!result.ok())
      std::cout << "SKIP resize OpenCV tests: " << result.status().ToString()
                << '\n';
    return true;
  }();
  (void)reported;
  return result.ok() ? result.value() : nullptr;
}

}  // namespace

MYCV_TEST(ResizeOpenCvMatchesCpuU8) {
  auto cpu = Context::Create(BackendType::kNativeCpu).value();
  auto opencv = GetOpenCvContext();
  if (!opencv) return;
  auto input = cpu->AllocateTensor(Shape{24, 32, 3}, DataType::kUInt8).value();
  for (int64_t i = 0; i < input.Numel(); ++i) {
    input.Data<uint8_t>()[i] = static_cast<uint8_t>((i * 17) & 0xFF);
  }
  auto reference = Resize(CpuOpContext(*cpu), input, 11, 15).value();
  auto device = opencv->AllocateTensor(input.shape(), input.dtype()).value();
  MYCV_EXPECT_OK(opencv->Copy(&device, input));
  auto actual = Resize(CpuOpContext(*opencv), device, 11, 15);
  MYCV_EXPECT_TRUE(actual.ok());
  if (!actual.ok()) return;
  auto back =
      cpu->AllocateTensor(actual.value().shape(), actual.value().dtype())
          .value();
  MYCV_EXPECT_OK(cpu->Copy(&back, actual.value()));
  bool within_one = true;
  for (int64_t i = 0; i < back.Numel() && within_one; ++i) {
    // u8 cross-backend: ±1 absolute (fixed-point / rounding differences).
    within_one =
        std::fabs(static_cast<double>(back.Data<uint8_t>()[i]) -
                  static_cast<double>(reference.Data<uint8_t>()[i])) <= 1.0;
  }
  MYCV_EXPECT_TRUE(within_one);
}
#endif

#ifdef MYCV_WITH_CUDA
namespace {

std::shared_ptr<Context> GetCudaContext() {
  static auto result = Context::Create(BackendType::kCUDA);
  static const bool reported = [] {
    if (!result.ok())
      std::cout << "SKIP resize CUDA tests: " << result.status().ToString()
                << '\n';
    return true;
  }();
  (void)reported;
  return result.ok() ? result.value() : nullptr;
}

}  // namespace

MYCV_TEST(ResizeCudaMatchesCpuFloat) {
  auto cpu = Context::Create(BackendType::kNativeCpu).value();
  auto cuda = GetCudaContext();
  if (!cuda) return;
  auto input =
      cpu->AllocateTensor(Shape{24, 32, 3}, DataType::kFloat32).value();
  for (int64_t i = 0; i < input.Numel(); ++i) {
    input.Data<float>()[i] = static_cast<float>(i % 101) * 0.125f;
  }
  auto reference = Resize(CpuOpContext(*cpu), input, 11, 15).value();
  auto device = cuda->AllocateTensor(input.shape(), input.dtype()).value();
  MYCV_EXPECT_OK(cuda->Copy(&device, input));
  auto actual = Resize(CpuOpContext(*cuda), device, 11, 15);
  MYCV_EXPECT_TRUE(actual.ok());
  if (!actual.ok()) return;
  auto back =
      cpu->AllocateTensor(actual.value().shape(), actual.value().dtype())
          .value();
  MYCV_EXPECT_OK(cpu->Copy(&back, actual.value()));
  bool equivalent = true;
  for (int64_t i = 0; i < back.Numel() && equivalent; ++i) {
    const double a = reference.Data<float>()[i];
    const double b = back.Data<float>()[i];
    equivalent = std::fabs(a - b) <= 1e-4 * std::fmax(1.0, std::fabs(a));
  }
  MYCV_EXPECT_TRUE(equivalent);
}
#endif

}  // namespace

MYCV_TEST_MAIN()
