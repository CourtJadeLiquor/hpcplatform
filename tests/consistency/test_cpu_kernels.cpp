#include "test_framework.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "mycv/mycv.h"

namespace {

using namespace mycv;

OpContext MakeOpContext(Context& context, CpuKernelLevel level) {
  OpContext result;
  result.context = &context;
  result.stream = &context.stream();
  result.cpu_level = level;
  return result;
}

std::vector<CpuKernelLevel> AvailableLevels(Context& context) {
  std::vector<CpuKernelLevel> levels;
  for (CpuKernelLevel level : {CpuKernelLevel::kNaive, CpuKernelLevel::kOpenMP,
                               CpuKernelLevel::kTbb, CpuKernelLevel::kSimdGeneric,
                               CpuKernelLevel::kSimdAvx2, CpuKernelLevel::kSimdNeon}) {
    if (context.backend().SupportsKernelLevel(level)) levels.push_back(level);
  }
  return levels;
}

MYCV_TEST(ElementwiseKernelsAreBitExact) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto lhs = context->AllocateTensor(Shape{1027}, DataType::kFloat32).value();
  auto rhs = context->AllocateTensor(Shape{1027}, DataType::kFloat32).value();
  for (int i = 0; i < 1027; ++i) {
    lhs.Data<float>()[i] = static_cast<float>((i % 37) - 18) * 0.25f;
    rhs.Data<float>()[i] = static_cast<float>((i % 11) + 1) * 0.5f;
  }
  auto single_channel = lhs.View(Shape{1027, 1}).value();
  auto reference_add = Add(MakeOpContext(*context, CpuKernelLevel::kNaive), lhs, rhs).value();
  auto reference_mul = Mul(MakeOpContext(*context, CpuKernelLevel::kNaive), lhs, rhs).value();
  auto reference_norm = Normalize(MakeOpContext(*context, CpuKernelLevel::kNaive),
                                  single_channel, {0.2}, {1.3}).value();
  auto reference_cast = Cast(MakeOpContext(*context, CpuKernelLevel::kNaive),
                             lhs, DataType::kUInt8).value();

  for (CpuKernelLevel level : AvailableLevels(*context)) {
    auto added = Add(MakeOpContext(*context, level), lhs, rhs);
    MYCV_EXPECT_TRUE(added.ok());
    if (added.ok()) {
      MYCV_EXPECT_EQ(std::memcmp(added.value().data(), reference_add.data(),
                                 reference_add.ByteSize()), 0);
    }
    auto multiplied = Mul(MakeOpContext(*context, level), lhs, rhs);
    MYCV_EXPECT_TRUE(multiplied.ok());
    if (multiplied.ok()) {
      MYCV_EXPECT_EQ(std::memcmp(multiplied.value().data(), reference_mul.data(),
                                 reference_mul.ByteSize()), 0);
    }
    auto normalized = Normalize(MakeOpContext(*context, level), single_channel, {0.2}, {1.3});
    MYCV_EXPECT_TRUE(normalized.ok());
    if (normalized.ok()) {
      MYCV_EXPECT_EQ(std::memcmp(normalized.value().data(), reference_norm.data(),
                                 reference_norm.ByteSize()), 0);
    }
    auto cast = Cast(MakeOpContext(*context, level), lhs, DataType::kUInt8);
    if (level == CpuKernelLevel::kNaive || level == CpuKernelLevel::kSimdGeneric ||
        level == CpuKernelLevel::kSimdAvx2) {
      MYCV_EXPECT_TRUE(cast.ok());
      if (cast.ok()) {
        MYCV_EXPECT_EQ(std::memcmp(cast.value().data(), reference_cast.data(),
                                   reference_cast.ByteSize()), 0);
      }
    } else {
      MYCV_EXPECT_EQ(cast.status().code(), StatusCode::kUnsupported);
    }
  }
}

MYCV_TEST(ParallelImageKernelsMatchNaive) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto input = context->AllocateTensor(Shape{9, 11, 3}, DataType::kFloat32).value();
  for (int i = 0; i < 297; ++i) input.Data<float>()[i] = static_cast<float>(i % 29);
  auto reference_resize = Resize(MakeOpContext(*context, CpuKernelLevel::kNaive),
                                 input, 7, 5).value();
  auto reference_transpose = Transpose(MakeOpContext(*context, CpuKernelLevel::kNaive),
                                       input, {2, 0, 1}).value();

  for (CpuKernelLevel level : {CpuKernelLevel::kOpenMP, CpuKernelLevel::kTbb}) {
    if (!context->backend().SupportsKernelLevel(level)) continue;
    auto resized = Resize(MakeOpContext(*context, level), input, 7, 5);
    MYCV_EXPECT_TRUE(resized.ok());
    if (resized.ok()) {
      MYCV_EXPECT_EQ(std::memcmp(resized.value().data(), reference_resize.data(),
                                 reference_resize.ByteSize()), 0);
    }
    auto transposed = Transpose(MakeOpContext(*context, level), input, {2, 0, 1});
    MYCV_EXPECT_TRUE(transposed.ok());
    if (transposed.ok()) {
      MYCV_EXPECT_EQ(std::memcmp(transposed.value().data(), reference_transpose.data(),
                                 reference_transpose.ByteSize()), 0);
    }
  }
}

MYCV_TEST(SimdRejectsStridedInputInsteadOfChangingKernel) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  if (!context->backend().SupportsKernelLevel(CpuKernelLevel::kSimdGeneric)) return;
  auto base = context->AllocateTensor(Shape{4, 4}, DataType::kFloat32).value();
  auto view = base.Slice(1, 1, 3).value();
  auto result = Add(MakeOpContext(*context, CpuKernelLevel::kSimdGeneric), view, view);
  MYCV_EXPECT_EQ(result.status().code(), StatusCode::kUnsupported);
}

}  // namespace

MYCV_TEST_MAIN()
