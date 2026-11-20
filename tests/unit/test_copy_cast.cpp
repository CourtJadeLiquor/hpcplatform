#include "test_framework.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "mycv/core/context.h"
#include "mycv/ops/cast.h"

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

  auto bytes = Cast(OpContext{context.get()}, source, DataType::kUInt8).value();
  const uint8_t expected[] = {0, 0, 1, 255, 3};
  for (int i = 0; i < 5; ++i) MYCV_EXPECT_EQ(bytes.Data<uint8_t>()[i], expected[i]);

  auto half = Cast(OpContext{context.get()}, source, DataType::kFloat16).value();
  MYCV_EXPECT_EQ(half.ElementSize(), static_cast<size_t>(2));
  auto restored = Cast(OpContext{context.get()}, half, DataType::kFloat32).value();
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
  auto cast = Cast(OpContext{context.get()}, slice, DataType::kFloat64).value();
  const double expected[] = {1, 2, 4, 5};
  for (int i = 0; i < 4; ++i) MYCV_EXPECT_EQ(cast.Data<double>()[i], expected[i]);
}

MYCV_TEST(WrapBorrowsExternalHostMemoryWithoutAllocating) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  const MemoryStats before = context->memory().Stats();

  // External storage from another runtime (here: a plain std::vector).
  std::vector<float> external = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  auto wrapped = context->Wrap(external.data(), Shape{5}, DataType::kFloat32);
  MYCV_EXPECT_TRUE(wrapped.ok());
  if (!wrapped.ok()) return;
  const Tensor& t = wrapped.value();

  // Zero-copy contract: the Tensor aliases the caller's pointer exactly and
  // the memory manager recorded no allocation.
  MYCV_EXPECT_EQ(t.data(), static_cast<void*>(external.data()));
  MYCV_EXPECT_EQ(t.buffer()->kind(), MemoryKind::kHost);
  const MemoryStats after = context->memory().Stats();
  MYCV_EXPECT_EQ(after.allocation_count, before.allocation_count);
  MYCV_EXPECT_EQ(after.allocated_bytes, before.allocated_bytes);

  // A borrowed tensor is a valid copy source.
  auto dst = context->AllocateTensor(Shape{5}, DataType::kFloat32).value();
  MYCV_EXPECT_OK(context->Copy(&dst, t));
  for (int i = 0; i < 5; ++i) MYCV_EXPECT_EQ(dst.Data<float>()[i], external[i]);

  // And a valid op input: dispatch accepts it because backend/device match.
  auto casted = Cast(OpContext{context.get()}, t, DataType::kFloat64);
  MYCV_EXPECT_TRUE(casted.ok());
  if (casted.ok()) {
    for (int i = 0; i < 5; ++i) {
      MYCV_EXPECT_EQ(casted.value().Data<double>()[i],
                     static_cast<double>(external[i]));
    }
  }
  // Borrow: dropping the Tensor must not touch the caller's memory.
  external[0] = 42.0f;
  MYCV_EXPECT_EQ(external[0], 42.0f);
}

MYCV_TEST(WrapAdoptsMemoryAndRunsDeleterOnLastRelease) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto deleted = std::make_shared<int>(0);
  constexpr size_t kCount = 6;
  void* raw = std::malloc(kCount * sizeof(float));
  MYCV_EXPECT_TRUE(raw != nullptr);
  if (raw == nullptr) return;
  float* pixels = static_cast<float*>(raw);
  for (size_t i = 0; i < kCount; ++i) pixels[i] = static_cast<float>(i);

  {
    auto wrapped =
        context->Wrap(raw, Shape{6}, DataType::kFloat32, MemoryKind::kHost,
                      [deleted](void* p, size_t) {
                        std::free(p);
                        ++(*deleted);
                      });
    if (!wrapped.ok()) {
      std::free(raw);
      MYCV_EXPECT_TRUE(wrapped.ok());
      return;
    }
    MYCV_EXPECT_EQ(*deleted, 0);

    // A second Tensor reference bumps the Buffer refcount; deleter stays idle.
    Tensor alias = wrapped.value();
    MYCV_EXPECT_EQ(alias.data(), raw);
    MYCV_EXPECT_EQ(*deleted, 0);
  }
  // Both references gone -> the adopted deleter ran exactly once.
  MYCV_EXPECT_EQ(*deleted, 1);
}

MYCV_TEST(WrapRejectsNullData) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto r = context->Wrap(nullptr, Shape{4}, DataType::kFloat32);
  MYCV_EXPECT_FALSE(r.ok());
  MYCV_EXPECT_EQ(r.status().code(), StatusCode::kInvalidArgument);
}

}  // namespace

MYCV_TEST_MAIN()
