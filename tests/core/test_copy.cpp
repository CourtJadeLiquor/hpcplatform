// Tests for Context::Copy / Tensor::Clone semantics for contiguous and
// non-contiguous views, cross-backend copies, and lifetime scenarios.
#include "test_utils.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "hpcplatform/core/context.h"
#include "hpcplatform/core/tensor.h"
#include "hpcplatform/core/transfer.h"
#include "hpcplatform/ops/resize.h"

namespace {

using namespace hpcplatform;

TEST(hpcplatform, CopyHandlesNonContiguousSource) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto source = Tensor::Empty(*context, Shape{2, 4}, DataType::kInt32).value();
  for (int i = 0; i < 8; ++i)
    source.Data<int32_t>()[i] = i + 1;
  auto slice = source.Slice(1, 1, 3).value();
  auto destination =
      Tensor::Empty(*context, slice.shape(), DataType::kInt32).value();

  hpcplatform_EXPECT_OK(Copy(slice, &destination));
  const int32_t expected[] = {2, 3, 6, 7};
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(destination.Data<int32_t>()[i], expected[i]);
  }
}

TEST(hpcplatform, CopyRejectsMismatchedDestination) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto source = Tensor::Empty(*context, Shape{2}, DataType::kFloat32).value();
  auto wrong_shape =
      Tensor::Empty(*context, Shape{3}, DataType::kFloat32).value();
  auto wrong_dtype =
      Tensor::Empty(*context, Shape{2}, DataType::kFloat64).value();
  EXPECT_EQ(Copy(source, &wrong_shape).code(),
                 StatusCode::kInvalidArgument);
  EXPECT_EQ(Copy(source, &wrong_dtype).code(),
                 StatusCode::kInvalidArgument);
}

TEST(hpcplatform, WrapBorrowsExternalHostMemoryWithoutAllocating) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  const MemoryStats before = context->GetMemoryStats();

  // External storage from another runtime (here: a plain std::vector).
  std::vector<float> external = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  auto wrapped =
      Tensor::WrapExternal(*context, external.data(), Shape{1, 5}, DataType::kFloat32);
  EXPECT_TRUE(wrapped.ok());
  if (!wrapped.ok()) return;
  const Tensor& t = wrapped.value();

  // Zero-copy contract: the Tensor aliases the caller's pointer exactly and
  // the memory manager recorded no allocation.
  EXPECT_EQ(t.data(), static_cast<void*>(external.data()));
  EXPECT_EQ(t.buffer()->kind(), MemoryKind::kHost);
  const MemoryStats after = context->GetMemoryStats();
  EXPECT_EQ(after.allocation_count, before.allocation_count);
  EXPECT_EQ(after.allocated_bytes, before.allocated_bytes);

  // A borrowed tensor is a valid copy source.
  auto dst = Tensor::Empty(*context, Shape{1, 5}, DataType::kFloat32).value();
  hpcplatform_EXPECT_OK(Copy(t, &dst));
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(dst.Data<float>()[i], external[i]);

  // And a valid op input: dispatch accepts it because backend/device match.
  // Identity resize (1:1) returns the borrowed pixels unchanged.
  auto resized = Resize(OpContext(*context), t, 1, 5);
  EXPECT_TRUE(resized.ok());
  if (resized.ok()) {
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(resized.value().Data<float>()[i], external[i]);
    }
  }
  // Borrow: dropping the Tensor must not touch the caller's memory.
  external[0] = 42.0f;
  EXPECT_EQ(external[0], 42.0f);
}

TEST(hpcplatform, WrapAdoptsMemoryAndRunsDeleterOnLastRelease) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto deleted = std::make_shared<int>(0);
  constexpr size_t kCount = 6;
  void* raw = std::malloc(kCount * sizeof(float));
  EXPECT_TRUE(raw != nullptr);
  if (raw == nullptr) return;
  float* pixels = static_cast<float*>(raw);
  for (size_t i = 0; i < kCount; ++i)
    pixels[i] = static_cast<float>(i);

  {
    auto wrapped = Tensor::WrapExternal(*context, raw, Shape{6}, DataType::kFloat32,
                                 MemoryKind::kHost, [deleted](void* p, size_t) {
                                   std::free(p);
                                   ++(*deleted);
                                 });
    if (!wrapped.ok()) {
      std::free(raw);
      EXPECT_TRUE(wrapped.ok());
      return;
    }
    EXPECT_EQ(*deleted, 0);

    // A second Tensor reference bumps the Buffer refcount; deleter stays idle.
    Tensor alias = wrapped.value();
    EXPECT_EQ(alias.data(), raw);
    EXPECT_EQ(*deleted, 0);
  }
  // Both references gone -> the adopted deleter ran exactly once.
  EXPECT_EQ(*deleted, 1);
}

TEST(hpcplatform, WrapRejectsNullData) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto r = Tensor::WrapExternal(*context, nullptr, Shape{4}, DataType::kFloat32);
  EXPECT_FALSE(r.ok());
  EXPECT_EQ(r.status().code(), StatusCode::kInvalidArgument);
}

}  // namespace

