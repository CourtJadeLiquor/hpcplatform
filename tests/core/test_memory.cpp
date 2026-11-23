// Tests for Buffer / MemoryManager allocation, alignment, counting, and
// deallocation (including custom deleters).
#include "test_utils.h"

#include <cstring>
#include <vector>

#include "hpcplatform/core/context.h"

namespace {

using namespace hpcplatform;

TEST(hpcplatform, CpuMemoryManagerCopiesBetweenBuffers) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto src = context->AllocateTensor(Shape{4}, DataType::kFloat32).value();
  auto dst = context->AllocateTensor(Shape{4}, DataType::kFloat32).value();

  float* src_data = src.Data<float>();
  for (int i = 0; i < 4; ++i)
    src_data[i] = static_cast<float>(i) + 1.0f;

  hpcplatform_EXPECT_OK(context->memory().Copy(
      dst.buffer().get(), 0, src.buffer().get(), 0, src.ByteSize(), nullptr));
  EXPECT_EQ(std::memcmp(src.data(), dst.data(), src.ByteSize()),
                 static_cast<int>(0));
  EXPECT_EQ(dst.Data<float>()[3], 4.0f);
}

TEST(hpcplatform, MemoryStatsTrackLifetime) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  const MemoryStats before = context->memory().Stats();

  {
    auto first = context->AllocateTensor(Shape{8}, DataType::kFloat32).value();
    auto second = context->AllocateTensor(Shape{8}, DataType::kFloat32).value();
    const MemoryStats during = context->memory().Stats();
    EXPECT_EQ(during.allocated_bytes, before.allocated_bytes + 64);
    EXPECT_EQ(during.allocation_count, before.allocation_count + 2);
    EXPECT_TRUE(during.peak_bytes >= during.allocated_bytes);
  }

  const MemoryStats after = context->memory().Stats();
  EXPECT_EQ(after.allocated_bytes, before.allocated_bytes);
  EXPECT_EQ(after.allocation_count, before.allocation_count);
}

TEST(hpcplatform, CopyRejectsOutOfBoundsRange) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto src = context->AllocateTensor(Shape{4}, DataType::kFloat32).value();
  auto dst = context->AllocateTensor(Shape{4}, DataType::kFloat32).value();

  const Status status = context->memory().Copy(
      dst.buffer().get(), 8, src.buffer().get(), 0, src.ByteSize(), nullptr);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(hpcplatform, AllocationRejectsInvalidAlignment) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  EXPECT_FALSE(context->memory().Allocate(64, 3).ok());
  EXPECT_FALSE(context->memory().Allocate(0, 64).ok());
}

}  // namespace


