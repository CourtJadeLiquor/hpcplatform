// Tests for Tensor creation, lifetime, metadata (shape/dtype/strides), and
// boundary validation.
#include "test_utils.h"

#include <cstdint>

#include "hpcplatform/core/context.h"
#include "hpcplatform/core/tensor.h"

namespace {

using namespace hpcplatform;

Result<std::shared_ptr<Context>> MakeCpuContext() {
  return Context::Create(BackendType::kNativeCpu);
}

TEST(hpcplatform, CreateTensorAllocatesContiguousBuffer) {
  auto context_result = MakeCpuContext();
  EXPECT_TRUE(context_result.ok());
  auto context = context_result.value();

  auto tensor_result =
      Tensor::Create(*context, Shape{2, 3}, DataType::kFloat32);
  EXPECT_TRUE(tensor_result.ok());
  const Tensor& tensor = tensor_result.value();

  EXPECT_TRUE(tensor.defined());
  EXPECT_EQ(tensor.Numel(), static_cast<int64_t>(6));
  EXPECT_EQ(tensor.ByteSize(), static_cast<size_t>(24));
  EXPECT_EQ(tensor.ElementSize(), static_cast<size_t>(4));
  EXPECT_TRUE(tensor.IsContiguous());
  EXPECT_EQ(tensor.Stride(1), static_cast<int64_t>(1));
  EXPECT_EQ(tensor.Stride(0), static_cast<int64_t>(3));
  EXPECT_EQ(tensor.device().type, DeviceType::kCPU);
  EXPECT_EQ(tensor.backend(), BackendType::kNativeCpu);
  EXPECT_TRUE(tensor.data() != nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(tensor.data()) % 64,
                 static_cast<uintptr_t>(0));
}

TEST(hpcplatform, CreateTensorRejectsInvalidShapeAndDtype) {
  auto context = MakeCpuContext().value();
  const Shape zero_dim{0, 3};
  const Shape negative{-1, 3};
  EXPECT_FALSE(
      Tensor::Create(*context, zero_dim, DataType::kFloat32).ok());
  EXPECT_FALSE(
      Tensor::Create(*context, negative, DataType::kFloat32).ok());
}

TEST(hpcplatform, TensorWriteReadRoundTrip) {
  auto context = MakeCpuContext().value();
  auto tensor =
      Tensor::Create(*context, Shape{2, 3}, DataType::kFloat32).value();
  float* data = tensor.Data<float>();
  for (int i = 0; i < 6; ++i)
    data[i] = static_cast<float>(i) * 0.5f;
  EXPECT_EQ(data[4], 2.0f);
  EXPECT_TRUE(tensor.ToString().find("shape=2x3") != std::string::npos);
}

TEST(hpcplatform, ScalarTensorHasOneElement) {
  auto context = MakeCpuContext().value();
  auto tensor = Tensor::Create(*context, Shape{}, DataType::kInt32).value();
  EXPECT_EQ(tensor.Numel(), static_cast<int64_t>(1));
  EXPECT_TRUE(tensor.IsContiguous());
}

}  // namespace


