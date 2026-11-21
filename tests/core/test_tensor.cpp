#include "test_framework.h"

#include <cstdint>

#include "mycv/core/context.h"
#include "mycv/core/tensor.h"

namespace {

using namespace mycv;

Result<std::shared_ptr<Context>> MakeCpuContext() {
  return Context::Create(BackendType::kNativeCpu);
}

MYCV_TEST(CreateTensorAllocatesContiguousBuffer) {
  auto context_result = MakeCpuContext();
  MYCV_EXPECT_TRUE(context_result.ok());
  auto context = context_result.value();

  auto tensor_result = Tensor::Create(*context, Shape{2, 3}, DataType::kFloat32);
  MYCV_EXPECT_TRUE(tensor_result.ok());
  const Tensor& tensor = tensor_result.value();

  MYCV_EXPECT_TRUE(tensor.defined());
  MYCV_EXPECT_EQ(tensor.Numel(), static_cast<int64_t>(6));
  MYCV_EXPECT_EQ(tensor.ByteSize(), static_cast<size_t>(24));
  MYCV_EXPECT_EQ(tensor.ElementSize(), static_cast<size_t>(4));
  MYCV_EXPECT_TRUE(tensor.IsContiguous());
  MYCV_EXPECT_EQ(tensor.Stride(1), static_cast<int64_t>(1));
  MYCV_EXPECT_EQ(tensor.Stride(0), static_cast<int64_t>(3));
  MYCV_EXPECT_EQ(tensor.device().type, DeviceType::kCPU);
  MYCV_EXPECT_EQ(tensor.backend(), BackendType::kNativeCpu);
  MYCV_EXPECT_TRUE(tensor.data() != nullptr);
  MYCV_EXPECT_EQ(reinterpret_cast<uintptr_t>(tensor.data()) % 64,
                 static_cast<uintptr_t>(0));
}

MYCV_TEST(CreateTensorRejectsInvalidShapeAndDtype) {
  auto context = MakeCpuContext().value();
  const Shape zero_dim{0, 3};
  const Shape negative{-1, 3};
  MYCV_EXPECT_FALSE(Tensor::Create(*context, zero_dim, DataType::kFloat32).ok());
  MYCV_EXPECT_FALSE(Tensor::Create(*context, negative, DataType::kFloat32).ok());
}

MYCV_TEST(TensorWriteReadRoundTrip) {
  auto context = MakeCpuContext().value();
  auto tensor = Tensor::Create(*context, Shape{2, 3}, DataType::kFloat32).value();
  float* data = tensor.Data<float>();
  for (int i = 0; i < 6; ++i) data[i] = static_cast<float>(i) * 0.5f;
  MYCV_EXPECT_EQ(data[4], 2.0f);
  MYCV_EXPECT_TRUE(tensor.ToString().find("shape=2x3") != std::string::npos);
}

MYCV_TEST(ScalarTensorHasOneElement) {
  auto context = MakeCpuContext().value();
  auto tensor = Tensor::Create(*context, Shape{}, DataType::kInt32).value();
  MYCV_EXPECT_EQ(tensor.Numel(), static_cast<int64_t>(1));
  MYCV_EXPECT_TRUE(tensor.IsContiguous());
}

}  // namespace

MYCV_TEST_MAIN()
