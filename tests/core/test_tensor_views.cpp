// 测试接口用于验证 Slice / Reshape / Transpose 等零拷贝视图的元数据与共享
// Buffer 语义。
#include "test_framework.h"

#include <cstdint>
#include <vector>

#include "mycv/core/context.h"
#include "mycv/core/tensor_iterator.h"

namespace {

using namespace mycv;

MYCV_TEST(ViewSliceAndPermutePreserveLogicalElements) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto tensor =
      context->AllocateTensor(Shape{2, 3, 4}, DataType::kFloat32).value();
  for (int i = 0; i < 24; ++i)
    tensor.Data<float>()[i] = static_cast<float>(i);

  auto view = tensor.View(Shape{6, 4}).value();
  MYCV_EXPECT_TRUE(view.IsContiguous());
  MYCV_EXPECT_EQ(view.buffer().get(), tensor.buffer().get());
  view.Data<float>()[5] = 99.0f;
  MYCV_EXPECT_EQ(tensor.Data<float>()[5], 99.0f);
  tensor.Data<float>()[5] = 5.0f;

  auto slice = tensor.Slice(1, 1, 3).value();
  MYCV_EXPECT_EQ(slice.Numel(), static_cast<int64_t>(16));
  MYCV_EXPECT_EQ(slice.Stride(0), static_cast<int64_t>(12));
  MYCV_EXPECT_EQ(slice.Stride(1), static_cast<int64_t>(4));
  MYCV_EXPECT_FALSE(slice.IsContiguous());
  MYCV_EXPECT_EQ(slice.Data<float>()[0], 4.0f);

  TensorIterator iterator(slice);
  std::vector<float> actual;
  while (iterator.valid()) {
    actual.push_back(*static_cast<const float*>(iterator.data()));
    iterator.Next();
  }
  const std::vector<float> expected{4,  5,  6,  7,  8,  9,  10, 11,
                                    16, 17, 18, 19, 20, 21, 22, 23};
  MYCV_EXPECT_TRUE(actual == expected);

  auto permuted = tensor.Permute({2, 0, 1}).value();
  MYCV_EXPECT_EQ(permuted.shape().Dim(0), static_cast<int64_t>(4));
  MYCV_EXPECT_EQ(permuted.shape().Dim(1), static_cast<int64_t>(2));
  MYCV_EXPECT_EQ(permuted.shape().Dim(2), static_cast<int64_t>(3));
  MYCV_EXPECT_EQ(permuted.Stride(0), static_cast<int64_t>(1));
  MYCV_EXPECT_EQ(permuted.Stride(1), static_cast<int64_t>(12));
  MYCV_EXPECT_EQ(permuted.Stride(2), static_cast<int64_t>(4));
}

MYCV_TEST(ReshapeOfStridedViewClonesIntoLogicalOrder) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto tensor = context->AllocateTensor(Shape{2, 4}, DataType::kInt32).value();
  for (int i = 0; i < 8; ++i)
    tensor.Data<int32_t>()[i] = i;
  auto slice = tensor.Slice(1, 1, 3).value();
  auto reshaped = slice.Reshape(Shape{4}).value();

  MYCV_EXPECT_TRUE(reshaped.IsContiguous());
  MYCV_EXPECT_TRUE(reshaped.buffer().get() != tensor.buffer().get());
  const std::vector<int32_t> expected{1, 2, 5, 6};
  for (size_t i = 0; i < expected.size(); ++i) {
    MYCV_EXPECT_EQ(reshaped.Data<int32_t>()[i], expected[i]);
  }
}

MYCV_TEST(CloneIsDeepAndInvalidViewsAreRejected) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto tensor = context->AllocateTensor(Shape{3}, DataType::kFloat32).value();
  tensor.Data<float>()[0] = 7.0f;
  auto clone = tensor.Clone().value();
  MYCV_EXPECT_TRUE(clone.buffer().get() != tensor.buffer().get());
  clone.Data<float>()[0] = 8.0f;
  MYCV_EXPECT_EQ(tensor.Data<float>()[0], 7.0f);
  MYCV_EXPECT_FALSE(tensor.View(Shape{4}).ok());
  MYCV_EXPECT_FALSE(tensor.Permute({0, 0}).ok());
  MYCV_EXPECT_FALSE(tensor.Slice(0, 2, 2).ok());
}

MYCV_TEST(TensorCanOutliveItsContext) {
  Tensor survivor;
  {
    auto context = Context::Create(BackendType::kNativeCpu).value();
    survivor = context->AllocateTensor(Shape{8}, DataType::kFloat64).value();
    survivor.Data<double>()[7] = 42.0;
  }
  MYCV_EXPECT_EQ(survivor.Data<double>()[7], 42.0);
}

}  // namespace

MYCV_TEST_MAIN()
