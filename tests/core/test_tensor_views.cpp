// Tests for metadata and shared Buffer semantics of zero-copy views such as
// Slice / Reshape / Transpose.
#include "test_utils.h"

#include <cstdint>
#include <vector>

#include "core/tensor_iterator.h"
#include "hpcplatform/core/context.h"

namespace {

using namespace hpcplatform;

TEST(hpcplatform, ViewSliceAndPermutePreserveLogicalElements) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto tensor =
      Tensor::Empty(*context, Shape{2, 3, 4}, DataType::kFloat32).value();
  for (int i = 0; i < 24; ++i)
    tensor.Data<float>()[i] = static_cast<float>(i);

  auto view = tensor.View(Shape{6, 4}).value();
  EXPECT_TRUE(view.IsContiguous());
  EXPECT_EQ(view.buffer().get(), tensor.buffer().get());
  view.Data<float>()[5] = 99.0f;
  EXPECT_EQ(tensor.Data<float>()[5], 99.0f);
  tensor.Data<float>()[5] = 5.0f;

  auto slice = tensor.Slice(1, 1, 3).value();
  EXPECT_EQ(slice.Numel(), static_cast<int64_t>(16));
  EXPECT_EQ(slice.Stride(0), static_cast<int64_t>(12));
  EXPECT_EQ(slice.Stride(1), static_cast<int64_t>(4));
  EXPECT_FALSE(slice.IsContiguous());
  EXPECT_EQ(slice.Data<float>()[0], 4.0f);

  TensorIterator iterator(slice);
  std::vector<float> actual;
  while (iterator.valid()) {
    actual.push_back(*static_cast<const float*>(iterator.data()));
    iterator.Next();
  }
  const std::vector<float> expected{4,  5,  6,  7,  8,  9,  10, 11,
                                    16, 17, 18, 19, 20, 21, 22, 23};
  EXPECT_TRUE(actual == expected);

  auto permuted = tensor.Permute({2, 0, 1}).value();
  EXPECT_EQ(permuted.shape().Dim(0), static_cast<int64_t>(4));
  EXPECT_EQ(permuted.shape().Dim(1), static_cast<int64_t>(2));
  EXPECT_EQ(permuted.shape().Dim(2), static_cast<int64_t>(3));
  EXPECT_EQ(permuted.Stride(0), static_cast<int64_t>(1));
  EXPECT_EQ(permuted.Stride(1), static_cast<int64_t>(12));
  EXPECT_EQ(permuted.Stride(2), static_cast<int64_t>(4));
}

TEST(hpcplatform, ReshapeOfStridedViewClonesIntoLogicalOrder) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto tensor = Tensor::Empty(*context, Shape{2, 4}, DataType::kInt32).value();
  for (int i = 0; i < 8; ++i)
    tensor.Data<int32_t>()[i] = i;
  auto slice = tensor.Slice(1, 1, 3).value();
  auto reshaped = slice.Reshape(Shape{4}).value();

  EXPECT_TRUE(reshaped.IsContiguous());
  EXPECT_TRUE(reshaped.buffer().get() != tensor.buffer().get());
  const std::vector<int32_t> expected{1, 2, 5, 6};
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(reshaped.Data<int32_t>()[i], expected[i]);
  }
}

TEST(hpcplatform, CloneIsDeepAndInvalidViewsAreRejected) {
  auto context = Context::Create(BackendType::kNativeCpu).value();
  auto tensor = Tensor::Empty(*context, Shape{3}, DataType::kFloat32).value();
  tensor.Data<float>()[0] = 7.0f;
  auto clone = tensor.Clone().value();
  EXPECT_TRUE(clone.buffer().get() != tensor.buffer().get());
  clone.Data<float>()[0] = 8.0f;
  EXPECT_EQ(tensor.Data<float>()[0], 7.0f);
  EXPECT_FALSE(tensor.View(Shape{4}).ok());
  EXPECT_FALSE(tensor.Permute({0, 0}).ok());
  EXPECT_FALSE(tensor.Slice(0, 2, 2).ok());
}

TEST(hpcplatform, TensorCanOutliveItsContext) {
  Tensor survivor;
  {
    auto context = Context::Create(BackendType::kNativeCpu).value();
    survivor = Tensor::Empty(*context, Shape{8}, DataType::kFloat64).value();
    survivor.Data<double>()[7] = 42.0;
  }
  EXPECT_EQ(survivor.Data<double>()[7], 42.0);
}

}  // namespace

