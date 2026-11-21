// 测试接口用于验证 Shape 的构造、Numel/Strides 计算与非法维度校验。
#include "test_framework.h"

#include "mycv/core/shape.h"

namespace {

using mycv::Shape;

MYCV_TEST(ShapeRankAndNumel) {
  const Shape shape{2, 3, 4};
  const Shape scalar;
  MYCV_EXPECT_EQ(shape.Rank(), 3);
  MYCV_EXPECT_EQ(shape.Numel(), static_cast<int64_t>(24));
  MYCV_EXPECT_EQ(shape.Dim(1), static_cast<int64_t>(3));
  MYCV_EXPECT_TRUE(scalar.IsScalar());
  MYCV_EXPECT_EQ(scalar.Numel(), static_cast<int64_t>(1));
}

MYCV_TEST(ShapeValidateRejectsNonPositiveDims) {
  const Shape valid{2, 3};
  const Shape zero_dim{2, 0};
  const Shape negative{-1, 3};
  MYCV_EXPECT_TRUE(valid.Validate().ok());
  MYCV_EXPECT_FALSE(zero_dim.Validate().ok());
  MYCV_EXPECT_FALSE(negative.Validate().ok());
}

MYCV_TEST(ShapeEquality) {
  const Shape a{1, 2};
  const Shape b{1, 2};
  const Shape transposed{2, 1};
  const Shape ranked{1, 2, 1};
  MYCV_EXPECT_TRUE(a == b);
  MYCV_EXPECT_TRUE(a != transposed);
  MYCV_EXPECT_TRUE(a != ranked);
}

MYCV_TEST(ShapeToString) {
  const Shape shape{2, 3};
  const Shape scalar;
  MYCV_EXPECT_EQ(shape.ToString(), std::string("2x3"));
  MYCV_EXPECT_EQ(scalar.ToString(), std::string("scalar"));
}

}  // namespace

MYCV_TEST_MAIN()
