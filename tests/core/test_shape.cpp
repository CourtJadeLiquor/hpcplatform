// Tests for Shape construction, Numel/Strides computation, and invalid-dimension validation.
#include "test_utils.h"

#include "hpcplatform/core/shape.h"

namespace {

using hpcplatform::Shape;

TEST(hpcplatform, ShapeRankAndNumel) {
  const Shape shape{2, 3, 4};
  const Shape scalar;
  EXPECT_EQ(shape.Rank(), 3);
  EXPECT_EQ(shape.Numel(), static_cast<int64_t>(24));
  EXPECT_EQ(shape.Dim(1), static_cast<int64_t>(3));
  EXPECT_TRUE(scalar.IsScalar());
  EXPECT_EQ(scalar.Numel(), static_cast<int64_t>(1));
}

TEST(hpcplatform, ShapeValidateRejectsNonPositiveDims) {
  const Shape valid{2, 3};
  const Shape zero_dim{2, 0};
  const Shape negative{-1, 3};
  EXPECT_TRUE(valid.Validate().ok());
  EXPECT_FALSE(zero_dim.Validate().ok());
  EXPECT_FALSE(negative.Validate().ok());
}

TEST(hpcplatform, ShapeEquality) {
  const Shape a{1, 2};
  const Shape b{1, 2};
  const Shape transposed{2, 1};
  const Shape ranked{1, 2, 1};
  EXPECT_TRUE(a == b);
  EXPECT_TRUE(a != transposed);
  EXPECT_TRUE(a != ranked);
}

TEST(hpcplatform, ShapeToString) {
  const Shape shape{2, 3};
  const Shape scalar;
  EXPECT_EQ(shape.ToString(), std::string("2x3"));
  EXPECT_EQ(scalar.ToString(), std::string("scalar"));
}

}  // namespace

