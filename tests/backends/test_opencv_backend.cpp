// Tests for OpenCV backend image wrapping, cross-runtime memory adoption, and copyTo semantics.
#include "test_utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <opencv2/core.hpp>

#include "hpcplatform/core/transfer.h"
#include "hpcplatform/hpcplatform.h"

namespace {

using namespace hpcplatform;

OpContext Ops(Context& context, CpuKernelLevel level = CpuKernelLevel::kNaive) {
  OpContext result(context);
  result.set_cpu_level(level);
  return result;
}

Result<Tensor> CopyTo(Context& destination, const Tensor& source) {
  auto output = Tensor::Empty(destination, source.shape(), source.dtype());
  if (!output.ok()) return output.status();
  hpcplatform_RETURN_IF_ERROR(Copy(source, &output.value()));
  return output;
}

void ExpectEquivalent(const Tensor& expected, const Tensor& actual,
                      Context& native, double tolerance = 1e-4) {
  EXPECT_EQ(actual.shape() == expected.shape(), true);
  EXPECT_EQ(actual.dtype(), expected.dtype());
  if (actual.shape() != expected.shape() || actual.dtype() != expected.dtype())
    return;
  auto host = CopyTo(native, actual);
  EXPECT_TRUE(host.ok());
  if (!host.ok()) return;
  for (int64_t i = 0; i < expected.Numel(); ++i) {
    if (expected.dtype() == DataType::kFloat32) {
      const double lhs = expected.Data<float>()[i];
      const double rhs = host.value().Data<float>()[i];
      EXPECT_TRUE(std::fabs(lhs - rhs) <=
                       tolerance * std::max(1.0, std::fabs(lhs)));
    } else if (expected.dtype() == DataType::kFloat64) {
      const double lhs = expected.Data<double>()[i];
      const double rhs = host.value().Data<double>()[i];
      EXPECT_TRUE(std::fabs(lhs - rhs) <=
                       tolerance * std::max(1.0, std::fabs(lhs)));
    } else if (expected.dtype() == DataType::kUInt8) {
      const int lhs = expected.Data<uint8_t>()[i];
      const int rhs = host.value().Data<uint8_t>()[i];
      const int allowed_error = tolerance == 0.0 ? 0 : 1;
      EXPECT_TRUE(std::abs(lhs - rhs) <= allowed_error);
    } else {
      EXPECT_EQ(std::memcmp(expected.data(), host.value().data(),
                                 expected.ByteSize()),
                     0);
      return;
    }
  }
}

TEST(hpcplatform, CvMatConversionCopiesRoiAndOwnsStorage) {
  auto context = Context::Create(BackendType::kOpenCV).value();
  cv::Mat source(3, 4, CV_32FC3);
  for (int y = 0; y < source.rows; ++y) {
    for (int x = 0; x < source.cols; ++x) {
      source.at<cv::Vec3f>(y, x) =
          cv::Vec3f(static_cast<float>(y * 12 + x * 3),
                    static_cast<float>(y * 12 + x * 3 + 1),
                    static_cast<float>(y * 12 + x * 3 + 2));
    }
  }
  cv::Mat roi = source(cv::Rect(1, 1, 2, 2));
  EXPECT_FALSE(roi.isContinuous());
  auto tensor = Tensor::From(roi, *context);
  EXPECT_TRUE(tensor.ok());
  if (!tensor.ok()) return;
  const Shape expected_shape{2, 2, 3};
  EXPECT_TRUE(tensor.value().shape() == expected_shape);
  EXPECT_EQ(tensor.value().Data<float>()[0], 15.0f);
  source.setTo(cv::Scalar::all(0));
  EXPECT_EQ(tensor.value().Data<float>()[0], 15.0f);

  auto restored = Tensor::Into<cv::Mat>(tensor.value());
  EXPECT_TRUE(restored.ok());
  if (!restored.ok()) return;
  EXPECT_EQ(restored.value().rows, 2);
  EXPECT_EQ(restored.value().cols, 2);
  EXPECT_EQ(restored.value().channels(), 3);
  EXPECT_EQ(restored.value().at<cv::Vec3f>(0, 0)[0], 15.0f);
}

TEST(hpcplatform, NativeAndOpenCvCopyBothWays) {
  auto native = Context::Create(BackendType::kNativeCpu).value();
  auto opencv = Context::Create(BackendType::kOpenCV).value();
  auto input = Tensor::Empty(*native, Shape{2, 3}, DataType::kFloat32).value();
  for (int i = 0; i < 6; ++i)
    input.Data<float>()[i] = static_cast<float>(i + 1);
  auto device_side = CopyTo(*opencv, input);
  EXPECT_TRUE(device_side.ok());
  auto output = CopyTo(*native, device_side.value());
  EXPECT_TRUE(output.ok());
  if (output.ok()) {
    EXPECT_EQ(
        std::memcmp(input.data(), output.value().data(), input.ByteSize()), 0);
  }
}

TEST(hpcplatform, OpenCvMemoryStatsReturnToZero) {
  auto context = Context::Create(BackendType::kOpenCV).value();
  {
    auto first =
        Tensor::Empty(*context, Shape{8, 8, 3}, DataType::kFloat32).value();
    auto second = Tensor::Empty(*context, Shape{16}, DataType::kUInt8).value();
    EXPECT_EQ(context->GetMemoryStats().allocation_count,
                   static_cast<size_t>(2));
    EXPECT_TRUE(context->GetMemoryStats().allocated_bytes >=
                     first.ByteSize() + second.ByteSize());
  }
  EXPECT_EQ(context->GetMemoryStats().allocation_count,
                 static_cast<size_t>(0));
  EXPECT_EQ(context->GetMemoryStats().allocated_bytes,
                 static_cast<size_t>(0));
  EXPECT_TRUE(context->GetMemoryStats().peak_bytes > 0);
}

TEST(hpcplatform, OpenCvRejectsUnsupportedModesWithoutFallback) {
  auto context = Context::Create(BackendType::kOpenCV).value();
  auto input =
      Tensor::Empty(*context, Shape{2, 2, 3}, DataType::kFloat32).value();
  auto optimized = Resize(Ops(*context, CpuKernelLevel::kOpenMP), input, 4, 4);
  EXPECT_EQ(optimized.status().code(), StatusCode::kInvalidArgument);
  auto half = Tensor::Empty(*context, Shape{1}, DataType::kFloat16);
  EXPECT_EQ(half.status().code(), StatusCode::kUnsupported);

  auto nchw =
      Tensor::Empty(*context, Shape{1, 3, 2, 2}, DataType::kFloat32).value();
  auto resized = Resize(Ops(*context), nchw, 4, 4, InterpolationMode::kBilinear,
                        ImageLayout::kNCHW);
  EXPECT_EQ(resized.status().code(), StatusCode::kInvalidArgument);
}

TEST(hpcplatform, OpenCvOperatorsMatchNativeCpu) {
  auto native = Context::Create(BackendType::kNativeCpu).value();
  auto opencv = Context::Create(BackendType::kOpenCV).value();
  auto input =
      Tensor::Empty(*native, Shape{4, 5, 3}, DataType::kFloat32).value();
  for (int64_t i = 0; i < input.Numel(); ++i) {
    input.Data<float>()[i] = static_cast<float>(i % 17) * 0.25f;
  }
  auto cv_input = CopyTo(*opencv, input).value();

  auto reference_resize = Resize(Ops(*native), input, 3, 4).value();
  auto actual_resize = Resize(Ops(*opencv), cv_input, 3, 4);
  EXPECT_TRUE(actual_resize.ok());
  if (actual_resize.ok())
    ExpectEquivalent(reference_resize, actual_resize.value(), *native);

  auto reference_blur = GaussianBlur(Ops(*native), input, 3, 1.0).value();
  auto actual_blur = GaussianBlur(Ops(*opencv), cv_input, 3, 1.0);
  EXPECT_TRUE(actual_blur.ok());
  if (actual_blur.ok())
    ExpectEquivalent(reference_blur, actual_blur.value(), *native);
}

}  // namespace

