// 测试接口用于验证 fourier 正逆变换的一致性、归一化与边界处理。
#include "test_framework.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "core/backend.h"
#include "mycv/mycv.h"

namespace {

using namespace mycv;

OpContext CpuOpContext(Context& context) {
  OpContext result;
  result.context = &context;
  result.stream = &context.stream();
  return result;
}

std::shared_ptr<Context> GetCpuContext() {
  static auto context = Context::Create(BackendType::kNativeCpu);
  return context.ok() ? context.value() : nullptr;
}

double ReadElement(const Tensor& tensor, int64_t index) {
  if (tensor.dtype() == DataType::kFloat32) {
    return tensor.Data<float>()[index];
  }
  return tensor.Data<double>()[index];
}

void WriteElement(Tensor* tensor, int64_t index, double value) {
  if (tensor->dtype() == DataType::kFloat32) {
    tensor->Data<float>()[index] = static_cast<float>(value);
    return;
  }
  tensor->Data<double>()[index] = value;
}

// Fills by logical (n, y, x, c) coordinates, independent of the memory order,
// so the same logical image can be materialized as NHWC or NCHW.
Tensor MakeImage(int64_t n, int64_t h, int64_t w, int64_t c, ImageLayout layout,
                 DataType dtype = DataType::kFloat32) {
  auto context = GetCpuContext();
  if (!context) return Tensor{};
  const Shape shape =
      layout == ImageLayout::kNHWC ? Shape{n, h, w, c} : Shape{n, c, h, w};
  auto tensor = context->AllocateTensor(shape, dtype);
  MYCV_EXPECT_TRUE(tensor.ok());
  if (!tensor.ok()) return Tensor{};
  for (int64_t ni = 0; ni < n; ++ni) {
    for (int64_t y = 0; y < h; ++y) {
      for (int64_t x = 0; x < w; ++x) {
        for (int64_t ci = 0; ci < c; ++ci) {
          const double value =
              0.01 *
              static_cast<double>((y * 31 + x * 17 + ci * 7 + ni * 5) % 97);
          const int64_t index = layout == ImageLayout::kNHWC
                                    ? ((ni * h + y) * w + x) * c + ci
                                    : ((ni * c + ci) * h + y) * w + x;
          WriteElement(&tensor.value(), index, value);
        }
      }
    }
  }
  return tensor.value();
}

Tensor MakeConstant(int64_t n, int64_t h, int64_t w, int64_t c, double value) {
  auto context = GetCpuContext();
  if (!context) return Tensor{};
  auto tensor = context->AllocateTensor(Shape{n, h, w, c}, DataType::kFloat32);
  MYCV_EXPECT_TRUE(tensor.ok());
  if (!tensor.ok()) return Tensor{};
  for (int64_t i = 0; i < tensor.value().Numel(); ++i) {
    WriteElement(&tensor.value(), i, value);
  }
  return tensor.value();
}

bool Near(const Tensor& actual, const Tensor& expected, double tolerance) {
  if (actual.shape() != expected.shape() ||
      actual.dtype() != expected.dtype()) {
    return false;
  }
  for (int64_t i = 0; i < actual.Numel(); ++i) {
    const double a = ReadElement(actual, i);
    const double e = ReadElement(expected, i);
    if (std::fabs(a - e) > tolerance * std::fmax(1.0, std::fabs(e))) {
      return false;
    }
  }
  return true;
}

bool NearValues(const Tensor& actual, const std::vector<double>& expected,
                double tolerance) {
  if (actual.Numel() != static_cast<int64_t>(expected.size())) return false;
  for (int64_t i = 0; i < actual.Numel(); ++i) {
    const double e = expected[static_cast<size_t>(i)];
    if (std::fabs(ReadElement(actual, i) - e) >
        tolerance * std::fmax(1.0, std::fabs(e))) {
      return false;
    }
  }
  return true;
}

bool BitExact(const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape() || a.dtype() != b.dtype()) return false;
  return std::memcmp(a.data(), b.data(), a.ByteSize()) == 0;
}

// Same logical element read from NHWC and NCHW tensors of the same image.
double LogicalElement(const Tensor& tensor, ImageLayout layout, int64_t n,
                      int64_t y, int64_t x, int64_t c, int64_t h, int64_t w,
                      int64_t channels) {
  const int64_t index = layout == ImageLayout::kNHWC
                            ? ((n * h + y) * w + x) * channels + c
                            : ((n * channels + c) * h + y) * w + x;
  return ReadElement(tensor, index);
}

}  // namespace

MYCV_TEST(FourierHandComputedFourPoint) {
  auto context = GetCpuContext();
  if (!context) return;
  auto input = context->AllocateTensor(Shape{1, 4}, DataType::kFloat32);
  MYCV_EXPECT_TRUE(input.ok());
  if (!input.ok()) return;
  const double values[4] = {1.0, 2.0, 3.0, 4.0};
  for (int64_t i = 0; i < 4; ++i)
    WriteElement(&input.value(), i, values[i]);

  auto result = FourierTransform(CpuOpContext(*context), input.value());
  MYCV_EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  MYCV_EXPECT_TRUE(
      NearValues(result.value().real, {10.0, -2.0, -2.0, -2.0}, 1e-6));
  MYCV_EXPECT_TRUE(
      NearValues(result.value().imag, {0.0, 2.0, 0.0, -2.0}, 1e-6));
}

MYCV_TEST(FourierConstantImageIsImpulse) {
  auto context = GetCpuContext();
  if (!context) return;
  // A constant 0.25 image has all of its energy at DC: H*W*0.25 = 16.
  auto input = MakeConstant(1, 8, 8, 1, 0.25);
  auto result = FourierTransform(CpuOpContext(*context), input);
  MYCV_EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  bool ok = true;
  for (int64_t i = 0; i < result.value().real.Numel() && ok; ++i) {
    const double expected = i == 0 ? 16.0 : 0.0;
    ok = std::fabs(ReadElement(result.value().real, i) - expected) <= 1e-5 &&
         std::fabs(ReadElement(result.value().imag, i)) <= 1e-6;
  }
  MYCV_EXPECT_TRUE(ok);
}

MYCV_TEST(FourierImpulseIsFlatSpectrum) {
  auto context = GetCpuContext();
  if (!context) return;
  auto input = MakeConstant(1, 8, 8, 1, 0.0);
  WriteElement(&input, 0, 1.0);  // impulse at (0, 0)
  auto result = FourierTransform(CpuOpContext(*context), input);
  MYCV_EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  bool ok = true;
  for (int64_t i = 0; i < result.value().real.Numel() && ok; ++i) {
    ok = std::fabs(ReadElement(result.value().real, i) - 1.0) <= 1e-6 &&
         std::fabs(ReadElement(result.value().imag, i)) <= 1e-9;
  }
  MYCV_EXPECT_TRUE(ok);
}

MYCV_TEST(FourierSatisfiesParseval) {
  auto context = GetCpuContext();
  if (!context) return;
  // Non-power-of-two extents exercise the DFT fallback.
  auto input = MakeImage(1, 7, 5, 1, ImageLayout::kNHWC, DataType::kFloat64);
  auto result = FourierTransform(CpuOpContext(*context), input);
  MYCV_EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  double spatial = 0.0;
  double spectral = 0.0;
  for (int64_t i = 0; i < input.Numel(); ++i) {
    const double value = ReadElement(input, i);
    spatial += value * value;
    const double re = ReadElement(result.value().real, i);
    const double im = ReadElement(result.value().imag, i);
    spectral += re * re + im * im;
  }
  // Sum |F|^2 = H*W * sum |x|^2 for an unnormalized forward 2-D DFT.
  const double expected = static_cast<double>(input.Numel()) * spatial;
  MYCV_EXPECT_TRUE(std::fabs(spectral - expected) <= 1e-9 * expected);
}

MYCV_TEST(FourierRoundTripIsIdentity) {
  auto context = GetCpuContext();
  if (!context) return;
  struct Case {
    int64_t h;
    int64_t w;
    int64_t c;
    ImageLayout layout;
    DataType dtype;
    double tolerance;
  };
  const Case cases[] = {
      {16, 12, 2, ImageLayout::kNHWC, DataType::kFloat32, 1e-5},
      {9, 7, 1, ImageLayout::kNHWC, DataType::kFloat64, 1e-9},
      {8, 10, 2, ImageLayout::kNCHW, DataType::kFloat32, 1e-5},
  };
  for (const Case& item : cases) {
    auto input = MakeImage(1, item.h, item.w, item.c, item.layout, item.dtype);
    const OpContext op_context = CpuOpContext(*context);
    auto forward = FourierTransform(op_context, input, item.layout);
    MYCV_EXPECT_TRUE(forward.ok());
    if (!forward.ok()) continue;
    auto inverse = InverseFourierTransform(op_context, forward.value().real,
                                           forward.value().imag, item.layout);
    MYCV_EXPECT_TRUE(inverse.ok());
    if (!inverse.ok()) continue;
    MYCV_EXPECT_TRUE(Near(inverse.value(), input, item.tolerance));
  }
}

MYCV_TEST(FourierLayoutsAgree) {
  auto context = GetCpuContext();
  if (!context) return;
  const int64_t h = 8;
  const int64_t w = 6;
  const int64_t c = 3;
  auto nhwc = MakeImage(1, h, w, c, ImageLayout::kNHWC);
  auto nchw = MakeImage(1, h, w, c, ImageLayout::kNCHW);
  const OpContext op_context = CpuOpContext(*context);
  auto from_nhwc = FourierTransform(op_context, nhwc, ImageLayout::kNHWC);
  auto from_nchw = FourierTransform(op_context, nchw, ImageLayout::kNCHW);
  MYCV_EXPECT_TRUE(from_nhwc.ok());
  MYCV_EXPECT_TRUE(from_nchw.ok());
  if (!from_nhwc.ok() || !from_nchw.ok()) return;
  bool equal = true;
  for (int64_t y = 0; y < h && equal; ++y) {
    for (int64_t x = 0; x < w && equal; ++x) {
      for (int64_t ci = 0; ci < c && equal; ++ci) {
        const double re_a = LogicalElement(
            from_nhwc.value().real, ImageLayout::kNHWC, 0, y, x, ci, h, w, c);
        const double re_b = LogicalElement(
            from_nchw.value().real, ImageLayout::kNCHW, 0, y, x, ci, h, w, c);
        const double im_a = LogicalElement(
            from_nhwc.value().imag, ImageLayout::kNHWC, 0, y, x, ci, h, w, c);
        const double im_b = LogicalElement(
            from_nchw.value().imag, ImageLayout::kNCHW, 0, y, x, ci, h, w, c);
        equal = re_a == re_b && im_a == im_b;
      }
    }
  }
  MYCV_EXPECT_TRUE(equal);
}

MYCV_TEST(FourierCpuLevelsAreBitExact) {
  auto context = GetCpuContext();
  if (!context) return;
  auto input = MakeImage(1, 17, 19, 3, ImageLayout::kNHWC);
  const OpContext reference_context = CpuOpContext(*context);
  auto reference = FourierTransform(reference_context, input);
  MYCV_EXPECT_TRUE(reference.ok());
  if (!reference.ok()) return;
  for (CpuKernelLevel level :
       {CpuKernelLevel::kOpenMP, CpuKernelLevel::kSimdGeneric,
        CpuKernelLevel::kSimdAvx2, CpuKernelLevel::kTbb}) {
    if (!context->backend_ptr()->SupportsKernelLevel(level)) continue;
    OpContext level_context = CpuOpContext(*context);
    level_context.cpu_level = level;
    auto actual = FourierTransform(level_context, input);
    // Backend-wide capability is broader than this operator's matrix; an
    // unavailable level is skipped instead of failing the comparison.
    if (!actual.ok() && actual.status().code() == StatusCode::kUnsupported) {
      continue;
    }
    MYCV_EXPECT_TRUE(actual.ok());
    if (!actual.ok()) continue;
    MYCV_EXPECT_TRUE(BitExact(actual.value().real, reference.value().real));
    MYCV_EXPECT_TRUE(BitExact(actual.value().imag, reference.value().imag));
  }
}

MYCV_TEST(FourierRejectsInvalidParameters) {
  auto context = GetCpuContext();
  if (!context) return;
  const OpContext op_context = CpuOpContext(*context);
  auto valid = MakeImage(1, 8, 8, 3, ImageLayout::kNHWC);

  auto u8 = context->AllocateTensor(Shape{1, 8, 8, 3}, DataType::kUInt8);
  MYCV_EXPECT_TRUE(u8.ok());
  if (u8.ok()) {
    auto unsupported = FourierTransform(op_context, u8.value());
    MYCV_EXPECT_EQ(unsupported.status().code(), StatusCode::kUnsupported);
  }
  auto rank5 =
      context->AllocateTensor(Shape{1, 1, 2, 2, 2}, DataType::kFloat32);
  MYCV_EXPECT_TRUE(rank5.ok());
  if (rank5.ok()) {
    auto bad_rank = FourierTransform(op_context, rank5.value());
    MYCV_EXPECT_EQ(bad_rank.status().code(), StatusCode::kInvalidArgument);
  }
  auto null_context = FourierTransform(OpContext{}, valid);
  MYCV_EXPECT_EQ(null_context.status().code(), StatusCode::kInvalidArgument);

  // Inverse: the imaginary part must describe the same image.
  auto forward = FourierTransform(op_context, valid);
  MYCV_EXPECT_TRUE(forward.ok());
  if (forward.ok()) {
    auto mismatch_shape =
        context->AllocateTensor(Shape{1, 8, 8, 1}, DataType::kFloat32);
    MYCV_EXPECT_TRUE(mismatch_shape.ok());
    if (mismatch_shape.ok()) {
      auto bad = InverseFourierTransform(op_context, forward.value().real,
                                         mismatch_shape.value());
      MYCV_EXPECT_EQ(bad.status().code(), StatusCode::kInvalidArgument);
    }
    auto double_imag =
        context->AllocateTensor(Shape{1, 8, 8, 3}, DataType::kFloat64);
    MYCV_EXPECT_TRUE(double_imag.ok());
    if (double_imag.ok()) {
      auto bad_dtype = InverseFourierTransform(op_context, forward.value().real,
                                               double_imag.value());
      MYCV_EXPECT_EQ(bad_dtype.status().code(), StatusCode::kInvalidArgument);
    }
    auto undefined =
        InverseFourierTransform(op_context, forward.value().real, Tensor{});
    MYCV_EXPECT_EQ(undefined.status().code(), StatusCode::kInvalidArgument);
  }
}

#ifdef MYCV_WITH_OPENCV
MYCV_TEST(FourierOpenCvIsNotImplemented) {
  auto backend = Context::Create(BackendType::kOpenCV);
  if (!backend.ok()) return;  // backend not usable on this machine
  auto input =
      backend.value()->AllocateTensor(Shape{1, 8, 8, 1}, DataType::kFloat32);
  MYCV_EXPECT_TRUE(input.ok());
  if (!input.ok()) return;
  OpContext context;
  context.context = backend.value().get();
  context.stream = &backend.value()->stream();
  auto result = FourierTransform(context, input.value());
  MYCV_EXPECT_EQ(result.status().code(), StatusCode::kNotImplemented);
}
#endif

#ifdef MYCV_WITH_CUDA
namespace {

std::shared_ptr<Context> GetCudaContext() {
  static auto result = Context::Create(BackendType::kCUDA);
  static const bool reported = [] {
    if (!result.ok()) {
      std::cout << "SKIP Fourier CUDA tests: " << result.status().ToString()
                << '\n';
    }
    return true;
  }();
  (void)reported;
  return result.ok() ? result.value() : nullptr;
}

Result<Tensor> CopyTo(Context& destination, const Tensor& source) {
  auto output = destination.AllocateTensor(source.shape(), source.dtype());
  if (!output.ok()) return output.status();
  MYCV_RETURN_IF_ERROR(destination.Copy(&output.value(), source));
  return output;
}

OpContext CudaOpContext(Context& context) {
  OpContext result;
  result.context = &context;
  result.stream = &context.stream();
  return result;
}

}  // namespace

// Device radix-2 path (16x12) and the out-of-place DFT fallback (9x7) are both
// compared against the CPU reference.
MYCV_TEST(FourierCudaMatchesCpu) {
  auto cpu_context = GetCpuContext();
  auto cuda_context = GetCudaContext();
  if (!cpu_context || !cuda_context) return;
  struct Case {
    int64_t h;
    int64_t w;
    int64_t c;
    ImageLayout layout;
    DataType dtype;
  };
  const Case cases[] = {
      {16, 12, 2, ImageLayout::kNHWC, DataType::kFloat32},
      {9, 7, 1, ImageLayout::kNHWC, DataType::kFloat64},
      {8, 10, 2, ImageLayout::kNCHW, DataType::kFloat32},
  };
  for (const Case& item : cases) {
    auto input = MakeImage(1, item.h, item.w, item.c, item.layout, item.dtype);
    auto reference =
        FourierTransform(CpuOpContext(*cpu_context), input, item.layout);
    MYCV_EXPECT_TRUE(reference.ok());
    if (!reference.ok()) continue;
    auto device_input = CopyTo(*cuda_context, input);
    MYCV_EXPECT_TRUE(device_input.ok());
    if (!device_input.ok()) continue;
    auto actual = FourierTransform(CudaOpContext(*cuda_context),
                                   device_input.value(), item.layout);
    MYCV_EXPECT_TRUE(actual.ok());
    if (!actual.ok()) continue;
    auto real = CopyTo(*cpu_context, actual.value().real);
    auto imag = CopyTo(*cpu_context, actual.value().imag);
    MYCV_EXPECT_TRUE(real.ok());
    MYCV_EXPECT_TRUE(imag.ok());
    if (!real.ok() || !imag.ok()) continue;
    MYCV_EXPECT_TRUE(Near(real.value(), reference.value().real, 1e-4));
    MYCV_EXPECT_TRUE(Near(imag.value(), reference.value().imag, 1e-4));
  }
}

MYCV_TEST(FourierCudaRoundTripIsIdentity) {
  auto cpu_context = GetCpuContext();
  auto cuda_context = GetCudaContext();
  if (!cpu_context || !cuda_context) return;
  auto input = MakeImage(1, 16, 12, 2, ImageLayout::kNHWC, DataType::kFloat32);
  auto device_input = CopyTo(*cuda_context, input);
  MYCV_EXPECT_TRUE(device_input.ok());
  if (!device_input.ok()) return;
  const OpContext cuda_op = CudaOpContext(*cuda_context);
  auto forward = FourierTransform(cuda_op, device_input.value());
  MYCV_EXPECT_TRUE(forward.ok());
  if (!forward.ok()) return;
  auto inverse = InverseFourierTransform(cuda_op, forward.value().real,
                                         forward.value().imag);
  MYCV_EXPECT_TRUE(inverse.ok());
  if (!inverse.ok()) return;
  auto back = CopyTo(*cpu_context, inverse.value());
  MYCV_EXPECT_TRUE(back.ok());
  if (!back.ok()) return;
  MYCV_EXPECT_TRUE(Near(back.value(), input, 1e-4));
}
#endif

MYCV_TEST_MAIN()
