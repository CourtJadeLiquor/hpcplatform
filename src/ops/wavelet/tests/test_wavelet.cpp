// 测试接口用于验证小波分解/重构的可逆性与边界延拓策略。
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

bool BitExact(const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape() || a.dtype() != b.dtype()) return false;
  return std::memcmp(a.data(), b.data(), a.ByteSize()) == 0;
}

bool BandsBitExact(const WaveletBands& a, const WaveletBands& b) {
  return BitExact(a.ll, b.ll) && BitExact(a.lh, b.lh) && BitExact(a.hl, b.hl) &&
         BitExact(a.hh, b.hh);
}

}  // namespace

MYCV_TEST(WaveletShapesAndBands) {
  auto context = GetCpuContext();
  if (!context) return;
  struct Case {
    ImageLayout layout;
    int64_t n;
    int64_t c;
  };
  const Case cases[] = {{ImageLayout::kNHWC, 1, 3}, {ImageLayout::kNCHW, 2, 3}};
  for (const Case& item : cases) {
    auto input = MakeImage(item.n, 8, 6, item.c, item.layout);
    auto result = WaveletTransform(CpuOpContext(*context), input,
                                   WaveletKind::kHaar, item.layout);
    MYCV_EXPECT_TRUE(result.ok());
    if (!result.ok()) continue;
    const Tensor* bands[4] = {&result.value().ll, &result.value().lh,
                              &result.value().hl, &result.value().hh};
    bool shapes_ok = true;
    for (const Tensor* band : bands) {
      shapes_ok = shapes_ok && band->dtype() == DataType::kFloat32 &&
                  band->backend() == BackendType::kNativeCpu &&
                  band->IsContiguous();
      const Shape expected = item.layout == ImageLayout::kNHWC
                                 ? Shape{item.n, 4, 3, item.c}
                                 : Shape{item.n, item.c, 4, 3};
      shapes_ok = shapes_ok && band->shape() == expected;
    }
    MYCV_EXPECT_TRUE(shapes_ok);
  }
}

MYCV_TEST(WaveletHaarHandComputedTwoByTwo) {
  auto context = GetCpuContext();
  if (!context) return;
  auto input = context->AllocateTensor(Shape{2, 2}, DataType::kFloat32);
  MYCV_EXPECT_TRUE(input.ok());
  if (!input.ok()) return;
  const double values[4] = {1.0, 2.0, 3.0, 4.0};  // [[1, 2], [3, 4]]
  for (int64_t i = 0; i < 4; ++i)
    WriteElement(&input.value(), i, values[i]);

  auto result = WaveletTransform(CpuOpContext(*context), input.value());
  MYCV_EXPECT_TRUE(result.ok());
  if (!result.ok()) return;
  // LL = (a+b+c+d)/2, LH = (a+b-c-d)/2, HL = (a-b+c-d)/2, HH = (a-b-c+d)/2.
  MYCV_EXPECT_TRUE((result.value().ll.shape() == Shape{1, 1}));
  MYCV_EXPECT_TRUE(result.value().ll.Numel() == 1);
  if (result.value().ll.Numel() != 1) return;
  MYCV_EXPECT_TRUE(std::fabs(ReadElement(result.value().ll, 0) - 5.0) <= 1e-6);
  MYCV_EXPECT_TRUE(std::fabs(ReadElement(result.value().lh, 0) + 2.0) <= 1e-6);
  MYCV_EXPECT_TRUE(std::fabs(ReadElement(result.value().hl, 0) + 1.0) <= 1e-6);
  MYCV_EXPECT_TRUE(std::fabs(ReadElement(result.value().hh, 0)) <= 1e-6);
}

MYCV_TEST(WaveletConstantImageHasZeroDetails) {
  auto context = GetCpuContext();
  if (!context) return;
  auto input = MakeConstant(1, 8, 6, 2, 0.25);
  for (WaveletKind kind : {WaveletKind::kHaar, WaveletKind::kDaubechies4}) {
    auto result = WaveletTransform(CpuOpContext(*context), input, kind);
    MYCV_EXPECT_TRUE(result.ok());
    if (!result.ok()) continue;
    // Every orthonormal low-pass here sums to sqrt(2), so a constant survives
    // one level with a gain of 2 and all detail bands vanish.
    const Tensor* details[3] = {&result.value().lh, &result.value().hl,
                                &result.value().hh};
    bool details_zero = true;
    for (const Tensor* band : details) {
      for (int64_t i = 0; i < band->Numel() && details_zero; ++i) {
        details_zero = std::fabs(ReadElement(*band, i)) <= 1e-6;
      }
    }
    MYCV_EXPECT_TRUE(details_zero);
    bool ll_ok = true;
    for (int64_t i = 0; i < result.value().ll.Numel() && ll_ok; ++i) {
      ll_ok = std::fabs(ReadElement(result.value().ll, i) - 0.5) <= 1e-6;
    }
    MYCV_EXPECT_TRUE(ll_ok);
  }
}

MYCV_TEST(WaveletRoundTripIsIdentity) {
  auto context = GetCpuContext();
  if (!context) return;
  struct Case {
    WaveletKind kind;
    DataType dtype;
    double tolerance;
  };
  const Case cases[] = {
      {WaveletKind::kHaar, DataType::kFloat32, 1e-6},
      {WaveletKind::kDaubechies4, DataType::kFloat32, 1e-6},
      {WaveletKind::kHaar, DataType::kFloat64, 1e-12},
      {WaveletKind::kDaubechies4, DataType::kFloat64, 1e-12},
  };
  auto input = MakeImage(1, 16, 12, 2, ImageLayout::kNHWC, DataType::kFloat64);
  for (const Case& item : cases) {
    auto source =
        item.dtype == DataType::kFloat32
            ? MakeImage(1, 16, 12, 2, ImageLayout::kNHWC, DataType::kFloat32)
            : input;
    auto decomposed =
        WaveletTransform(CpuOpContext(*context), source, item.kind);
    MYCV_EXPECT_TRUE(decomposed.ok());
    if (!decomposed.ok()) continue;
    auto rebuilt = InverseWaveletTransform(CpuOpContext(*context),
                                           decomposed.value(), item.kind);
    MYCV_EXPECT_TRUE(rebuilt.ok());
    if (!rebuilt.ok()) continue;
    MYCV_EXPECT_TRUE(Near(rebuilt.value(), source, item.tolerance));
  }
}

MYCV_TEST(WaveletConservesEnergy) {
  auto context = GetCpuContext();
  if (!context) return;
  auto input = MakeImage(1, 8, 8, 1, ImageLayout::kNHWC, DataType::kFloat64);
  double spatial = 0.0;
  for (int64_t i = 0; i < input.Numel(); ++i) {
    const double value = ReadElement(input, i);
    spatial += value * value;
  }
  for (WaveletKind kind : {WaveletKind::kHaar, WaveletKind::kDaubechies4}) {
    auto result = WaveletTransform(CpuOpContext(*context), input, kind);
    MYCV_EXPECT_TRUE(result.ok());
    if (!result.ok()) continue;
    const Tensor* bands[4] = {&result.value().ll, &result.value().lh,
                              &result.value().hl, &result.value().hh};
    double spectral = 0.0;
    for (const Tensor* band : bands) {
      for (int64_t i = 0; i < band->Numel(); ++i) {
        const double value = ReadElement(*band, i);
        spectral += value * value;
      }
    }
    // Orthonormal analysis: the transform preserves the L2 norm exactly.
    MYCV_EXPECT_TRUE(std::fabs(spectral - spatial) <= 1e-9 * spatial);
  }
}

MYCV_TEST(WaveletCpuLevelsAreBitExact) {
  auto context = GetCpuContext();
  if (!context) return;
  auto input = MakeImage(1, 18, 20, 3, ImageLayout::kNHWC);
  const OpContext reference_context = CpuOpContext(*context);
  auto reference =
      WaveletTransform(reference_context, input, WaveletKind::kDaubechies4);
  MYCV_EXPECT_TRUE(reference.ok());
  if (!reference.ok()) return;
  for (CpuKernelLevel level :
       {CpuKernelLevel::kOpenMP, CpuKernelLevel::kSimdGeneric,
        CpuKernelLevel::kSimdAvx2, CpuKernelLevel::kTbb}) {
    if (!context->backend_ptr()->SupportsKernelLevel(level)) continue;
    OpContext level_context = CpuOpContext(*context);
    level_context.cpu_level = level;
    auto actual =
        WaveletTransform(level_context, input, WaveletKind::kDaubechies4);
    // Backend-wide capability is broader than this operator's matrix; an
    // unavailable level is skipped instead of failing the comparison.
    if (!actual.ok() && actual.status().code() == StatusCode::kUnsupported) {
      continue;
    }
    MYCV_EXPECT_TRUE(actual.ok());
    if (!actual.ok()) continue;
    MYCV_EXPECT_TRUE(BandsBitExact(actual.value(), reference.value()));
  }
}

MYCV_TEST(WaveletRejectsInvalidParameters) {
  auto context = GetCpuContext();
  if (!context) return;
  const OpContext op_context = CpuOpContext(*context);
  auto valid = MakeImage(1, 8, 8, 3, ImageLayout::kNHWC);

  // The periodic single-level transform halves both extents, so odd extents
  // have no well-defined band shape.
  auto odd = MakeImage(1, 7, 8, 3, ImageLayout::kNHWC);
  auto odd_result = WaveletTransform(op_context, odd);
  MYCV_EXPECT_EQ(odd_result.status().code(), StatusCode::kInvalidArgument);

  auto u8 = context->AllocateTensor(Shape{1, 8, 8, 3}, DataType::kUInt8);
  MYCV_EXPECT_TRUE(u8.ok());
  if (u8.ok()) {
    // Signed detail bands cannot be stored in u8.
    auto unsupported = WaveletTransform(op_context, u8.value());
    MYCV_EXPECT_EQ(unsupported.status().code(), StatusCode::kUnsupported);
  }
  auto rank5 =
      context->AllocateTensor(Shape{1, 1, 2, 2, 2}, DataType::kFloat32);
  MYCV_EXPECT_TRUE(rank5.ok());
  if (rank5.ok()) {
    auto bad_rank = WaveletTransform(op_context, rank5.value());
    MYCV_EXPECT_EQ(bad_rank.status().code(), StatusCode::kInvalidArgument);
  }
  auto null_context = WaveletTransform(OpContext{}, valid);
  MYCV_EXPECT_EQ(null_context.status().code(), StatusCode::kInvalidArgument);

  auto bands = WaveletTransform(op_context, valid);
  MYCV_EXPECT_TRUE(bands.ok());
  if (!bands.ok()) return;
  WaveletBands missing = bands.value();
  missing.hh = Tensor{};
  auto undefined = InverseWaveletTransform(op_context, missing);
  MYCV_EXPECT_EQ(undefined.status().code(), StatusCode::kInvalidArgument);

  WaveletBands mismatched = bands.value();
  auto other = context->AllocateTensor(Shape{1, 4, 4, 1}, DataType::kFloat32);
  MYCV_EXPECT_TRUE(other.ok());
  if (other.ok()) {
    mismatched.lh = other.value();
    auto bad_shape = InverseWaveletTransform(op_context, mismatched);
    MYCV_EXPECT_EQ(bad_shape.status().code(), StatusCode::kInvalidArgument);
  }
  WaveletBands wrong_dtype = bands.value();
  auto double_band =
      context->AllocateTensor(Shape{1, 4, 4, 3}, DataType::kFloat64);
  MYCV_EXPECT_TRUE(double_band.ok());
  if (double_band.ok()) {
    wrong_dtype.hl = double_band.value();
    auto bad_dtype = InverseWaveletTransform(op_context, wrong_dtype);
    MYCV_EXPECT_EQ(bad_dtype.status().code(), StatusCode::kInvalidArgument);
  }
}

#ifdef MYCV_WITH_OPENCV
MYCV_TEST(WaveletOpenCvIsNotImplemented) {
  auto backend = Context::Create(BackendType::kOpenCV);
  if (!backend.ok()) return;  // backend not usable on this machine
  auto input =
      backend.value()->AllocateTensor(Shape{1, 8, 8, 1}, DataType::kFloat32);
  MYCV_EXPECT_TRUE(input.ok());
  if (!input.ok()) return;
  OpContext context;
  context.context = backend.value().get();
  context.stream = &backend.value()->stream();
  auto result = WaveletTransform(context, input.value());
  MYCV_EXPECT_EQ(result.status().code(), StatusCode::kNotImplemented);
}
#endif

#ifdef MYCV_WITH_CUDA
namespace {

std::shared_ptr<Context> GetCudaContext() {
  static auto result = Context::Create(BackendType::kCUDA);
  static const bool reported = [] {
    if (!result.ok()) {
      std::cout << "SKIP Wavelet CUDA tests: " << result.status().ToString()
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

MYCV_TEST(WaveletCudaMatchesCpu) {
  auto cpu_context = GetCpuContext();
  auto cuda_context = GetCudaContext();
  if (!cpu_context || !cuda_context) return;
  auto input = MakeImage(1, 16, 12, 2, ImageLayout::kNHWC);
  auto device_input = CopyTo(*cuda_context, input);
  MYCV_EXPECT_TRUE(device_input.ok());
  if (!device_input.ok()) return;
  for (WaveletKind kind : {WaveletKind::kHaar, WaveletKind::kDaubechies4}) {
    auto reference = WaveletTransform(CpuOpContext(*cpu_context), input, kind);
    MYCV_EXPECT_TRUE(reference.ok());
    if (!reference.ok()) continue;
    auto actual = WaveletTransform(CudaOpContext(*cuda_context),
                                   device_input.value(), kind);
    MYCV_EXPECT_TRUE(actual.ok());
    if (!actual.ok()) continue;
    const Tensor* reference_bands[4] = {
        &reference.value().ll, &reference.value().lh, &reference.value().hl,
        &reference.value().hh};
    const Tensor* actual_bands[4] = {&actual.value().ll, &actual.value().lh,
                                     &actual.value().hl, &actual.value().hh};
    for (int i = 0; i < 4; ++i) {
      auto back = CopyTo(*cpu_context, *actual_bands[i]);
      MYCV_EXPECT_TRUE(back.ok());
      if (!back.ok()) continue;
      MYCV_EXPECT_TRUE(Near(back.value(), *reference_bands[i], 1e-4));
    }
  }
}

MYCV_TEST(WaveletCudaRoundTripIsIdentity) {
  auto cpu_context = GetCpuContext();
  auto cuda_context = GetCudaContext();
  if (!cpu_context || !cuda_context) return;
  auto input = MakeImage(1, 16, 12, 2, ImageLayout::kNHWC);
  auto device_input = CopyTo(*cuda_context, input);
  MYCV_EXPECT_TRUE(device_input.ok());
  if (!device_input.ok()) return;
  for (WaveletKind kind : {WaveletKind::kHaar, WaveletKind::kDaubechies4}) {
    const OpContext cuda_op = CudaOpContext(*cuda_context);
    auto decomposed = WaveletTransform(cuda_op, device_input.value(), kind);
    MYCV_EXPECT_TRUE(decomposed.ok());
    if (!decomposed.ok()) continue;
    auto rebuilt = InverseWaveletTransform(cuda_op, decomposed.value(), kind);
    MYCV_EXPECT_TRUE(rebuilt.ok());
    if (!rebuilt.ok()) continue;
    auto back = CopyTo(*cpu_context, rebuilt.value());
    MYCV_EXPECT_TRUE(back.ok());
    if (!back.ok()) continue;
    MYCV_EXPECT_TRUE(Near(back.value(), input, 1e-4));
  }
}
#endif

MYCV_TEST_MAIN()
