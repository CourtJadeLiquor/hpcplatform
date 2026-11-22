// Per-operator benchmark for gaussian_blur (docs/architecture.md §6.1/§7.2).
// Measures gaussian_blur throughput and latency for every backend/kernel level available in this build.
//
// Sweeps kernel size / sigma / channel count (c>=4 unlocks the AVX2 block),
// dtype and every execution backend available in this build, emitting CSV for
// src/ops/gaussian_blur/README.md:
//
//   ./build/src/ops/gaussian_blur/gaussian_blur_benchmark
//
// Columns:
//   op,backend,level,dtype,kernel,sigma,in_h,in_w,c,ms_per_call,gb_per_s,speedup
// speedup is relative to the naive CPU row of the same (dtype, kernel, shape).

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>

#include "core/backend.h"
#include "mycv/mycv.h"

using namespace mycv;

namespace {

constexpr int kIterations = 30;
constexpr int kWarmup = 3;
constexpr int64_t kHeight = 512;
constexpr int64_t kWidth = 512;

struct Config {
  int kernel_size;
  double sigma;
  int64_t channels;
};

struct Measurement {
  double ms_per_call = 0.0;
  double gb_per_second = 0.0;
};

const char* DTypeLabel(DataType dtype) {
  return dtype == DataType::kUInt8 ? "u8" : "f32";
}

void FillInput(Tensor* tensor) {
  const int64_t count = tensor->Numel();
  if (tensor->dtype() == DataType::kUInt8) {
    for (int64_t i = 0; i < count; ++i) {
      tensor->Data<uint8_t>()[i] = static_cast<uint8_t>((i * 47) & 0xFF);
    }
  } else {
    for (int64_t i = 0; i < count; ++i) {
      tensor->Data<float>()[i] = static_cast<float>((i * 47) & 0xFF) / 255.0f;
    }
  }
}

Measurement Measure(const OpContext& context, const Tensor& input,
                    const Config& config, bool* ok) {
  *ok = false;
  for (int i = 0; i < kWarmup; ++i) {
    auto warmup =
        GaussianBlur(context, input, config.kernel_size, config.sigma);
    if (!warmup.ok()) return {};
  }
  const auto start = std::chrono::steady_clock::now();
  double checksum = 0.0;
  for (int i = 0; i < kIterations; ++i) {
    auto output =
        GaussianBlur(context, input, config.kernel_size, config.sigma);
    if (!output.ok()) return {};
    // Consume metadata only: accelerator outputs live in device memory and
    // must not be dereferenced from the host.
    checksum += static_cast<double>(output.value().Numel());
  }
  const auto stop = std::chrono::steady_clock::now();
  *ok = true;
  const double ms =
      std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
          stop - start)
          .count() /
      kIterations;
  const double element_size = static_cast<double>(DTypeSize(input.dtype()));
  // in + out (+ the f64 horizontal-pass scratch buffer used by the CPU path).
  const double bytes = static_cast<double>(input.Numel()) *
                       (2.0 * element_size + sizeof(double));
  if (checksum == -1.0) std::cerr << "";
  return {ms, bytes / (ms * 1e6)};
}

void PrintRow(const char* backend, const char* level, DataType dtype,
              const Config& config, const Measurement& run, double baseline) {
  std::cout << "gaussian_blur," << backend << ',' << level << ','
            << DTypeLabel(dtype) << ',' << config.kernel_size << ','
            << std::fixed << std::setprecision(2) << config.sigma << ','
            << kHeight << ',' << kWidth << ',' << config.channels << ','
            << std::fixed << std::setprecision(4) << run.ms_per_call << ','
            << run.gb_per_second << ','
            << (run.ms_per_call > 0.0 ? baseline / run.ms_per_call : 0.0)
            << '\n';
}

}  // namespace

int main() {
  static auto cpu = Context::Create(BackendType::kNativeCpu);
  if (!cpu.ok()) {
    std::cerr << "FAIL: " << cpu.status().ToString() << '\n';
    return 1;
  }

  const Config configs[] = {
      {3, 1.0, 3},  // typical denoise, AVX2 block not applicable (c < 4)
      {7, 2.0, 3},  // wider kernel: tap cost dominates
      {7, 2.0, 4},  // c = 4 enables the AVX2 4-channel block
  };

  std::cout << "op,backend,level,dtype,kernel,sigma,in_h,in_w,c,"
               "ms_per_call,gb_per_s,speedup\n";

  auto opencv = Context::Create(BackendType::kOpenCV);
  auto cuda = Context::Create(BackendType::kCUDA);

  for (const Config& config : configs) {
    for (DataType dtype : {DataType::kUInt8, DataType::kFloat32}) {
      auto input = cpu.value()->AllocateTensor(
          Shape{kHeight, kWidth, config.channels}, dtype);
      if (!input.ok()) continue;
      FillInput(&input.value());

      OpContext op_context;
      op_context.context = cpu.value().get();
      op_context.stream = &cpu.value()->stream();

      double baseline = 0.0;
      for (CpuKernelLevel level :
           {CpuKernelLevel::kNaive, CpuKernelLevel::kOpenMP,
            CpuKernelLevel::kSimdGeneric, CpuKernelLevel::kSimdAvx2,
            CpuKernelLevel::kTbb}) {
        if (!cpu.value()->backend_ptr()->SupportsKernelLevel(level)) continue;
        OpContext level_context = op_context;
        level_context.cpu_level = level;
        bool ok = false;
        const Measurement run =
            Measure(level_context, input.value(), config, &ok);
        if (!ok) continue;
        if (baseline == 0.0) baseline = run.ms_per_call;
        PrintRow("cpu", CpuKernelLevelName(level), dtype, config, run,
                 baseline);
      }

      if (opencv.ok()) {
        OpContext accelerated;
        accelerated.context = opencv.value().get();
        accelerated.stream = &opencv.value()->stream();
        auto device =
            opencv.value()->AllocateTensor(input.value().shape(), dtype);
        if (device.ok() &&
            opencv.value()->Copy(&device.value(), input.value()).ok()) {
          bool ok = false;
          const Measurement run =
              Measure(accelerated, device.value(), config, &ok);
          if (ok) PrintRow("opencv", "naive", dtype, config, run, baseline);
        }
      }
      if (cuda.ok()) {
        OpContext accelerated;
        accelerated.context = cuda.value().get();
        accelerated.stream = &cuda.value()->stream();
        auto device =
            cuda.value()->AllocateTensor(input.value().shape(), dtype);
        if (device.ok() &&
            cuda.value()->Copy(&device.value(), input.value()).ok()) {
          bool ok = false;
          const Measurement run =
              Measure(accelerated, device.value(), config, &ok);
          if (ok) PrintRow("cuda", "naive", dtype, config, run, baseline);
        }
      }
    }
  }
  return 0;
}
