// Per-operator benchmark for resize (docs/architecture.md §6.1).
// Measures resize throughput and latency for every backend/kernel level available in this build.
//
// Sweeps interpolation mode, scale direction, dtype and every execution
// backend available in this build, and emits CSV for direct recording in
// src/ops/resize/README.md:
//
//   ./build/src/ops/resize/resize_benchmark
//
// Columns:
//   op,backend,level,dtype,mode,in_h,in_w,c,out_h,out_w,ms_per_call,gb_per_s,speedup
// speedup is relative to the naive CPU row of the same (dtype, mode, shape).

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "core/backend.h"
#include "hpcplatform/hpcplatform.h"

using namespace hpcplatform;

namespace {

constexpr int kIterations = 30;
constexpr int kWarmup = 3;

struct Config {
  const char* mode;
  InterpolationMode interpolation;
  int64_t h;
  int64_t w;
  int64_t c;
  int64_t out_h;
  int64_t out_w;
};

struct Measurement {
  double ms_per_call = 0.0;
  double gb_per_second = 0.0;
};

const char* DTypeLabel(DataType dtype) {
  switch (dtype) {
    case DataType::kUInt8:
      return "u8";
    case DataType::kFloat32:
      return "f32";
    default:
      return "f64";
  }
}

void FillInput(Tensor* tensor) {
  const int64_t count = tensor->Numel();
  switch (tensor->dtype()) {
    case DataType::kUInt8:
      for (int64_t i = 0; i < count; ++i) {
        tensor->Data<uint8_t>()[i] = static_cast<uint8_t>((i * 31) & 0xFF);
      }
      break;
    case DataType::kFloat32:
      for (int64_t i = 0; i < count; ++i) {
        tensor->Data<float>()[i] = static_cast<float>((i * 31) & 0xFF) / 255.0f;
      }
      break;
    default:
      for (int64_t i = 0; i < count; ++i) {
        tensor->Data<double>()[i] =
            static_cast<double>((i * 31) & 0xFF) / 255.0;
      }
      break;
  }
}

Measurement Measure(const OpContext& context, const Tensor& input,
                    const Config& config, bool* ok) {
  *ok = false;
  for (int i = 0; i < kWarmup; ++i) {
    auto warmup = Resize(context, input, config.out_h, config.out_w,
                         config.interpolation);
    if (!warmup.ok()) return {};
  }
  const auto start = std::chrono::steady_clock::now();
  double checksum = 0.0;
  for (int i = 0; i < kIterations; ++i) {
    auto output = Resize(context, input, config.out_h, config.out_w,
                         config.interpolation);
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
  const double bytes =
      element_size *
      static_cast<double>(input.Numel() +
                          config.out_h * config.out_w * config.c);
  if (checksum == -1.0) std::cerr << "";
  return {ms, bytes / (ms * 1e6)};
}

void PrintRow(const char* backend, const char* level, DataType dtype,
              const Config& config, const Measurement& run, double baseline) {
  std::cout << "resize," << backend << ',' << level << ',' << DTypeLabel(dtype)
            << ',' << config.mode << ',' << config.h << ',' << config.w << ','
            << config.c << ',' << config.out_h << ',' << config.out_w << ','
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
      {"bilinear_down", InterpolationMode::kBilinear, 1024, 1024, 3, 512, 768},
      {"bilinear_up", InterpolationMode::kBilinear, 512, 768, 3, 1024, 1024},
      {"nearest_down", InterpolationMode::kNearest, 1024, 1024, 3, 512, 768},
  };

  std::cout << "op,backend,level,dtype,mode,in_h,in_w,c,out_h,out_w,"
               "ms_per_call,gb_per_s,speedup\n";

  auto opencv = Context::Create(BackendType::kOpenCV);
  auto cuda = Context::Create(BackendType::kCUDA);

  for (const Config& config : configs) {
    for (DataType dtype :
         {DataType::kUInt8, DataType::kFloat32, DataType::kFloat64}) {
      auto input = cpu.value()->AllocateTensor(
          Shape{config.h, config.w, config.c}, dtype);
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

      // Accelerated backends take only kNaive; cover the dtypes they expose.
      if (dtype != DataType::kFloat64) {
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
  }
  return 0;
}
