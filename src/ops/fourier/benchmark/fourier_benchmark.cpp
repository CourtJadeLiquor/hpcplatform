// Per-operator benchmark for fourier (docs/architecture.md §6.1/§7.2).
// 基准程序用于测量 fourier 正逆变换在不同尺寸与后端上的耗时。
//
//   ./build-release-full/src/ops/fourier/fourier_benchmark
//
// Columns:
//   op,backend,level,dtype,direction,in_h,in_w,c,ms_per_call,mpix_per_s,speedup
// speedup is relative to the naive CPU row of the same (size, channels,
// direction) configuration.
//
// Measurement order follows the other operator benchmarks: accelerator rows
// (CUDA) are taken first, in a separate phase, so seconds of saturated CPU work
// cannot drop the GPU clock before the device rows run; CPU levels then run
// level-major so each level sees the same machine state. CUDA is created
// lazily: an unavailable backend simply produces no "cuda" rows.

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/backend.h"
#include "mycv/mycv.h"

using namespace mycv;

namespace {

constexpr int kIterations = 5;
constexpr int kWarmup = 1;

struct Config {
  int64_t height;
  int64_t width;
  int64_t channels;
};

struct Entry {
  std::string backend;
  std::string level;
  bool has_forward = false;
  double forward_ms = 0.0;
  bool has_inverse = false;
  double inverse_ms = 0.0;
};

struct ConfigResult {
  Config config;
  bool has_cuda = false;
  Entry cuda;
  std::vector<Entry> cpu;
};

void FillInput(Tensor* tensor) {
  float* data = tensor->Data<float>();
  for (int64_t i = 0; i < tensor->Numel(); ++i) {
    data[i] = static_cast<float>((i * 37) % 251) / 251.0f;
  }
}

template <typename Call>
bool MeasureMs(const Call& call, int passes, double* best_ms) {
  *best_ms = 1e300;
  for (int pass = 0; pass < passes; ++pass) {
    for (int i = 0; i < kWarmup; ++i) {
      if (!call()) return false;
    }
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < kIterations; ++i) {
      if (!call()) return false;
    }
    const auto stop = std::chrono::steady_clock::now();
    const double ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            stop - start)
            .count() /
        kIterations;
    if (ms < *best_ms) *best_ms = ms;
  }
  return true;
}

// Measures forward and inverse of one backend; returns false when the operator
// is unavailable for that backend.
bool MeasureEntry(const OpContext& context, const Tensor& input, int passes,
                  Entry* entry) {
  auto spectrum = FourierTransform(context, input);
  if (!spectrum.ok()) return false;
  double ms = 0.0;
  if (MeasureMs([&] { return FourierTransform(context, input).ok(); }, passes,
                &ms)) {
    entry->has_forward = true;
    entry->forward_ms = ms;
  }
  if (MeasureMs(
          [&] {
            return InverseFourierTransform(context, spectrum.value().real,
                                           spectrum.value().imag)
                .ok();
          },
          passes, &ms)) {
    entry->has_inverse = true;
    entry->inverse_ms = ms;
  }
  return true;
}

void PrintRow(const Config& config, const std::string& backend,
              const std::string& level, const char* direction, double ms,
              double baseline) {
  const double megapixels =
      static_cast<double>(config.height * config.width) / 1e6;
  std::cout << "fourier_transform," << backend << ',' << level << ",f32,"
            << direction << ',' << config.height << ',' << config.width << ','
            << config.channels << ',' << std::fixed << std::setprecision(4)
            << ms << ',' << (ms > 0.0 ? megapixels / (ms / 1e3) : 0.0) << ','
            << (baseline > 0.0 ? baseline / ms : 0.0) << '\n';
}

}  // namespace

int main() {
  auto cpu = Context::Create(BackendType::kNativeCpu);
  if (!cpu.ok()) {
    std::cerr << "FAIL: " << cpu.status().ToString() << '\n';
    return 1;
  }

  const Config configs[] = {
      {256, 256, 1}, {256, 256, 3},   {512, 512, 1},
      {512, 512, 3}, {1024, 1024, 1}, {1024, 1024, 3},
  };
  std::vector<ConfigResult> results;
  results.reserve(sizeof(configs) / sizeof(configs[0]));
  for (const Config& config : configs) {
    ConfigResult result;
    result.config = config;
    results.push_back(std::move(result));
  }

  std::cout << "op,backend,level,dtype,direction,in_h,in_w,c,ms_per_call,"
               "mpix_per_s,speedup\n";

  auto cuda = Context::Create(BackendType::kCUDA);
  if (!cuda.ok()) {
    std::cerr << "SKIP CUDA rows: " << cuda.status().ToString() << '\n';
  }

  // Phase 1: device rows first, while the GPU clock is untouched. Two passes
  // let the first batch absorb any clock ramp.
  for (ConfigResult& result : results) {
    if (!cuda.ok()) break;
    const Config& config = result.config;
    auto input = cpu.value()->AllocateTensor(
        Shape{config.height, config.width, config.channels},
        DataType::kFloat32);
    if (!input.ok()) continue;
    FillInput(&input.value());
    auto device_input = cuda.value()->AllocateTensor(input.value().shape(),
                                                     input.value().dtype());
    if (!device_input.ok()) continue;
    if (!cuda.value()->Copy(&device_input.value(), input.value()).ok())
      continue;
    OpContext context;
    context.context = cuda.value().get();
    context.stream = &cuda.value()->stream();
    Entry entry;
    entry.backend = "cuda";
    entry.level = "naive";
    if (!MeasureEntry(context, device_input.value(), 2, &entry)) continue;
    result.cuda = std::move(entry);
    result.has_cuda = true;
  }

  // Phase 2: CPU levels, level-major so each level sees the same machine state.
  for (CpuKernelLevel level : {CpuKernelLevel::kNaive, CpuKernelLevel::kOpenMP,
                               CpuKernelLevel::kTbb}) {
    if (!cpu.value()->backend_ptr()->SupportsKernelLevel(level)) continue;
    for (ConfigResult& result : results) {
      const Config& config = result.config;
      auto input = cpu.value()->AllocateTensor(
          Shape{config.height, config.width, config.channels},
          DataType::kFloat32);
      if (!input.ok()) continue;
      FillInput(&input.value());
      OpContext context;
      context.context = cpu.value().get();
      context.stream = &cpu.value()->stream();
      context.cpu_level = level;
      Entry entry;
      entry.backend = "cpu";
      entry.level = CpuKernelLevelName(level);
      if (!MeasureEntry(context, input.value(), 1, &entry)) continue;
      result.cpu.push_back(std::move(entry));
    }
  }

  // Phase 3: print per configuration with the naive CPU row as baseline.
  for (const ConfigResult& result : results) {
    double forward_base = 0.0;
    double inverse_base = 0.0;
    for (const Entry& entry : result.cpu) {
      if (entry.level != "naive") continue;
      if (entry.has_forward) forward_base = entry.forward_ms;
      if (entry.has_inverse) inverse_base = entry.inverse_ms;
    }
    for (const Entry& entry : result.cpu) {
      if (entry.has_forward) {
        PrintRow(result.config, entry.backend, entry.level, "forward",
                 entry.forward_ms, forward_base);
      }
      if (entry.has_inverse) {
        PrintRow(result.config, entry.backend, entry.level, "inverse",
                 entry.inverse_ms, inverse_base);
      }
    }
    if (result.has_cuda) {
      if (result.cuda.has_forward) {
        PrintRow(result.config, "cuda", "naive", "forward",
                 result.cuda.forward_ms, forward_base);
      }
      if (result.cuda.has_inverse) {
        PrintRow(result.config, "cuda", "naive", "inverse",
                 result.cuda.inverse_ms, inverse_base);
      }
    }
  }
  return 0;
}
