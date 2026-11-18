// End-to-end image pipeline benchmark.
//
// Runs one realistic preprocessing chain (resize -> normalize -> warp_affine ->
// crop -> transpose) and compares the native CPU reference against the CPU
// kernel levels and the optional OpenCV / CUDA backends.
//
// Three properties are reported explicitly:
//   * cache reuse: transient stage buffers are released every iteration, so a
//     caching allocator must hand the same blocks back. The probe in Measure()
//     compares stage pointers across iterations instead of assuming it.
//   * no silent fallback: a backend/level that cannot run the whole chain is
//     skipped with the exact operator and level in the message.
//   * first divergence: correctness is checked stage by stage, so a backend
//     mismatch names the operator that introduced it instead of only reporting
//     a final error.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "mycv/mycv.h"

namespace {

using namespace mycv;
using Clock = std::chrono::steady_clock;

// Pipeline geometry: a 1024x768 RGB frame preprocessed for inference.
constexpr int64_t kInputHeight = 768;
constexpr int64_t kInputWidth = 1024;
constexpr int64_t kInputChannels = 3;
constexpr int64_t kResizeHeight = 384;
constexpr int64_t kResizeWidth = 512;
constexpr int64_t kWarpHeight = 448;
constexpr int64_t kWarpWidth = 448;
constexpr int64_t kCropTop = 32;
constexpr int64_t kCropLeft = 32;
constexpr int64_t kCropHeight = 384;
constexpr int64_t kCropWidth = 384;

constexpr int kIterations = 20;
constexpr int kWarmupIterations = 3;

// Cross-backend acceptance. Two independent image implementations do not agree
// bit-for-bit: cv::warpAffine evaluates coordinates in fixed point (~1/128 px),
// which shifts pixels that straddle the rotated source border. The gate is
// therefore the mean absolute difference normalised by the stage value range
// (the usual way images are compared); a wrong transform or layout moves a
// large share of the image and fails by orders of magnitude. Max values are
// reported as diagnostics.
constexpr double kMeanTolerance = 1e-3;  // 0.1% of the stage value range

const std::array<const char*, 5> kStageNames{"resize", "normalize",
                                             "warp_affine", "crop", "transpose"};

const std::vector<double> kMean{0.485, 0.456, 0.406};
const std::vector<double> kStd{0.229, 0.224, 0.225};
const std::vector<int> kTransposeOrder{2, 0, 1};

// Forward source->destination affine: rotate 12 degrees and scale by 0.9 about
// the image center while mapping into the 448x448 destination canvas.
std::array<double, 6> ComputeAffineMatrix() {
  constexpr double kPi = 3.14159265358979323846;
  const double angle = 12.0 * kPi / 180.0;
  const double scale = 0.9;
  const double cosine = std::cos(angle) * scale;
  const double sine = std::sin(angle) * scale;
  const double source_cx = static_cast<double>(kResizeWidth) / 2.0;
  const double source_cy = static_cast<double>(kResizeHeight) / 2.0;
  const double target_cx = static_cast<double>(kWarpWidth) / 2.0;
  const double target_cy = static_cast<double>(kWarpHeight) / 2.0;
  return {cosine, sine,
          target_cx - cosine * source_cx - sine * source_cy,
          -sine, cosine,
          target_cy + sine * source_cx - cosine * source_cy};
}

const std::array<double, 6> kAffineMatrix = ComputeAffineMatrix();

// Per-run observations from the pipeline.
struct PipelineProbe {
  std::vector<const void*> stages;  // one data pointer per stage output
  std::array<double, 5> stage_ms{};  // resize, normalize, warp, crop, transpose
};

template <typename Fn>
Result<Tensor> TimedStage(double* slot, Fn&& fn) {
  const auto start = Clock::now();
  Result<Tensor> result = fn();
  *slot += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
  return result;
}

// Intermediates are local, so every stage output except the last is released
// before the pipeline returns and becomes reusable for the next call. When
// 'keep_stages' is non-null the outputs are retained for correctness checking.
Result<Tensor> RunPipeline(const OpContext& ops, const Tensor& input,
                           PipelineProbe* probe,
                           std::vector<Tensor>* keep_stages) {
  PipelineProbe scratch;
  PipelineProbe* observed = probe != nullptr ? probe : &scratch;
  observed->stages.clear();
  observed->stage_ms = {};

  Result<Tensor> resized = TimedStage(&observed->stage_ms[0], [&] {
    return Resize(ops, input, kResizeHeight, kResizeWidth);
  });
  if (!resized.ok()) return resized.status();
  observed->stages.push_back(resized.value().data());
  if (keep_stages != nullptr) keep_stages->push_back(resized.value());

  Result<Tensor> normalized = TimedStage(&observed->stage_ms[1], [&] {
    return Normalize(ops, resized.value(), kMean, kStd, 1e-6);
  });
  if (!normalized.ok()) return normalized.status();
  observed->stages.push_back(normalized.value().data());
  if (keep_stages != nullptr) keep_stages->push_back(normalized.value());

  Result<Tensor> warped = TimedStage(&observed->stage_ms[2], [&] {
    return WarpAffine(ops, normalized.value(), kAffineMatrix, kWarpHeight,
                      kWarpWidth);
  });
  if (!warped.ok()) return warped.status();
  observed->stages.push_back(warped.value().data());
  if (keep_stages != nullptr) keep_stages->push_back(warped.value());

  Result<Tensor> cropped = TimedStage(&observed->stage_ms[3], [&] {
    return Crop(ops, warped.value(), kCropTop, kCropLeft, kCropHeight,
                kCropWidth);
  });
  if (!cropped.ok()) return cropped.status();
  observed->stages.push_back(cropped.value().data());
  if (keep_stages != nullptr) keep_stages->push_back(cropped.value());

  Result<Tensor> transposed = TimedStage(&observed->stage_ms[4], [&] {
    return Transpose(ops, cropped.value(), kTransposeOrder);
  });
  if (!transposed.ok()) return transposed.status();
  observed->stages.push_back(transposed.value().data());
  if (keep_stages != nullptr) keep_stages->push_back(transposed.value());
  return transposed;
}

// Cross-backend comparison. Normalize does not rescale u8 input, so stages hold
// values up to ~255/std; an absolute float tolerance would flag pure rounding
// there, which is why the mean error is normalised by the stage range.
struct Difference {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double sum_abs = 0.0;
  double reference_range = 1.0;
  int64_t count = 0;
  int64_t outliers = 0;

  // Mean absolute difference relative to the stage's value range.
  double mean_relative() const {
    if (count == 0) return 0.0;
    return sum_abs / static_cast<double>(count) / reference_range;
  }
  double outlier_ratio() const {
    return count > 0 ? static_cast<double>(outliers) / static_cast<double>(count)
                     : 0.0;
  }
};

template <typename T>
Difference ComputeDifference(const T* actual, const T* reference, int64_t count) {
  Difference result;
  result.count = count;
  double range = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    const double expected = static_cast<double>(reference[i]);
    const double diff = std::fabs(static_cast<double>(actual[i]) - expected);
    result.max_abs = std::max(result.max_abs, diff);
    result.max_rel = std::max(result.max_rel, diff / std::max(1.0, std::fabs(expected)));
    result.sum_abs += diff;
    range = std::max(range, std::fabs(expected));
    if (diff > 1e-3 * std::max(1.0, std::fabs(expected))) ++result.outliers;
  }
  result.reference_range = std::max(1.0, range);
  return result;
}

Difference CompareStage(const Tensor& actual, const Tensor& reference) {
  if (actual.Numel() != reference.Numel() ||
      actual.dtype() != reference.dtype()) {
    Difference mismatch;
    mismatch.count = actual.Numel();
    mismatch.outliers = actual.Numel();
    mismatch.max_abs = std::numeric_limits<double>::infinity();
    mismatch.max_rel = std::numeric_limits<double>::infinity();
    mismatch.sum_abs = std::numeric_limits<double>::infinity();
    mismatch.reference_range = 1.0;
    return mismatch;
  }
  if (actual.dtype() == DataType::kUInt8) {
    return ComputeDifference(actual.Data<uint8_t>(), reference.Data<uint8_t>(),
                             actual.Numel());
  }
  if (actual.dtype() == DataType::kFloat32) {
    return ComputeDifference(actual.Data<float>(), reference.Data<float>(),
                             actual.Numel());
  }
  Difference unsupported;
  unsupported.count = actual.Numel();
  unsupported.outliers = actual.Numel();
  unsupported.sum_abs = std::numeric_limits<double>::infinity();
  return unsupported;
}

struct PipelineComparison {
  bool matches = false;
  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;
  double mean_rel_diff = 0.0;
  double outlier_ratio = 0.0;
  int divergent_stage = -1;  // -1 when every stage is within tolerance
};

PipelineComparison ComparePipelines(const std::vector<Tensor>& actual,
                                    const std::vector<Tensor>& reference) {
  PipelineComparison comparison;
  if (actual.size() != reference.size()) return comparison;
  comparison.matches = true;
  for (size_t stage = 0; stage < actual.size(); ++stage) {
    const Difference diff = CompareStage(actual[stage], reference[stage]);
    comparison.max_abs_diff = std::max(comparison.max_abs_diff, diff.max_abs);
    comparison.max_rel_diff = std::max(comparison.max_rel_diff, diff.max_rel);
    comparison.mean_rel_diff =
        std::max(comparison.mean_rel_diff, diff.mean_relative());
    comparison.outlier_ratio =
        std::max(comparison.outlier_ratio, diff.outlier_ratio());
    if (diff.mean_relative() > kMeanTolerance) {
      comparison.matches = false;
      comparison.divergent_stage = static_cast<int>(stage);
      return comparison;
    }
  }
  return comparison;
}

struct BenchmarkRow {
  std::string backend;
  std::string level;
  double ms_per_iteration = 0.0;
  double images_per_second = 0.0;
  double speedup = 1.0;
  size_t peak_bytes = 0;
  bool cache_reused = false;
  bool matches = false;
  double max_abs_diff = 0.0;
  double max_rel_diff = 0.0;
  double mean_rel_diff = 0.0;
  double outlier_ratio = 0.0;
  std::string divergent_stage;
  std::array<double, 5> stage_ms{};
};

// Uploads/prepares the input once, then times steady-state pipeline calls. The
// input stays resident for every iteration; no per-iteration host<->device copy
// is included in the measurement.
Result<BenchmarkRow> Measure(const std::string& backend_name, Context& context,
                             Context& host_context, CpuKernelLevel level,
                             const Tensor& host_input,
                             const std::vector<Tensor>& reference_stages) {
  OpContext ops;
  ops.context = &context;
  ops.stream = &context.stream();
  ops.cpu_level = level;

  auto input = context.AllocateTensor(host_input.shape(), host_input.dtype());
  if (!input.ok()) return input.status();
  MYCV_RETURN_IF_ERROR(context.Copy(&input.value(), host_input));

  // Cache-reuse probe: two identical runs with all transients released in
  // between. Equal stage pointers mean the allocator recycled its blocks.
  std::vector<const void*> first_run;
  {
    PipelineProbe probe;
    auto output = RunPipeline(ops, input.value(), &probe, nullptr);
    if (!output.ok()) return output.status();
    first_run = probe.stages;
  }
  std::vector<const void*> second_run;
  {
    PipelineProbe probe;
    auto output = RunPipeline(ops, input.value(), &probe, nullptr);
    if (!output.ok()) return output.status();
    second_run = probe.stages;
  }
  bool cache_reused = first_run.size() == second_run.size();
  for (size_t i = 0; cache_reused && i < first_run.size(); ++i) {
    cache_reused = first_run[i] == second_run[i];
  }

  for (int i = 0; i < kWarmupIterations; ++i) {
    auto output = RunPipeline(ops, input.value(), nullptr, nullptr);
    if (!output.ok()) return output.status();
  }

  BenchmarkRow row;
  row.backend = backend_name;
  row.level = CpuKernelLevelName(level);

  Tensor last;
  const auto start = Clock::now();
  for (int i = 0; i < kIterations; ++i) {
    PipelineProbe probe;
    auto output = RunPipeline(ops, input.value(), &probe, nullptr);
    if (!output.ok()) return output.status();
    for (size_t stage = 0; stage < probe.stage_ms.size(); ++stage) {
      row.stage_ms[stage] += probe.stage_ms[stage];
    }
    last = output.value();
  }
  const double elapsed_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - start).count();

  row.ms_per_iteration = elapsed_ms / kIterations;
  for (double& stage : row.stage_ms) stage /= kIterations;
  row.images_per_second = 1000.0 / row.ms_per_iteration;
  row.peak_bytes = context.memory().Stats().peak_bytes;
  row.cache_reused = cache_reused;

  // Correctness runs after the timed loop and keeps every stage alive, which
  // would inflate peak_bytes, so memory is sampled before this point.
  std::vector<Tensor> actual_stages;
  auto checked = RunPipeline(ops, input.value(), nullptr, &actual_stages);
  if (!checked.ok()) return checked.status();

  // Device stages are copied back through host memory before comparison.
  std::vector<Tensor> host_stages;
  host_stages.reserve(actual_stages.size());
  for (const Tensor& stage : actual_stages) {
    auto host_stage = host_context.AllocateTensor(stage.shape(), stage.dtype());
    if (!host_stage.ok()) return host_stage.status();
    MYCV_RETURN_IF_ERROR(host_context.Copy(&host_stage.value(), stage));
    host_stages.push_back(host_stage.value());
  }

  const PipelineComparison comparison =
      ComparePipelines(host_stages, reference_stages);
  row.matches = comparison.matches;
  row.max_abs_diff = comparison.max_abs_diff;
  row.max_rel_diff = comparison.max_rel_diff;
  row.mean_rel_diff = comparison.mean_rel_diff;
  row.outlier_ratio = comparison.outlier_ratio;
  if (comparison.divergent_stage >= 0) {
    row.divergent_stage = kStageNames[static_cast<size_t>(
        comparison.divergent_stage)];
  }
  return row;
}

struct Candidate {
  std::string backend;
  BackendType type;
  CpuKernelLevel level;
};

void PrintHeader() {
  std::cout << "mycv image pipeline benchmark\n"
            << "  input      : " << kInputHeight << "x" << kInputWidth << "x"
            << kInputChannels << " uint8 (NHWC, reused across iterations)\n"
            << "  chain      : resize -> normalize -> warp_affine -> crop -> "
               "transpose\n"
            << "  stages     : resize " << kResizeHeight << "x" << kResizeWidth
            << " | warp_affine " << kWarpHeight << "x" << kWarpWidth
            << " | crop " << kCropHeight << "x" << kCropWidth
            << " | transpose 3x" << kCropHeight << "x" << kCropWidth
            << " float32\n"
            << "  iterations : " << kIterations << " timed, "
            << kWarmupIterations << " warmup, match = mean |diff| <= "
            << (kMeanTolerance * 100.0) << "% of stage range\n"
            << "  cache_reuse: identical stage buffer addresses across two "
               "consecutive runs\n\n";
}

}  // namespace

int main() {
  auto host_result = Context::Create(BackendType::kNativeCpu);
  if (!host_result.ok()) {
    std::cerr << "failed to create native CPU context: "
              << host_result.status().ToString() << "\n";
    return 1;
  }
  std::shared_ptr<Context> host = host_result.value();

  auto image_result = host->AllocateTensor(
      Shape{kInputHeight, kInputWidth, kInputChannels}, DataType::kUInt8);
  if (!image_result.ok()) {
    std::cerr << "failed to allocate input: "
              << image_result.status().ToString() << "\n";
    return 1;
  }
  Tensor image = image_result.value();
  uint8_t* pixels = image.Data<uint8_t>();
  constexpr double kPi = 3.14159265358979323846;
  for (int64_t y = 0; y < kInputHeight; ++y) {
    for (int64_t x = 0; x < kInputWidth; ++x) {
      // Smooth synthetic frame: a diagonal gradient plus a low-frequency
      // pattern. Sharp per-pixel noise would amplify sub-pixel precision
      // differences between backends and make the timing comparison noisy.
      const double wave = 40.0 * std::sin(2.0 * kPi * x / 256.0) *
                         std::cos(2.0 * kPi * y / 192.0);
      for (int64_t c = 0; c < kInputChannels; ++c) {
        const double base = 96.0 + 96.0 * x / kInputWidth +
                            64.0 * y / kInputHeight + 16.0 * c + wave;
        const double clamped = std::min(255.0, std::max(0.0, base));
        pixels[(y * kInputWidth + x) * kInputChannels + c] =
            static_cast<uint8_t>(clamped);
      }
    }
  }

  OpContext reference_ops;
  reference_ops.context = host.get();
  reference_ops.stream = &host->stream();
  reference_ops.cpu_level = CpuKernelLevel::kNaive;

  std::vector<Tensor> reference_stages;
  auto reference_result =
      RunPipeline(reference_ops, image, nullptr, &reference_stages);
  if (!reference_result.ok()) {
    std::cerr << "reference pipeline failed: "
              << reference_result.status().ToString() << "\n";
    return 1;
  }

  const std::vector<Candidate> candidates = {
      {"native_cpu", BackendType::kNativeCpu, CpuKernelLevel::kNaive},
      {"native_cpu", BackendType::kNativeCpu, CpuKernelLevel::kOpenMP},
      {"native_cpu", BackendType::kNativeCpu, CpuKernelLevel::kTbb},
      {"opencv", BackendType::kOpenCV, CpuKernelLevel::kNaive},
      {"cuda", BackendType::kCUDA, CpuKernelLevel::kNaive},
  };

  std::vector<BenchmarkRow> rows;
  for (const Candidate& candidate : candidates) {
    const std::string label =
        candidate.backend + "/" + CpuKernelLevelName(candidate.level);

    auto context_result = Context::Create(candidate.type);
    if (!context_result.ok()) {
      std::cout << "# skip " << label << ": "
                << context_result.status().ToString() << "\n";
      continue;
    }
    std::shared_ptr<Context> context = context_result.value();
    if (!context->backend().SupportsKernelLevel(candidate.level)) {
      std::cout << "# skip " << label << ": backend does not provide this level\n";
      continue;
    }

    auto row = Measure(candidate.backend, *context, *host, candidate.level, image,
                       reference_stages);
    if (!row.ok()) {
      // Most commonly a stage has no kernel at this level; surface the exact
      // operator and level instead of falling back to another implementation.
      std::cout << "# skip " << label << ": " << row.status().ToString()
                << "\n";
      continue;
    }
    rows.push_back(row.value());
  }

  double baseline_ms = 0.0;
  for (const BenchmarkRow& row : rows) {
    if (row.backend == "native_cpu" && row.level == "naive") {
      baseline_ms = row.ms_per_iteration;
      break;
    }
  }
  if (baseline_ms <= 0.0 && !rows.empty()) {
    baseline_ms = rows.front().ms_per_iteration;
  }
  for (BenchmarkRow& row : rows) {
    row.speedup = baseline_ms > 0.0 ? baseline_ms / row.ms_per_iteration : 0.0;
  }

  PrintHeader();

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "backend,level,ms_per_iteration,images_per_second,"
               "speedup_vs_native_naive,peak_mib,cache_reuse,matches_cpu,"
               "first_divergent_stage,max_abs_diff,max_rel_diff,mean_rel_diff,"
               "outlier_pct,resize_ms,normalize_ms,warp_affine_ms,crop_ms,"
               "transpose_ms\n";
  for (const BenchmarkRow& row : rows) {
    std::cout << row.backend << "," << row.level << ","
              << row.ms_per_iteration << "," << row.images_per_second << ","
              << row.speedup << ","
              << (static_cast<double>(row.peak_bytes) / (1024.0 * 1024.0)) << ","
              << (row.cache_reused ? "yes" : "no") << ","
              << (row.matches ? "yes" : "no") << ","
              << (row.divergent_stage.empty() ? "-" : row.divergent_stage) << ","
              << std::setprecision(7) << row.max_abs_diff << ","
              << row.max_rel_diff << "," << row.mean_rel_diff << ","
              << (row.outlier_ratio * 100.0) << std::setprecision(3);
    for (double stage_ms : row.stage_ms) std::cout << "," << stage_ms;
    std::cout << "\n";
  }
  std::cout << "\n";

  std::cout << std::left << std::setw(12) << "backend" << std::setw(14)
            << "level" << std::right << std::setw(11) << "ms/frame"
            << std::setw(11) << "img/s" << std::setw(10) << "speedup"
            << std::setw(10) << "peak MiB" << std::setw(8) << "cache"
            << std::setw(8) << "match" << "\n";
  for (const BenchmarkRow& row : rows) {
    std::cout << std::left << std::setw(12) << row.backend << std::setw(14)
              << row.level << std::right << std::setw(11)
              << row.ms_per_iteration << std::setw(11)
              << row.images_per_second << std::setw(10) << row.speedup
              << std::setw(10)
              << (static_cast<double>(row.peak_bytes) / (1024.0 * 1024.0))
              << std::setw(8) << (row.cache_reused ? "yes" : "no")
              << std::setw(8) << (row.matches ? "yes" : "no") << "\n";
  }
  if (rows.empty()) {
    std::cout << "no candidate completed the pipeline\n";
    return 1;
  }

  const bool all_match = std::all_of(rows.begin(), rows.end(),
                                    [](const BenchmarkRow& row) {
                                      return row.matches;
                                    });
  if (!all_match) {
    std::cerr << "\nWARNING: these backends differ from the native CPU reference:\n";
    for (const BenchmarkRow& row : rows) {
      if (row.matches) continue;
      std::cerr << "  " << row.backend << "/" << row.level << " at stage '"
                << (row.divergent_stage.empty() ? "?" : row.divergent_stage)
                << "' (mean_rel=" << row.mean_rel_diff
                << ", max_abs=" << row.max_abs_diff
                << ", outliers=" << (row.outlier_ratio * 100.0) << "%)\n";
    }
    return 1;
  }
  return 0;
}
