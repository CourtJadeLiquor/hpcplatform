// Real implementation of LoadOperators, compiled only when hpcplatform_WITH_YAML is
// on. The always-compiled stub in load_operators.cpp forwards to the two
// entry points defined here (external linkage, matching the forward decls).
//
// Flow:
//   1. Parse YAML -> OperatorsConfig.
//   2. Resolve the execution environment (env var > shared context > per-entry).
//   3. Fail-fast validate every entry BEFORE constructing any object:
//      known op, valid params, kernel available, context create succeeds.
//      Any failure returns an error with an empty OperatorMap (no partial
//      objects, no silent fallback).
//   4. Construct all entries into the map and return.

#include "operator_config.h"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "framework/op_descriptor.h"
#include "framework/op_registry.h"
#include "hpcplatform/core/context.h"
#include "hpcplatform/core/device.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/kernel_level.h"
#include "hpcplatform/operator.h"

namespace hpcplatform {
namespace {

// Parses a non-negative integer from an env value (no exceptions).
Result<int> ParseDeviceIndex(const std::string& value) {
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  if (end == value.c_str() || *end != '\0' || errno == ERANGE || parsed < 0 ||
      parsed > INT_MAX) {
    return Status::InvalidArgument(
        "hpcplatform_DEVICE_INDEX must be a non-negative integer: " + value);
  }
  return static_cast<int>(parsed);
}

// Resolved top-level environment. `shared` means every entry reuses `context`
// and `cpu_level`; otherwise each entry builds its own.
struct Env {
  bool shared = false;
  BackendType backend = BackendType::kNativeCpu;
  CpuKernelLevel cpu_level = CpuKernelLevel::kNaive;
  int device_index = 0;
  std::shared_ptr<Context> context;
};

// Priority (high -> low): hpcplatform_BACKEND env var, caller-provided shared context,
// per-entry config. An env-requested backend that cannot be created fails the
// whole load (no fallback), per spec.
Result<Env> ResolveTopLevelEnv(
    const std::shared_ptr<Context>* shared_context) {
  Env env;

  const char* env_backend = std::getenv("hpcplatform_BACKEND");
  if (env_backend != nullptr && env_backend[0] != '\0') {
    auto backend = BackendTypeFromName(env_backend);
    if (!backend.ok()) return backend.status();
    env.backend = backend.value();

    const char* env_level = std::getenv("hpcplatform_CPU_LEVEL");
    if (env_level != nullptr && env_level[0] != '\0') {
      auto level = CpuKernelLevelFromName(env_level);
      if (!level.ok()) return level.status();
      env.cpu_level = level.value();
    }

    const char* env_device = std::getenv("hpcplatform_DEVICE_INDEX");
    if (env_device != nullptr && env_device[0] != '\0') {
      auto device = ParseDeviceIndex(env_device);
      if (!device.ok()) return device.status();
      env.device_index = device.value();
    }

    auto created = Context::Create(env.backend, env.device_index);
    if (!created.ok()) return created.status();
    env.context = std::move(created.value());
    env.shared = true;
    return env;
  }

  if (shared_context != nullptr) {
    if (*shared_context == nullptr) {
      return Status::InvalidArgument("shared_context must not be null");
    }
    env.backend = (*shared_context)->backend_type();
    env.cpu_level = CpuKernelLevel::kNaive;
    env.context = *shared_context;
    env.shared = true;
    return env;
  }

  // Default: each entry owns its own context.
  return env;
}

struct ResolvedEntry {
  const OpDescriptor* desc = nullptr;
  std::shared_ptr<Context> context;
  CpuKernelLevel cpu_level = CpuKernelLevel::kNaive;
};

// Validates every entry and resolves its (descriptor, context, cpu_level)
// without constructing any Operator. Kernel availability is checked with
// FindKernel, which also enforces the non-CPU + non-naive -> kInvalidArgument
// rule (mirrors ValidateInvocation in src/ops/common/op_common.h).
Result<std::vector<ResolvedEntry>> ValidateAll(const OperatorsConfig& config,
                                               const Env& env) {
  std::vector<ResolvedEntry> resolved;
  resolved.reserve(config.operators.size());
  for (const auto& entry : config.operators) {
    ResolvedEntry r;
    r.desc = FindOpDescriptor(entry.op);
    if (r.desc == nullptr) {
      return Status::InvalidArgument("unknown op: " + entry.op);
    }
    hpcplatform_RETURN_IF_ERROR(r.desc->ValidateParams(entry.params));

    if (env.shared) {
      r.context = env.context;
      r.cpu_level = env.cpu_level;
    } else {
      auto backend = BackendTypeFromName(entry.backend);
      if (!backend.ok()) return backend.status();
      CpuKernelLevel cpu_level = CpuKernelLevel::kNaive;
      if (!entry.cpu_level.empty()) {
        auto level = CpuKernelLevelFromName(entry.cpu_level);
        if (!level.ok()) return level.status();
        cpu_level = level.value();
      }
      auto created = Context::Create(backend.value(), entry.device_index);
      if (!created.ok()) return created.status();
      r.context = std::move(created.value());
      r.cpu_level = cpu_level;
    }

    auto kernel = ops_internal::FindKernel(r.desc->Kind(),
                                           r.context->backend_type(),
                                           r.cpu_level);
    if (!kernel.ok()) return kernel.status();

    resolved.push_back(std::move(r));
  }
  return resolved;
}

// Constructs all entries into a map. Returns an error (and discards the map) if
// any Construct fails, so callers never see a partially built OperatorMap.
Result<OperatorMap> BuildOperators(
    const OperatorsConfig& config,
    const std::vector<ResolvedEntry>& resolved) {
  OperatorMap built;
  for (std::size_t i = 0; i < config.operators.size(); ++i) {
    auto op = resolved[i].desc->Construct(
        config.operators[i].name, resolved[i].context, resolved[i].cpu_level,
        config.operators[i].params);
    if (!op.ok()) return op.status();
    built[config.operators[i].name] = std::move(op.value());
  }
  return built;
}

}  // namespace

Result<OperatorMap> LoadOperatorsFromText(
    const std::string& yaml,
    const std::shared_ptr<Context>* shared_context) {
  auto config = ParseOperatorsConfig(yaml);
  if (!config.ok()) return config.status();
  auto env = ResolveTopLevelEnv(shared_context);
  if (!env.ok()) return env.status();
  auto resolved = ValidateAll(config.value(), env.value());
  if (!resolved.ok()) return resolved.status();
  return BuildOperators(config.value(), resolved.value());
}

Result<OperatorMap> LoadOperatorsFromFile(
    const std::string& config_path,
    const std::shared_ptr<Context>* shared_context) {
  std::ifstream input(config_path);
  if (!input.is_open()) {
    return Status::InvalidArgument("cannot open config file: " + config_path);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return LoadOperatorsFromText(buffer.str(), shared_context);
}

}  // namespace hpcplatform
