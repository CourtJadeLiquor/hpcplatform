#ifndef hpcplatform_OPERATOR_H_
#define hpcplatform_OPERATOR_H_

#include <map>
#include <memory>
#include <string>

#include "hpcplatform/core/context.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/kernel_level.h"
#include "hpcplatform/ops/op.h"

namespace hpcplatform {

// A configured operator instance: holds a bound execution environment (Context
// + CPU kernel level) and the operator's typed parameters. Instances are built
// from a YAML config file by LoadOperators; callers may hold many instances
// (different parameters and/or backends) and orchestrate execution themselves.
//
// This base carries only metadata. Each operator provides a typed subclass with
// its own Run signature, so no input/output arity is assumed here: single-input,
// multi-input, and multi-output operators all fit the same pattern.
class Operator {
 public:
  virtual ~Operator() = default;

  // The instance name assigned in the config (the key in the returned map).
  virtual const std::string& name() const = 0;
  // The operator kind.
  virtual OpKind kind() const = 0;

  // Bound execution environment. context() is non-null for a fully constructed
  // instance.
  BackendType backend() const {
    return context_ ? context_->backend_type() : BackendType::kNativeCpu;
  }
  CpuKernelLevel cpu_level() const { return cpu_level_; }
  const std::shared_ptr<Context>& context() const { return context_; }

 protected:
  Operator() = default;

  std::shared_ptr<Context> context_;
  CpuKernelLevel cpu_level_ = CpuKernelLevel::kNaive;
};

// A collection of named, configured operator instances loaded from config.
using OperatorMap = std::map<std::string, std::unique_ptr<Operator>>;

// Loads operator instances from a YAML config file. Each entry specifies its own
// environment unless overridden (see docs/operator_config.md). On any validation
// failure returns an error and an empty map: fail-fast, no partially built
// objects, no silent fallback to another backend. Requires hpcplatform_WITH_YAML;
// returns kUnsupported when the feature is not compiled in.
Result<OperatorMap> LoadOperators(const std::string& config_path);

// Same as LoadOperators but reads the YAML document from a string. Useful for
// tests and for embedding configs in source.
Result<OperatorMap> LoadOperatorsFromString(const std::string& yaml);

// Loads operator instances that all share the given context. Per-entry
// backend/cpu_level/device_index in the config is ignored; the shared context's
// backend binds every instance. A null shared_context returns kInvalidArgument.
Result<OperatorMap> LoadOperators(const std::string& config_path,
                                  const std::shared_ptr<Context>& shared_context);

// Typed accessor: returns a pointer to T if `name` exists and its concrete type
// is T (or derives from T), otherwise nullptr.
template <typename T>
T* GetOperator(const OperatorMap& operators, const std::string& name) {
  const auto it = operators.find(name);
  if (it == operators.end() || it->second == nullptr) {
    return nullptr;
  }
  return dynamic_cast<T*>(it->second.get());
}

}  // namespace hpcplatform

#endif  // hpcplatform_OPERATOR_H_
