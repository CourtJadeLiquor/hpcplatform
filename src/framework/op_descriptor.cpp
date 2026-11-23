#include "framework/op_descriptor.h"

#include <cerrno>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

namespace hpcplatform {
namespace {

using DescriptorMap = std::map<std::string, std::unique_ptr<OpDescriptor>>;

DescriptorMap& MutableRegistry() {
  static DescriptorMap registry;
  return registry;
}

std::once_flag g_register_once;

// Lazily registers every operator descriptor exactly once. The manifest call is
// compiled out when hpcplatform_WITH_YAML is off so a default build never references
// operator descriptors (keeping the link closed and the default build minimal).
void EnsureRegistered() {
  std::call_once(g_register_once, []() {
#ifdef hpcplatform_WITH_YAML
    RegisterAllOpDescriptors();
#endif
  });
}

}  // namespace

void RegisterOpDescriptor(std::unique_ptr<OpDescriptor> descriptor) {
  if (descriptor == nullptr) {
    return;
  }
  const std::string name = descriptor->Name();
  MutableRegistry()[name] = std::move(descriptor);
}

const OpDescriptor* FindOpDescriptor(const std::string& name) {
  EnsureRegistered();
  const auto& registry = MutableRegistry();
  const auto it = registry.find(name);
  return it == registry.end() ? nullptr : it->second.get();
}

// --- ParamMap typed accessors (no-throw parse, see header note) ------------

Result<int64_t> ParamGetInt64(const ParamMap& params, const std::string& key) {
  const auto it = params.find(key);
  if (it == params.end()) {
    return Status::InvalidArgument("missing required param: " + key);
  }
  const std::string& s = it->second;
  char* end = nullptr;
  errno = 0;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  if (end == s.c_str() || *end != '\0' || errno == ERANGE) {
    return Status::InvalidArgument("param '" + key +
                                   "' is not a valid integer: " + s);
  }
  return static_cast<int64_t>(v);
}

Result<double> ParamGetDouble(const ParamMap& params, const std::string& key) {
  const auto it = params.find(key);
  if (it == params.end()) {
    return Status::InvalidArgument("missing required param: " + key);
  }
  const std::string& s = it->second;
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || *end != '\0' || errno == ERANGE) {
    return Status::InvalidArgument("param '" + key +
                                   "' is not a valid number: " + s);
  }
  return v;
}

Result<std::string> ParamGetString(const ParamMap& params,
                                   const std::string& key) {
  const auto it = params.find(key);
  if (it == params.end()) {
    return Status::InvalidArgument("missing required param: " + key);
  }
  return it->second;
}

}  // namespace hpcplatform
