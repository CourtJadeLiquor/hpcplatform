// YAML parser for the operator config. Compiled only with MYCV_WITH_YAML, so
// this is the only translation unit that includes yaml-cpp. yaml-cpp throws
// YAML::Exception on syntax errors; those are caught and converted to
// kInvalidArgument, matching how the OpenCV backend isolates third-party
// exceptions (see src/backends/opencv/opencv_tensor_access.cpp).

#include "operator_config.h"

#include <cstddef>
#include <map>
#include <string>

#include <yaml-cpp/yaml.h>

namespace mycv {
namespace {

// Reads a required string field from a map node. Returns kInvalidArgument if
// the key is absent or the node is not a scalar.
Result<std::string> RequireString(const YAML::Node& node, const std::string& key) {
  const YAML::Node field = node[key];
  if (!field || !field.IsScalar()) {
    return Status::InvalidArgument("missing or non-string field: " + key);
  }
  return field.Scalar();
}

}  // namespace

Result<OperatorsConfig> ParseOperatorsConfig(const std::string& yaml_text) {
  YAML::Node root;
  try {
    root = YAML::Load(yaml_text);
  } catch (const YAML::Exception& error) {
    return Status::InvalidArgument(std::string("YAML parse error: ") +
                                   error.what());
  }
  if (!root.IsMap()) {
    return Status::InvalidArgument(
        "config root must be a map with an 'operators' list");
  }
  const YAML::Node operators = root["operators"];
  if (!operators || !operators.IsSequence()) {
    return Status::InvalidArgument("'operators' must be a list");
  }

  OperatorsConfig config;
  std::map<std::string, bool> seen_names;
  for (std::size_t i = 0; i < operators.size(); ++i) {
    const YAML::Node entry = operators[i];
    if (!entry.IsMap()) {
      return Status::InvalidArgument("operator entry must be a map (index " +
                                     std::to_string(i) + ")");
    }

    OperatorInstanceConfig instance;

    auto name = RequireString(entry, "name");
    if (!name.ok()) return name.status();
    instance.name = std::move(name.value());
    if (instance.name.empty()) {
      return Status::InvalidArgument("operator name must not be empty");
    }
    if (seen_names.count(instance.name) != 0) {
      return Status::InvalidArgument("duplicate operator name: " +
                                     instance.name);
    }
    seen_names[instance.name] = true;

    auto op = RequireString(entry, "op");
    if (!op.ok()) return op.status();
    instance.op = std::move(op.value());

    auto backend = RequireString(entry, "backend");
    if (!backend.ok()) return backend.status();
    instance.backend = std::move(backend.value());

    if (entry["cpu_level"]) {
      const YAML::Node field = entry["cpu_level"];
      if (!field.IsScalar()) {
        return Status::InvalidArgument("cpu_level must be a string");
      }
      instance.cpu_level = field.Scalar();
    }

    if (entry["device_index"]) {
      const YAML::Node field = entry["device_index"];
      if (!field.IsScalar()) {
        return Status::InvalidArgument("device_index must be an integer");
      }
      try {
        instance.device_index = field.as<int>();
      } catch (const YAML::Exception&) {
        return Status::InvalidArgument("device_index must be an integer");
      }
      if (instance.device_index < 0) {
        return Status::InvalidArgument("device_index must be >= 0");
      }
    }

    if (entry["params"]) {
      const YAML::Node params = entry["params"];
      if (!params.IsMap()) {
        return Status::InvalidArgument("params must be a map");
      }
      for (YAML::const_iterator it = params.begin(); it != params.end(); ++it) {
        if (!it->first.IsScalar() || !it->second.IsScalar()) {
          return Status::InvalidArgument(
              "params keys and values must be scalars");
        }
        instance.params[it->first.Scalar()] = it->second.Scalar();
      }
    }

    config.operators.push_back(std::move(instance));
  }

  return config;
}

}  // namespace mycv
