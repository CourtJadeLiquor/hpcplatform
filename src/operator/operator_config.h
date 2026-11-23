#ifndef hpcplatform_SRC_OPERATOR_OPERATOR_CONFIG_H_
#define hpcplatform_SRC_OPERATOR_OPERATOR_CONFIG_H_

#include <map>
#include <string>
#include <vector>

#include "framework/op_descriptor.h"

namespace hpcplatform {

// One named operator instance as written in the config. Environment fields are
// kept as raw strings here; they are validated and resolved to typed values by
// the loader (see load_operators_real.cpp). `params` carries scalar values as
// strings; each descriptor parses them into its own typed parameter struct.
struct OperatorInstanceConfig {
  std::string name;
  std::string op;
  std::string backend;
  std::string cpu_level;  // empty => default naive
  int device_index = 0;
  ParamMap params;
};

struct OperatorsConfig {
  std::vector<OperatorInstanceConfig> operators;
};

// Parses a YAML document (already read into a string) into an OperatorsConfig.
// Syntax errors, missing required fields, wrong types, or duplicate instance
// names yield kInvalidArgument. Defined in yaml_parser.cpp (compiled only with
// hpcplatform_WITH_YAML).
Result<OperatorsConfig> ParseOperatorsConfig(const std::string& yaml_text);

}  // namespace hpcplatform

#endif  // hpcplatform_SRC_OPERATOR_OPERATOR_CONFIG_H_
