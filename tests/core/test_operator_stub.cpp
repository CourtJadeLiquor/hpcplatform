// AC-6 / TR-3.1: the public LoadOperators API is linkable in every build.
// Without hpcplatform_WITH_YAML it returns kUnsupported (message mentions the macro);
// with the feature, an empty operator list loads to an empty map. This file is
// compiled in both build modes and selects the matching assertion.
#include "test_utils.h"

#include <string>

#include "hpcplatform/core/error.h"
#include "hpcplatform/ops/operator.h"

namespace {
using namespace hpcplatform;

#ifndef hpcplatform_WITH_YAML
TEST(hpcplatform, LoadOperatorsReturnsUnsupportedWithoutYaml) {
  auto result = LoadOperatorsFromString("operators: []");
  EXPECT_FALSE(result.ok());
  if (!result.ok()) {
    EXPECT_TRUE(result.status().code() == StatusCode::kUnsupported);
    EXPECT_TRUE(result.status().message().find("hpcplatform_WITH_YAML") !=
                     std::string::npos);
  }
}
#else
TEST(hpcplatform, LoadOperatorsAcceptsEmptyConfigWithYaml) {
  auto result = LoadOperatorsFromString("operators: []");
  EXPECT_TRUE(result.ok());
  if (result.ok()) {
    EXPECT_TRUE(result.value().empty());
  }
}
#endif

}  // namespace

