// AC-6 / TR-3.1: the public LoadOperators API is linkable in every build.
// Without MYCV_WITH_YAML it returns kUnsupported (message mentions the macro);
// with the feature, an empty operator list loads to an empty map. This file is
// compiled in both build modes and selects the matching assertion.
#include "test_framework.h"

#include <string>

#include "mycv/core/error.h"
#include "mycv/operator.h"

namespace {
using namespace mycv;

#ifndef MYCV_WITH_YAML
MYCV_TEST(LoadOperatorsReturnsUnsupportedWithoutYaml) {
  auto result = LoadOperatorsFromString("operators: []");
  MYCV_EXPECT_FALSE(result.ok());
  if (!result.ok()) {
    MYCV_EXPECT_TRUE(result.status().code() == StatusCode::kUnsupported);
    MYCV_EXPECT_TRUE(result.status().message().find("MYCV_WITH_YAML") !=
                     std::string::npos);
  }
}
#else
MYCV_TEST(LoadOperatorsAcceptsEmptyConfigWithYaml) {
  auto result = LoadOperatorsFromString("operators: []");
  MYCV_EXPECT_TRUE(result.ok());
  if (result.ok()) {
    MYCV_EXPECT_TRUE(result.value().empty());
  }
}
#endif

}  // namespace

MYCV_TEST_MAIN()
