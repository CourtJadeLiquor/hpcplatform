// Dependency-free assertion and test registration framework shared by
// core/backends/ops tests.
#ifndef MYCV_TESTS_TEST_FRAMEWORK_H_
#define MYCV_TESTS_TEST_FRAMEWORK_H_

// Tiny dependency-free test harness: the research platform must build offline
// and on bare toolchains. Swap for GoogleTest later without touching test
// bodies if richer features are needed.

#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace mycv_test {

using TestFn = void (*)();

struct TestCase {
  const char* name;
  TestFn fn;
};

inline std::vector<TestCase>& Registry() {
  static std::vector<TestCase> registry;
  return registry;
}

inline int& CurrentFailures() {
  static int failures = 0;
  return failures;
}

struct Registrar {
  Registrar(const char* name, TestFn fn) { Registry().push_back({name, fn}); }
};

inline void ReportFailure(const char* file, int line,
                          const std::string& message) {
  ++CurrentFailures();
  std::cerr << "  FAIL " << file << ":" << line << "  " << message << "\n";
}

// Printable fallback for enum class values (BackendType, StatusCode, ...).
template <typename T>
typename std::enable_if<std::is_enum<T>::value, std::ostream&>::type operator<<(
    std::ostream& os, T value) {
  return os << static_cast<long long>(value);
}

template <typename A, typename B>
bool ExpectEq(const A& actual, const B& expected, const char* actual_expr,
              const char* expected_expr, const char* file, int line) {
  if (actual == expected) return true;
  std::ostringstream oss;
  oss << "EXPECT_EQ(" << actual_expr << ", " << expected_expr
      << ") actual=" << actual << " expected=" << expected;
  ReportFailure(file, line, oss.str());
  return false;
}

inline int RunAll() {
  int failed_tests = 0;
  int passed_tests = 0;
  for (const auto& test : Registry()) {
    CurrentFailures() = 0;
    std::cout << "[ RUN      ] " << test.name << "\n";
    test.fn();
    if (CurrentFailures() == 0) {
      ++passed_tests;
      std::cout << "[       OK ] " << test.name << "\n";
    } else {
      ++failed_tests;
      std::cout << "[  FAILED  ] " << test.name << "\n";
    }
  }
  std::cout << passed_tests << " test(s) passed, " << failed_tests
            << " test(s) failed\n";
  return failed_tests == 0 ? 0 : 1;
}

}  // namespace mycv_test

#define MYCV_TEST(test_name)                                                  \
  static void test_name();                                                    \
  static ::mycv_test::Registrar test_name##_registrar(#test_name, test_name); \
  static void test_name()

#define MYCV_EXPECT_TRUE(cond)                              \
  do {                                                      \
    if (!(cond)) {                                          \
      ::mycv_test::ReportFailure(__FILE__, __LINE__,        \
                                 "EXPECT_TRUE(" #cond ")"); \
    }                                                       \
  } while (0)

#define MYCV_EXPECT_FALSE(cond)                              \
  do {                                                       \
    if (cond) {                                              \
      ::mycv_test::ReportFailure(__FILE__, __LINE__,         \
                                 "EXPECT_FALSE(" #cond ")"); \
    }                                                        \
  } while (0)

#define MYCV_EXPECT_EQ(actual, expected)                                    \
  ::mycv_test::ExpectEq((actual), (expected), #actual, #expected, __FILE__, \
                        __LINE__)

#define MYCV_EXPECT_OK(expr)                                                 \
  do {                                                                       \
    ::mycv::Status mycv_status = (expr);                                     \
    if (!mycv_status.ok()) {                                                 \
      ::mycv_test::ReportFailure(                                            \
          __FILE__, __LINE__, "expected OK, got " + mycv_status.ToString()); \
    }                                                                        \
  } while (0)

#define MYCV_TEST_MAIN() \
  int main() { return ::mycv_test::RunAll(); }

#endif  // MYCV_TESTS_TEST_FRAMEWORK_H_
