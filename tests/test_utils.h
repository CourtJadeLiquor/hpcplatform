// GoogleTest glue shared by the hpcplatform test suites. Everything else (TEST,
// EXPECT_*, gtest_main) comes straight from GoogleTest; only the Status
// assertion needs a project-side macro because hpcplatform reports errors without
// exceptions.
#ifndef hpcplatform_TESTS_TEST_UTILS_H_
#define hpcplatform_TESTS_TEST_UTILS_H_

#include <gtest/gtest.h>

#include "hpcplatform/core/error.h"

// Non-fatal assertion for Status: keeps the test body running (like EXPECT_*)
// and prints the status message on failure.
#define hpcplatform_EXPECT_OK(expr)                                    \
  do {                                                          \
    const ::hpcplatform::Status hpcplatform_status = (expr);                  \
    EXPECT_TRUE(hpcplatform_status.ok()) << hpcplatform_status.ToString();    \
  } while (0)

#endif  // hpcplatform_TESTS_TEST_UTILS_H_
