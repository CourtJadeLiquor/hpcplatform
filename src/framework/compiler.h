#ifndef MYCV_SRC_FRAMEWORK_COMPILER_H_
#define MYCV_SRC_FRAMEWORK_COMPILER_H_

// Compiler hint macros, centralized (docs/architecture.md §6.4). Every macro
// degrades to a plain expression on compilers without the hint, so kernels
// stay portable.

// Marks a pointer as not aliasing any other pointer accessed in the loop.
#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define MYCV_RESTRICT __restrict
#else
#define MYCV_RESTRICT
#endif

// Branch-prediction hints for cold error paths inside hot loops.
#if defined(__GNUC__) || defined(__clang__)
#define MYCV_LIKELY(x) (__builtin_expect(!!(x), 1))
#define MYCV_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define MYCV_LIKELY(x) (x)
#define MYCV_UNLIKELY(x) (x)
#endif

#endif  // MYCV_SRC_FRAMEWORK_COMPILER_H_
