#ifndef hpcplatform_SRC_FRAMEWORK_COMPILER_H_
#define hpcplatform_SRC_FRAMEWORK_COMPILER_H_

// Compiler hint macros, centralized (docs/architecture.md §6.4). Every macro
// degrades to a plain expression on compilers without the hint, so kernels
// stay portable.

// Marks a pointer as not aliasing any other pointer accessed in the loop.
#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)
#define hpcplatform_RESTRICT __restrict
#else
#define hpcplatform_RESTRICT
#endif

// Branch-prediction hints for cold error paths inside hot loops.
#if defined(__GNUC__) || defined(__clang__)
#define hpcplatform_LIKELY(x) (__builtin_expect(!!(x), 1))
#define hpcplatform_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
#define hpcplatform_LIKELY(x) (x)
#define hpcplatform_UNLIKELY(x) (x)
#endif

#endif  // hpcplatform_SRC_FRAMEWORK_COMPILER_H_
