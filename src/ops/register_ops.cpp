// The sole explicit operator registry (docs/architecture.md §6.3).
//
// Each operator has one registration-function line; adding an operator requires
// only one more line here, and the framework itself names no operators.
// Static self-registration is deliberately avoided: unreferenced translation
// units in a static library are not pulled in by the linker, silently dropping
// the constructors of global registration objects.
//
// FindKernel looks up the (OpKind, BackendType, CpuKernelLevel) triple and
// returns the kernel function pointer on a match. The CPU backend returns
// kUnsupported when a level was not built; other backends return
// kNotImplemented for an unregistered operator/level instead of falling back
// to a host kernel.

#include "framework/op_registry.h"

namespace mycv {
namespace ops_internal {

void RegisterResizeKernels();
void RegisterGaussianBlurKernels();
void RegisterLaplacianPyramidKernels();
void RegisterFourierKernels();
void RegisterWaveletKernels();

// Calls each operator's registration entry point in a fixed order. Order does
// not affect lookup because the registry is keyed by the triple. Listing them
// here ensures static-library implementation files are referenced and linked
// into the final binary. Complexity is O(1); only registration entries run.
void RegisterAllKernels() {
  RegisterResizeKernels();
  RegisterGaussianBlurKernels();
  RegisterLaplacianPyramidKernels();
  RegisterFourierKernels();
  RegisterWaveletKernels();
}

}  // namespace ops_internal
}  // namespace mycv
