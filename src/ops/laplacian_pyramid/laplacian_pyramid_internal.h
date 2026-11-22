#ifndef MYCV_SRC_OPS_LAPLACIAN_PYRAMID_LAPLACIAN_PYRAMID_INTERNAL_H_
#define MYCV_SRC_OPS_LAPLACIAN_PYRAMID_LAPLACIAN_PYRAMID_INTERNAL_H_

// Internal contract for the laplacian_pyramid operator directory
// (docs/architecture.md §6.1/§11). This operator has no pixel-level math of its
// own: each stage composes the registered GaussianBlur and Resize suboperators
// with one element-wise subtraction. Its "version" is therefore the
// (backend, CpuKernelLevel) pair forwarded to those two suboperators.
// This file only declares the parameter structure and host/device stage entry
// points; implementations reside in laplacian_pyramid_host.cpp and
// laplacian_pyramid_cuda.cu.
//
// Per-level semantics (plain text):
//   G_{i+1} = ResizeNearest(GaussianBlur(G_i))   size ceil(H/2) x ceil(W/2)
//   L_i     = G_i - upsample(G_{i+1})            same size as G_i
//   Top level L_n = G_n (the coarsest Gaussian level is the low-frequency residual)
// Reconstruction: G_i = L_i + upsample(G_{i+1}).

#include "framework/image_utils.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

namespace mycv {
namespace ops_internal {

// Immutable parameters for one stage, valid only for a synchronous kernel call.
struct LaplacianPyramidParams {
  int kernel_size;     // GaussianBlur kernel size: odd and >= 3
  double sigma;        // GaussianBlur standard deviation: finite and > 0
  ImageLayout layout;  // Image layout: NHWC or NCHW
  // Output slot for the next (half-size) Gaussian level G_{i+1}. The stage
  // allocates and move-assigns to this slot; the public entry point keeps it
  // alive for the duration of the call.
  Tensor* next_gaussian;
};

// Host stage: write L_i to output and G_{i+1} to params->next_gaussian.
// Inputs: context (backend/level/stream), G_i (input), and params; other is unused.
// Outputs: output is the band-pass L_i with the same dimensions as input;
// next_gaussian is the next Gaussian level.
// Steps: GaussianBlur -> ResizeNearest to ceil(H/2) x ceil(W/2) for G_{i+1},
//        then ResizeBilinear back to G_i dimensions and subtract element-wise.
// Complexity per level: O(N_i * kernel_size) for Gaussian filtering plus O(N_i)
// for resizing and subtraction.
Status LaplacianPyramidHostStage(const OpContext& context, const Tensor& input,
                                 const Tensor* other, const void* params,
                                 Tensor* output);
#ifdef MYCV_WITH_CUDA
// Device stage: uses the same per-level semantics as the host version, with CUDA
// suboperators and a final device subtraction kernel. OpContext must provide a
// CudaStream (context.stream or the stream in context.context).
// Complexity matches the host version; cross-backend values use a 1e-4 relative
// tolerance.
Status LaplacianPyramidCudaStage(const OpContext& context, const Tensor& input,
                                 const Tensor* other, const void* params,
                                 Tensor* output);
#endif

// Register every version built into this binary. Register only levels implemented
// by both resize and gaussian_blur (currently no SIMD because resize has no SIMD
// kernel), so unsupported levels fail at lookup rather than midway through the
// pyramid. Complexity: O(1); only writes to the registry.
void RegisterLaplacianPyramidKernels();

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_LAPLACIAN_PYRAMID_LAPLACIAN_PYRAMID_INTERNAL_H_
