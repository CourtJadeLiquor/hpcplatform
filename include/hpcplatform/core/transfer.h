#ifndef hpcplatform_CORE_TRANSFER_H_
#define hpcplatform_CORE_TRANSFER_H_

#include "hpcplatform/core/error.h"

namespace hpcplatform {

class Tensor;
class Stream;

// ---------------------------------------------------------------------------
// Data movement between existing tensors.
//
// One primitive, two direction asserts:
//
//   Copy(src, &dst)                  direction-agnostic primitive
//   Upload(host_src, &device_dst)    Copy + "destination must be device"
//   Download(device_src, &host_dst)  Copy + "destination must be host"
//
// All three take (source, destination) with the destination written through a
// pointer, and all three are synchronous: `dst` is readable when the call
// returns. The executing backend is inferred from tensor ownership, so callers
// never name a Context for a transfer.
//
// Choose by one question — is the direction known at the call site?
//   yes -> Upload / Download (self-documenting, asserts the memory kinds)
//   no  -> Copy (host<->host across backends, device-to-device, pipelines whose
//          direction depends on data)
//
// Both tensors must already exist. To move a third-party type (cv::Mat, ...)
// use Tensor::From / Tensor::Into instead: the conversion owns the staging, so
// user code never spells Upload/Download.
// ---------------------------------------------------------------------------

// Copies logical elements, including non-contiguous views: reads follow the
// source strides and writes follow the destination strides. Requires matching
// shape and dtype. Works host<->host (any layout), host<->device and
// device<->device within one device; peer-to-peer across devices returns
// kUnsupported. Strided movement is a backend capability — the CUDA backend
// runs a single strided-copy kernel, host backends loop over elements.
Status Copy(const Tensor& src, Tensor* dst, Stream* stream = nullptr);

// Copies host memory (kHost or kHostPinned) into a pre-allocated device tensor.
// The device backend is inferred from `device_dst`, so any Context that
// allocated the destination can be used interchangeably.
Status Upload(const Tensor& host_src, Tensor* device_dst,
              Stream* stream = nullptr);

// Copies a device tensor into pre-allocated host memory. The device backend is
// inferred from `device_src`.
Status Download(const Tensor& device_src, Tensor* host_dst,
                Stream* stream = nullptr);

}  // namespace hpcplatform

#endif  // hpcplatform_CORE_TRANSFER_H_
