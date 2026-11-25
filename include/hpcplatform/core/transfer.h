#ifndef hpcplatform_CORE_TRANSFER_H_
#define hpcplatform_CORE_TRANSFER_H_

#include "hpcplatform/core/error.h"

namespace hpcplatform {

class Tensor;
class Stream;

// Copies logical elements, including non-contiguous views. The backend that
// owns the destination tensor drives the transfer, so callers never need to
// name a Context: encode the destination domain in `dst` instead.
//
// Execution is synchronous: `dst` is readable when the call returns.
Status Copy(Tensor* dst, const Tensor& src, Stream* stream = nullptr);

// Copies host memory into a pre-allocated device tensor. The device backend is
// inferred from `device_dst`, so any Context that allocated the destination can
// be used interchangeably. `host_src` must reference host memory (kHost or
// kHostPinned) and match `device_dst` in shape and dtype.
Status Upload(const Tensor& host_src, Tensor* device_dst,
              Stream* stream = nullptr);

// Copies a device tensor into pre-allocated host memory. The device backend is
// inferred from `device_src`. `device_src` must reference device memory and
// match `host_dst` in shape and dtype.
Status Download(const Tensor& device_src, Tensor* host_dst,
                Stream* stream = nullptr);

}  // namespace hpcplatform

#endif  // hpcplatform_CORE_TRANSFER_H_
