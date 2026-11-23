// hpcplatform umbrella header: includes all public APIs at once; optional OpenCV
// interoperability is gated by hpcplatform_WITH_OPENCV.
#ifndef hpcplatform_hpcplatform_H_
#define hpcplatform_hpcplatform_H_

#ifdef hpcplatform_WITH_OPENCV
#include "hpcplatform/interop/opencv.h"
#endif
#include "hpcplatform/core/context.h"
#include "hpcplatform/core/device.h"
#include "hpcplatform/core/dtype.h"
#include "hpcplatform/core/error.h"
#include "hpcplatform/core/kernel_level.h"
#include "hpcplatform/core/memory.h"
#include "hpcplatform/core/shape.h"
#include "hpcplatform/core/stream.h"
#include "hpcplatform/core/tensor.h"
#include "hpcplatform/core/timing.h"
#include "hpcplatform/operator.h"
#include "hpcplatform/ops/fourier.h"
#include "hpcplatform/ops/gaussian_blur.h"
#include "hpcplatform/ops/image.h"
#include "hpcplatform/ops/laplacian_pyramid.h"
#include "hpcplatform/ops/op.h"
#include "hpcplatform/ops/resize.h"
#include "hpcplatform/ops/wavelet.h"
#include "hpcplatform/version.h"

#endif  // hpcplatform_hpcplatform_H_
