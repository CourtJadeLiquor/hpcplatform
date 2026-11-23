// Host-side composed suboperator for laplacian_pyramid (native_cpu and OpenCV
// backends).
//
// A stage introduces no new pixel-level math and instead reuses two registered
// suboperators:
//   1. GaussianBlur: separable Gaussian smoothing of the current level G_i;
//   2. Resize: nearest downsampling to ceil(H/2) x ceil(W/2) for G_{i+1}, then
//      bilinear upsampling back to G_i dimensions;
//   3. Host element-wise subtraction: L_i = G_i - upsample(G_{i+1}).
// Outside the loop, the entry point directly uses G_n as the top-level residual,
// so L_n = G_n; reconstruction is G_i = L_i + upsample(G_{i+1}).
//
// Backend and level are not passed separately: OpContext is forwarded unchanged
// to both suboperators, which query kernels using input.backend() and
// context.cpu_level. The composed path therefore uses the same kernels and
// parameters as the public single-operator entry points, with identical semantics
// (including boundary and sampling rules).

#include <cstdint>
#include <utility>

#include "framework/image_utils.h"
#include "hpcplatform/ops/gaussian_blur.h"
#include "hpcplatform/ops/resize.h"
#include "laplacian_pyramid_internal.h"

namespace hpcplatform {
namespace ops_internal {
namespace {

// Read a band-pass sample from host memory and promote it to double; only f32/f64
// are supported (the entry point rejects other dtypes). Complexity: O(1).
inline double LoadBandValue(const char* source, DataType dtype) {
  if (dtype == DataType::kFloat32) {
    return *reinterpret_cast<const float*>(source);
  }
  return *reinterpret_cast<const double*>(source);
}

// Write a double result back to a band-pass tensor, narrowing to float for f32
// and otherwise storing double. Takes a byte pointer, dtype, and value, and
// returns nothing. Complexity: O(1).
inline void StoreBandValue(char* destination, DataType dtype, double value) {
  if (dtype == DataType::kFloat32) {
    *reinterpret_cast<float*>(destination) = static_cast<float>(value);
    return;
  }
  *reinterpret_cast<double*>(destination) = value;
}

// Linear index -> (n, y, x, c), using the same decomposition order as
// ImageLinearIndex: n,y,x,c for NHWC and n,c,y,x for NCHW.
// Complexity: O(rank), where rank <= 4.
inline void DecodeLinear(const ImageInfo& info, int64_t linear, int64_t& n,
                         int64_t& y, int64_t& x, int64_t& c) {
  int64_t remaining = linear;
  if (info.layout == ImageLayout::kNHWC) {
    c = remaining % info.c;
    remaining /= info.c;
    x = remaining % info.w;
    remaining /= info.w;
    y = remaining % info.h;
    n = remaining / info.h;
  } else {
    x = remaining % info.w;
    remaining /= info.w;
    y = remaining % info.h;
    remaining /= info.h;
    c = remaining % info.c;
    n = remaining / info.c;
  }
}

// Element-wise subtraction in host memory: L = fine - up (native_cpu and OpenCV
// tensors). fine may be a strided view (such as the caller's input), so use
// ImageByteOffset to resolve actual byte offsets from (n,y,x,c). up and output are
// newly allocated contiguous tensors and can use linear indices directly.
// Complexity: O(Numel), with one subtraction and one type conversion per element.
inline void SubtractBands(const Tensor& fine, const ImageInfo& info,
                          const Tensor& up, Tensor* output) {
  const char* fine_base = static_cast<const char*>(fine.data());
  const char* up_base = static_cast<const char*>(up.data());
  char* out_base = static_cast<char*>(output->data());
  const size_t up_element = up.ElementSize();
  const size_t out_element = output->ElementSize();
  for (int64_t i = 0; i < output->Numel(); ++i) {
    int64_t n = 0;
    int64_t y = 0;
    int64_t x = 0;
    int64_t c = 0;
    DecodeLinear(info, i, n, y, x, c);
    const double high = LoadBandValue(
        fine_base + ImageByteOffset(fine, info, n, y, x, c), fine.dtype());
    const double low = LoadBandValue(
        up_base + static_cast<size_t>(i) * up_element, up.dtype());
    StoreBandValue(out_base + static_cast<size_t>(i) * out_element,
                   output->dtype(), high - low);
  }
}

}  // namespace

// Host stage: compute one band-pass level L_i and the next Gaussian level G_{i+1}.
// Inputs: context (forwarded unchanged to suboperators), G_i (input), and params;
// other is unused.
// Outputs: output receives L_i (same dimensions as input);
// params->next_gaussian receives G_{i+1}.
// Steps: GaussianBlur followed by ResizeNearest for G_{i+1}, then ResizeBilinear
//        back to current dimensions, and finally host element-wise subtraction.
// Dimensions: the next level uses (h+1)/2 and (w+1)/2 (ceil half-size), matching
//             classic pyrDown level shapes for odd extents; upsampling always
//             targets the current level's h x w.
// Complexity: O(N * kernel_size), where N is the current level's element count.
Status LaplacianPyramidHostStage(const OpContext& context, const Tensor& input,
                                 const Tensor* /*other*/, const void* params,
                                 Tensor* output) {
  const auto& options = *static_cast<const LaplacianPyramidParams*>(params);
  auto image = ParseImage(input.shape(), options.layout);
  if (!image.ok()) return image.status();

  // Downsampling: apply Gaussian smoothing, then nearest sampling of every other
  // row and column; ceil produces classic pyrDown level shapes for odd extents.
  const int64_t next_h = (image.value().h + 1) / 2;
  const int64_t next_w = (image.value().w + 1) / 2;
  auto blurred = GaussianBlur(context, input, options.kernel_size,
                              options.sigma, options.layout);
  if (!blurred.ok()) return blurred.status();
  auto next = Resize(context, blurred.value(), next_h, next_w,
                     InterpolationMode::kNearest, options.layout);
  if (!next.ok()) return next.status();

  // Upsample to current level dimensions and subtract to obtain the band-pass
  // L_i = G_i - upsample(G_{i+1}).
  auto up = Resize(context, next.value(), image.value().h, image.value().w,
                   InterpolationMode::kBilinear, options.layout);
  if (!up.ok()) return up.status();
  SubtractBands(input, image.value(), up.value(), output);

  *options.next_gaussian = std::move(next.value());
  return Status::Ok();
}

}  // namespace ops_internal
}  // namespace hpcplatform
