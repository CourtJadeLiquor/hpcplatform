#include <immintrin.h>

#include <cstring>

#include "gaussian_blur_cpu.h"

namespace hpcplatform
{
    namespace ops_internal
    {

        namespace
        {

            // AVX2 backend: two separable 1D convolution passes (horizontal, then
            // vertical), each O(H*W*K), reducing per-pixel cost from O(K^2) to O(2K).
            // Packs the channel dimension into 256-bit registers: a __m256d accumulator
            // processes 4 consecutive channels of the same pixel per iteration (256 bits =
            // 4 double lanes, equivalent to 8 float lanes). Prerequisites: contiguous input,
            // NHWC layout (HWC also parses as NHWC), and c >= 4. Other cases (HW/CHW/c < 4)
            // fall back wholesale to the shared scalar skeleton with byte-identical results.
            // Tap order matches naive, and separate mul + add operations avoid FMA,
            // ensuring bit-exact agreement with naive; the c % 4 channel tail uses scalars.
            constexpr int kLanes = 4;  // Channels covered by one __m256d (4 doubles)

            // Check whether the vector path applies: rank >= 3, NHWC layout, and enough channels to
            // fill a register.
            bool Avx2Applicable(const ImageInfo& info)
            {
                return info.rank >= 3 && info.layout == ImageLayout::kNHWC && info.c >= kLanes;
            }

            // Load 4 consecutive channels from base at first_linear and convert them to
            // 4 double lanes.
            // u8: memcpy 4 bytes -> _mm_cvtepu8_epi32 zero-extends to 4 int32 values
            //     -> _mm256_cvtepi32_pd; zero extension converts 0..255 directly to double
            //     without sign issues.
            // f32: _mm_loadu_ps loads 4 floats -> _mm256_cvtps_pd.
            // f64: _mm256_loadu_pd loads 4 doubles directly.
            // Returns a __m256d without modifying memory; O(1).
            __m256d LoadFourPixels(const char* base, DataType dtype, int64_t first_linear)
            {
                if (dtype == DataType::kUInt8)
                {
                    uint32_t four_bytes = 0;
                    std::memcpy(&four_bytes, base + first_linear, sizeof(four_bytes));
                    const __m128i bytes = _mm_cvtsi32_si128(static_cast<int>(four_bytes));
                    return _mm256_cvtepi32_pd(_mm_cvtepu8_epi32(bytes));
                }
                if (dtype == DataType::kFloat32)
                {
                    return _mm256_cvtps_pd(
                        _mm_loadu_ps(reinterpret_cast<const float*>(base + first_linear * 4)));
                }
                return _mm256_loadu_pd(reinterpret_cast<const double*>(base + first_linear * 8));
            }

            // One 4-channel vector block of the horizontal pass: compute 4 channels
            // starting at channel_base for (n,y,x) and write them to tmp. Steps:
            // 1) acc = 0 (__m256d; 4 double lanes correspond to 4 channels);
            // 2) For each tap i, clamp the x offset with replicate to obtain xx, broadcast
            //    coefficient k[i] to 4 lanes with _mm256_broadcast_sd, load 4 pixels, then
            //    mul and add (same order as naive; no FMA contraction);
            // 3) Store the 4 results to tmp with _mm256_storeu_pd.
            // O(K) per block; O(H*W*K) for the full horizontal pass.
            void HorizontalBlock(const Tensor& input,
                                 const ImageInfo& info,
                                 const double* kernel,
                                 int radius,
                                 int64_t n,
                                 int64_t y,
                                 int64_t x,
                                 int64_t channel_base,
                                 double* tmp)
            {
                const char* base = static_cast<const char*>(input.data());
                const int64_t tap_count = 2 * radius + 1;
                const int64_t block_base = ImageLinearIndex(info, n, y, x, channel_base);
                __m256d acc = _mm256_setzero_pd();
                for (int i = 0; i < tap_count; ++i)
                {
                    const int64_t xx = ClampIndex(x + i - radius, info.w);
                    const int64_t source = ImageLinearIndex(info, n, y, xx, channel_base);
                    const __m256d k = _mm256_broadcast_sd(&kernel[i]);
                    acc = _mm256_add_pd(
                        acc, _mm256_mul_pd(k, LoadFourPixels(base, input.dtype(), source)));
                }
                _mm256_storeu_pd(tmp + block_base, acc);
            }

            // One 4-channel vector block of the vertical pass: accumulate 4 channels at
            // (n,y,x) along y. Steps match the horizontal block, except clamping is along
            // y and the source is the double tmp buffer. Store according to output dtype:
            // - u8: store 4 lanes on the stack, then std::round and narrow each to uint8.
            //   Positive normalized coefficients keep results within [0,255], so no
            //   hardware saturation is needed.
            // - f32: _mm256_cvtpd_ps converts to 4 floats; _mm_storeu_ps stores them.
            // - f64: _mm256_storeu_pd directly stores 4 doubles.
            // O(K) per block; O(H*W*K) for the full vertical pass.
            void VerticalBlock(const double* tmp,
                               const ImageInfo& info,
                               const double* kernel,
                               int radius,
                               int64_t n,
                               int64_t y,
                               int64_t x,
                               int64_t channel_base,
                               Tensor* output)
            {
                const int64_t tap_count = 2 * radius + 1;
                __m256d acc = _mm256_setzero_pd();
                for (int i = 0; i < tap_count; ++i)
                {
                    const int64_t yy = ClampIndex(y + i - radius, info.h);
                    const int64_t source = ImageLinearIndex(info, n, yy, x, channel_base);
                    const __m256d k = _mm256_broadcast_sd(&kernel[i]);
                    acc = _mm256_add_pd(acc, _mm256_mul_pd(k, _mm256_loadu_pd(tmp + source)));
                }
                char* base = static_cast<char*>(output->data());
                if (output->dtype() == DataType::kUInt8)
                {
                    double lanes[kLanes];
                    _mm256_storeu_pd(lanes, acc);
                    uint8_t* bytes = reinterpret_cast<uint8_t*>(base);
                    const int64_t first = ImageLinearIndex(info, n, y, x, channel_base);
                    for (int j = 0; j < kLanes; ++j)
                    {
                        bytes[first + j] = static_cast<uint8_t>(std::round(lanes[j]));
                    }
                }
                else if (output->dtype() == DataType::kFloat32)
                {
                    _mm_storeu_ps(reinterpret_cast<float*>(
                                      base + ImageLinearIndex(info, n, y, x, channel_base) * 4),
                                  _mm256_cvtpd_ps(acc));
                }
                else
                {
                    _mm256_storeu_pd(reinterpret_cast<double*>(
                                         base + ImageLinearIndex(info, n, y, x, channel_base) * 8),
                                     acc);
                }
            }

            // Two-pass AVX2 implementation. Steps:
            // 1) Reject non-contiguous tensors and parse the image; if Avx2Applicable is
            //    false, fall back wholesale to BlurNaiveImpl (same accumulation order and
            //    byte-identical output);
            // 2) vectorized = c - c % 4, the portion aligned to 4 channels;
            // 3) Horizontal pass: traverse (n,y,x), call HorizontalBlock at each position
            //    with cb stepping by 4 channels, then send tail channels [vectorized, c) to
            //    scalar BlurHorizontalElement; write results to the double tmp buffer;
            // 4) Vertical pass: use the same traversal and tail split, accumulate from tmp,
            //    and write output.
            // Each pass is O(H*W*K), with O(H*W*C) doubles of additional memory.
            Status BlurAvx2Impl(const Tensor& input,
                                const GaussianBlurParams& options,
                                Tensor* output)
            {
                if (!input.IsContiguous())
                {
                    return Status::Unsupported(
                        "GaussianBlur AVX2 level requires contiguous tensors");
                }
                ImageInfo info;
                hpcplatform_RETURN_IF_ERROR(PrepareBlur(input, options, &info));
                if (!Avx2Applicable(info))
                {
                    return BlurNaiveImpl(input, options, output);
                }
                const std::vector<double> kernel =
                    BuildGaussianKernel(options.kernel_size, options.sigma);
                const int radius = options.kernel_size / 2;
                const int64_t vectorized = info.c - info.c % kLanes;
                std::vector<double> tmp(static_cast<size_t>(output->Numel()));
                for (int64_t n = 0; n < info.n; ++n)
                {
                    for (int64_t y = 0; y < info.h; ++y)
                    {
                        for (int64_t x = 0; x < info.w; ++x)
                        {
                            for (int64_t cb = 0; cb < vectorized; cb += kLanes)
                            {
                                HorizontalBlock(input, info, kernel.data(), radius, n, y, x, cb,
                                                tmp.data());
                            }
                            for (int64_t c = vectorized; c < info.c; ++c)
                            {
                                BlurHorizontalElement(input, info, kernel.data(), radius,
                                                      ImageLinearIndex(info, n, y, x, c),
                                                      tmp.data());
                            }
                        }
                    }
                }
                for (int64_t n = 0; n < info.n; ++n)
                {
                    for (int64_t y = 0; y < info.h; ++y)
                    {
                        for (int64_t x = 0; x < info.w; ++x)
                        {
                            for (int64_t cb = 0; cb < vectorized; cb += kLanes)
                            {
                                VerticalBlock(tmp.data(), info, kernel.data(), radius, n, y, x, cb,
                                              output);
                            }
                            for (int64_t c = vectorized; c < info.c; ++c)
                            {
                                BlurVerticalElement(tmp.data(), info, kernel.data(), radius,
                                                    ImageLinearIndex(info, n, y, x, c), output);
                            }
                        }
                    }
                }
                return Status::Ok();
            }

        }  // namespace

        // Entry point: convert parameters to GaussianBlurParams and run the AVX2
        // implementation; BlurAvx2Impl falls back to naive for inapplicable or
        // non-contiguous cases without changing semantics.
        Status GaussianBlurCpuAvx2(const OpContext&,
                                   const Tensor& input,
                                   const Tensor*,
                                   const void* params,
                                   Tensor* output)
        {
            return BlurAvx2Impl(input, *static_cast<const GaussianBlurParams*>(params), output);
        }

    }  // namespace ops_internal
}  // namespace hpcplatform
