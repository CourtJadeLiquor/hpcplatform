#ifndef hpcplatform_SRC_OPS_FOURIER_FOURIER_CPU_H_
#define hpcplatform_SRC_OPS_FOURIER_FOURIER_CPU_H_

// Shared CPU skeleton for the fourier operator
// (docs/architecture.md §6.1/§7.2). The 2-D transform is separable: each (n, c)
// plane first applies a length-W 1-D transform to every row, then a length-H
// 1-D transform to every column, giving the 2-D form of
// X[k] = sum_n x[n]*exp(-i*2*pi*k*n/N).
//
// The Runner policy template distributes independent row/column tasks across
// parallel backends. Sequential, OpenMP, and TBB Runners each provide only
// For(begin, end, fn), differing solely in loop scheduling. The plane driver
// and 1-D kernel share one template implementation instead of three copies.
//
// Precision policy: all accumulation uses double and is narrowed to the tensor
// dtype only during write-back (as in gaussian_blur). Operation order within
// each row/column (bit-reversal, butterflies in ascending len order, and DFT
// fallback accumulation in ascending j order) is thread-independent, so all
// three CPU levels produce byte-identical (bit-exact) output.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef hpcplatform_WITH_TBB
#include <tbb/parallel_for.h>
#endif

#include "fourier_internal.h"

namespace hpcplatform
{
    namespace ops_internal
    {

        // Execution policy: each translation unit selects one, keeping the shared plane
        // driver as a single template. Each index passed to For(begin, end, fn) denotes
        // an independent row or column; fn does not modify shared state.
        struct SequentialRunner
        {
            template <typename Fn>
            static void For(int64_t begin, int64_t end, Fn&& fn)
            {
                for (int64_t i = begin; i < end; ++i) fn(i);
            }
        };

#ifdef hpcplatform_WITH_OPENMP
        // OpenMP policy: distributes independent rows/columns across omp parallel for
        // with the default static scheduling.
        struct OpenMPRunner
        {
            template <typename Fn>
            static void For(int64_t begin, int64_t end, Fn&& fn)
            {
#pragma omp parallel for
                for (int64_t i = begin; i < end; ++i) fn(i);
            }
        };
#endif

#ifdef hpcplatform_WITH_TBB
        // TBB policy: parallel_for automatically partitions work and balances load.
        struct TbbRunner
        {
            template <typename Fn>
            static void For(int64_t begin, int64_t end, Fn&& fn)
            {
                tbb::parallel_for(begin, end, [&](int64_t i) { fn(i); });
            }
        };
#endif

        // 1-D twiddle table for length n. It stores cos/sin of -2*pi*k/n, namely
        // twiddle = cos(-2*pi*k/N) + i*sin(-2*pi*k/N). The inverse only negates the
        // sine, so one table serves both directions and is shared by all rows/columns.
        struct FftPlan1D
        {
            int64_t n = 0;
            bool radix2 = false;
            std::vector<double> cos_table;
            std::vector<double> sin_table;
        };

        // Whether n is a power of two; only such lengths use O(n log n) radix-2.
        inline bool IsPowerOfTwo(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

        // Precomputes a length-n twiddle table in O(n). The table depends only on n;
        // constructing it once per plane lets all rows/columns reuse it and avoids
        // repeated trigonometric evaluations.
        inline FftPlan1D MakeFftPlan(int64_t n)
        {
            constexpr double kTwoPi = 6.283185307179586476925286766559;
            FftPlan1D plan;
            plan.n = n;
            plan.radix2 = IsPowerOfTwo(n);
            plan.cos_table.resize(static_cast<size_t>(n));
            plan.sin_table.resize(static_cast<size_t>(n));
            for (int64_t k = 0; k < n; ++k)
            {
                const double angle = -kTwoPi * static_cast<double>(k) / static_cast<double>(n);
                plan.cos_table[static_cast<size_t>(k)] = std::cos(angle);
                plan.sin_table[static_cast<size_t>(k)] = std::sin(angle);
            }
            return plan;
        }

        // Performs an in-place 1-D transform on `n` complex samples. `stride` is in
        // complex samples (1 within rows, W across columns). The transform kernel is
        // exp(-i*2*pi*k*n/N); the inverse negates the sine. The caller applies 1/(H*W)
        // normalization once during write-back.
        //
        // Power-of-two sizes: apply bit-reversal permutation, then iterative butterflies
        // for len = 2,4,...,n:
        //   a' = a + w*b, b' = a - w*b; complexity O(n log n).
        // Other sizes: fall back to an out-of-place direct DFT with accumulation in
        // ascending j order and O(n^2) complexity. Ascending j is deliberate, matching
        // the addition order of the CUDA DFT fallback kernel.
        inline void Transform1D(
            double* re, double* im, int64_t stride, const FftPlan1D& plan, bool inverse)
        {
            const int64_t n = plan.n;
            const double sign = inverse ? -1.0 : 1.0;

            if (plan.radix2)
            {
                // Bit-reversal permutation: swap index i with its bit-reversed index.
                for (int64_t i = 1, j = 0; i < n; ++i)
                {
                    int64_t bit = n >> 1;
                    for (; (j & bit) != 0; bit >>= 1) j ^= bit;
                    j ^= bit;
                    if (i < j)
                    {
                        std::swap(re[i * stride], re[j * stride]);
                        std::swap(im[i * stride], im[j * stride]);
                    }
                }
                // Iterative Cooley-Tukey butterflies: len doubles at each stage, and output
                // indices advance in ascending order within a stage.
                for (int64_t len = 2; len <= n; len <<= 1)
                {
                    const int64_t half = len >> 1;
                    const int64_t step = n / len;
                    for (int64_t base = 0; base < n; base += len)
                    {
                        for (int64_t j = 0; j < half; ++j)
                        {
                            const double wr = plan.cos_table[static_cast<size_t>(j * step)];
                            const double wi = sign * plan.sin_table[static_cast<size_t>(j * step)];
                            const int64_t a = (base + j) * stride;
                            const int64_t b = (base + j + half) * stride;
                            const double vr = re[b] * wr - im[b] * wi;
                            const double vi = re[b] * wi + im[b] * wr;
                            re[b] = re[a] - vr;
                            im[b] = im[a] - vi;
                            re[a] += vr;
                            im[a] += vi;
                        }
                    }
                }
                return;
            }

            // Non-power-of-two sizes: evaluate the O(n^2) DFT directly by definition,
            // which is correct for any n.
            std::vector<double> out_re(static_cast<size_t>(n));
            std::vector<double> out_im(static_cast<size_t>(n));
            for (int64_t k = 0; k < n; ++k)
            {
                double sum_re = 0.0;
                double sum_im = 0.0;
                int64_t index = 0;  // (j*k) mod n; add k per sample and wrap by subtraction
                for (int64_t j = 0; j < n; ++j)
                {
                    const double wr = plan.cos_table[static_cast<size_t>(index)];
                    const double wi = sign * plan.sin_table[static_cast<size_t>(index)];
                    const double xr = re[j * stride];
                    const double xi = im[j * stride];
                    sum_re += xr * wr - xi * wi;
                    sum_im += xr * wi + xi * wr;
                    index += k;
                    if (index >= n) index -= n;
                }
                out_re[static_cast<size_t>(k)] = sum_re;
                out_im[static_cast<size_t>(k)] = sum_im;
            }
            for (int64_t k = 0; k < n; ++k)
            {
                re[k * stride] = out_re[static_cast<size_t>(k)];
                im[k * stride] = out_im[static_cast<size_t>(k)];
            }
        }

        // Rows first, then columns: rows use stride 1 and length W; columns use stride
        // W and length H. Rows and columns are mutually independent. The policy only
        // decides which thread handles each line and never changes operation order
        // within a line, which is required for bit-exact output across CPU levels.
        template <typename Runner>
        inline void TransformPlane(double* re,
                                   double* im,
                                   int64_t h,
                                   int64_t w,
                                   const FftPlan1D& plan_h,
                                   const FftPlan1D& plan_w,
                                   bool inverse)
        {
            Runner::For(
                0, h, [&](int64_t y) { Transform1D(re + y * w, im + y * w, 1, plan_w, inverse); });
            Runner::For(0, w, [&](int64_t x) { Transform1D(re + x, im + x, w, plan_h, inverse); });
        }

        // Reads a double from a possibly non-contiguous input view using a byte offset;
        // supports f32/f64.
        inline double LoadFourierValue(const char* source, DataType dtype)
        {
            if (dtype == DataType::kFloat32)
            {
                return *reinterpret_cast<const float*>(source);
            }
            return *reinterpret_cast<const double*>(source);
        }

        // Writes a double to the output tensor, narrowing to f32 or retaining f64 only here.
        inline void StoreFourierValue(char* destination, DataType dtype, double value)
        {
            if (dtype == DataType::kFloat32)
            {
                *reinterpret_cast<float*>(destination) = static_cast<float>(value);
                return;
            }
            *reinterpret_cast<double*>(destination) = value;
        }

        // Loads one (n, c) plane into contiguous complex scratch space (real and
        // imaginary parts) in O(H*W). A nullptr `imag_source` denotes real-valued
        // forward input, so the imaginary part is filled with zero. Reads use
        // ImageByteOffset, allowing non-contiguous tensor views.
        inline void LoadPlane(const Tensor& real_source,
                              const Tensor* imag_source,
                              const ImageInfo& info,
                              int64_t n,
                              int64_t c,
                              std::vector<double>* re,
                              std::vector<double>* im)
        {
            const int64_t plane_size = info.h * info.w;
            re->resize(static_cast<size_t>(plane_size));
            im->resize(static_cast<size_t>(plane_size));
            const char* real_base = static_cast<const char*>(real_source.data());
            const char* imag_base =
                imag_source == nullptr ? nullptr : static_cast<const char*>(imag_source->data());
            const DataType real_dtype = real_source.dtype();
            const DataType imag_dtype =
                imag_source == nullptr ? DataType::kFloat32 : imag_source->dtype();
            for (int64_t y = 0; y < info.h; ++y)
            {
                for (int64_t x = 0; x < info.w; ++x)
                {
                    const int64_t linear = y * info.w + x;
                    (*re)[static_cast<size_t>(linear)] = LoadFourierValue(
                        real_base + ImageByteOffset(real_source, info, n, y, x, c), real_dtype);
                    if (imag_base != nullptr)
                    {
                        (*im)[static_cast<size_t>(linear)] = LoadFourierValue(
                            imag_base + ImageByteOffset(*imag_source, info, n, y, x, c),
                            imag_dtype);
                    }
                    else
                    {
                        (*im)[static_cast<size_t>(linear)] = 0.0;
                    }
                }
            }
        }

        // Writes one plane to the output tensor in O(H*W). `scale` is the inverse
        // normalization factor 1/(H*W) (1 for forward); a nullptr `imag_output` skips
        // the imaginary part. Output tensors are contiguous memory allocated by the
        // entry point.
        inline void StorePlane(const std::vector<double>& re,
                               const std::vector<double>& im,
                               const ImageInfo& info,
                               int64_t n,
                               int64_t c,
                               double scale,
                               Tensor* real_output,
                               Tensor* imag_output)
        {
            char* real_base = static_cast<char*>(real_output->data());
            char* imag_base =
                imag_output == nullptr ? nullptr : static_cast<char*>(imag_output->data());
            const DataType real_dtype = real_output->dtype();
            const DataType imag_dtype =
                imag_output == nullptr ? DataType::kFloat32 : imag_output->dtype();
            for (int64_t y = 0; y < info.h; ++y)
            {
                for (int64_t x = 0; x < info.w; ++x)
                {
                    const int64_t linear = y * info.w + x;
                    StoreFourierValue(real_base + ImageByteOffset(*real_output, info, n, y, x, c),
                                      real_dtype, re[static_cast<size_t>(linear)] * scale);
                    if (imag_base != nullptr)
                    {
                        StoreFourierValue(
                            imag_base + ImageByteOffset(*imag_output, info, n, y, x, c), imag_dtype,
                            im[static_cast<size_t>(linear)] * scale);
                    }
                }
            }
        }

        // Driver shared by all CPU levels: constructs row/column twiddle tables once,
        // then runs load -> TransformPlane -> store for each (n, c) plane. `imag_input`
        // is the entry point's `other` (inverse only); `options.imag` is the forward
        // imaginary-output slot.
        //
        // Complexity: O(H*W*(log H + log W)) per plane for radix-2; if either dimension
        // is not a power of two, falls back to O(H*W*(H + W)).
        template <typename Runner>
        inline Status FourierImpl(const Tensor& input,
                                  const Tensor* imag_input,
                                  const FourierParams& options,
                                  Tensor* real_output)
        {
            auto parsed = ParseImage(input.shape(), options.layout);
            if (!parsed.ok()) return parsed.status();
            const ImageInfo& info = parsed.value();

            const FftPlan1D plan_h = MakeFftPlan(info.h);
            const FftPlan1D plan_w = MakeFftPlan(info.w);
            const double scale =
                options.inverse ? 1.0 / (static_cast<double>(info.h) * static_cast<double>(info.w))
                                : 1.0;

            std::vector<double> re;
            std::vector<double> im;
            for (int64_t n = 0; n < info.n; ++n)
            {
                for (int64_t c = 0; c < info.c; ++c)
                {
                    LoadPlane(input, imag_input, info, n, c, &re, &im);
                    TransformPlane<Runner>(re.data(), im.data(), info.h, info.w, plan_h, plan_w,
                                           options.inverse);
                    StorePlane(re, im, info, n, c, scale, real_output, options.imag);
                }
            }
            return Status::Ok();
        }

    }  // namespace ops_internal
}  // namespace hpcplatform

#endif  // hpcplatform_SRC_OPS_FOURIER_FOURIER_CPU_H_
