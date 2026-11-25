// CPU naive level for wavelet: instantiates the shared skeleton with only
// WaveletSequentialRunner and introduces no parallelism. It is the correctness
// baseline for the other levels (docs/architecture.md §7.2).

#include "wavelet_cpu.h"

namespace hpcplatform
{
    namespace ops_internal
    {

        // Sequentially executes row/column filter banks for every (n, c) plane.
        // Forward writes LL/LH/HL/HH; inverse reconstructs the image from the four
        // bands. Complexity is O(H*W*taps), with taps = 2 for Haar and 4 for Db4.
        Status WaveletCpuNaive(const OpContext&,
                               const Tensor& input,
                               const Tensor* /*other*/,
                               const void* params,
                               Tensor* output)
        {
            return WaveletImpl<WaveletSequentialRunner>(input, params, output);
        }

    }  // namespace ops_internal
}  // namespace hpcplatform
