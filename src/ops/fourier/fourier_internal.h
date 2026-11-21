#ifndef MYCV_SRC_OPS_FOURIER_FOURIER_INTERNAL_H_
#define MYCV_SRC_OPS_FOURIER_FOURIER_INTERNAL_H_

// fourier 算子目录的内部契约（docs/architecture.md §6.1）。
// 本文件只声明入口与各后端共享的参数结构和函数，不含任何计算。
//
// 一维 DFT 定义为 X[k] = sum_n x[n] * exp(-i*2*pi*k*n/N)。二维核可分解
// 为行、列两个一维核的乘积，因此每个 (n, c) 平面都按“先对每行做 1-D、
// 再对每列做 1-D”完成 2-D 变换（两步顺序可交换，本实现固定先行后列）。
//
// 入口 fourier.cpp 负责校验、分派与分配输出，所有后端收到同一份参数：
//   正向  input = 图像（实部）  other = nullptr   output = Re(X)
//         params->imag = 新分配的 Im(X)，后端把虚部写入该张量
//   逆向  input = Re(X)         other = Im(X)     output = x 的实部
//         params->imag = nullptr，后端写回时统一乘逆向归一化 1/(H*W)

#include "framework/image_utils.h"
#include "mycv/ops/image.h"
#include "mycv/ops/op.h"

namespace mycv {
namespace ops_internal {

struct FourierParams {
  ImageLayout layout;  // 输入/输出图像的布局（NHWC 或 NCHW）
  bool inverse;        // false 正向；true 逆向
  // 正向：虚部输出槽位，由入口分配并持有，后端写入 Im(X)。
  // 逆向：不使用；虚部输入通过 `other` 传入，归一化在写回时一次完成。
  Tensor* imag;
};

// 各后端入口签名统一为 (context, input, other, params, output)。
// 张量由入口统一分配，后端只读 input/other、只写 output/params->imag。
// 参数含义见文件顶部；每个 (n, c) 平面独立变换。

// CPU 参考实现：顺序执行所有行、列，作为其他级别的逐字节基线。
Status FourierCpuNaive(const OpContext& context, const Tensor& input,
                       const Tensor* other, const void* params, Tensor* output);
#ifdef MYCV_WITH_OPENMP
// CPU OpenMP 级别：只把互相独立的行/列分给线程，累加顺序不变。
Status FourierCpuOpenMP(const OpContext& context, const Tensor& input,
                        const Tensor* other, const void* params,
                        Tensor* output);
#endif
#ifdef MYCV_WITH_TBB
// CPU TBB 级别：同一骨架，用 oneTBB 的 parallel_for 调度行/列。
Status FourierCpuTbb(const OpContext& context, const Tensor& input,
                     const Tensor* other, const void* params, Tensor* output);
#endif
#ifdef MYCV_WITH_CUDA
// CUDA 后端：整条流水线由设备内核完成，主机侧只生成 twiddle 表并发射。
Status FourierCuda(const OpContext& context, const Tensor& input,
                   const Tensor* other, const void* params, Tensor* output);
#endif

// 把本二进制内编译进来的实现注册到算子注册表。OpenCV 没有对应变换，
// 查找时返回 kNotImplemented，而不是悄悄用主机代码跑设备内存。
void RegisterFourierKernels();

}  // namespace ops_internal
}  // namespace mycv

#endif  // MYCV_SRC_OPS_FOURIER_FOURIER_INTERNAL_H_
