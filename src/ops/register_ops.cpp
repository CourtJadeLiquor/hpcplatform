// 唯一的显式算子注册表（docs/architecture.md §6.3）。
//
// 每个算子一行注册函数；新增算子只需在此加一行，框架本身不点名任何算子。
// 刻意不使用静态自注册：静态库中未被引用的翻译单元不会被链接器拉入，
// 全局注册对象的构造函数会静默丢失。
//
// 查找流程（FindKernel）：以 (OpKind, BackendType, CpuKernelLevel) 三元组
// 查表，命中即返回内核函数指针。CPU 后端未构建该级别返回 kUnsupported；
// 其他后端未注册该算子/级别返回 kNotImplemented，而不是回退到主机内核。

#include "framework/op_registry.h"

namespace mycv {
namespace ops_internal {

void RegisterResizeKernels();
void RegisterGaussianBlurKernels();
void RegisterLaplacianPyramidKernels();
void RegisterFourierKernels();
void RegisterWaveletKernels();

// 按固定顺序调用各算子的注册入口。顺序不影响查找结果，注册表以三元组为键；
// 集中列出可保证静态库里的实现文件被引用，从而被链接进最终二进制。
// 复杂度 O(1)，仅调用各注册入口。
void RegisterAllKernels() {
  RegisterResizeKernels();
  RegisterGaussianBlurKernels();
  RegisterLaplacianPyramidKernels();
  RegisterFourierKernels();
  RegisterWaveletKernels();
}

}  // namespace ops_internal
}  // namespace mycv
