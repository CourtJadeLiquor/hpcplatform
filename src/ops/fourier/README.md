# fourier —— 二维离散傅里叶变换算子

`FourierTransform` / `InverseFourierTransform` 对图像张量的每个 (n, c) 平面做二维
离散傅里叶变换（DFT），是频域滤波、频谱分析、卷积定理加速等链路的基础算子。
正向输出实部与虚部两个张量，逆向按 1/(H*W) 归一化后返回实部，保证
“正向 → 逆向”是单位增益。

- 公共头：[`include/hpcplatform/ops/fourier.h`](../../../include/hpcplatform/ops/fourier.h)
- 入口与注册：[`fourier.cpp`](fourier.cpp)
- 共享 CPU 骨架：[`fourier_cpu.h`](fourier_cpu.h)
- 测试：[`tests/test_fourier.cpp`](tests/test_fourier.cpp)
- 基准：[`benchmark/fourier_benchmark.cpp`](benchmark/fourier_benchmark.cpp)

## 1. 原理

### 1.1 定义

对 H × W 的实值平面 x[y, x]，正向变换（无归一化）为：

```
X[u, v] = sum over y = 0..H-1, x = 0..W-1 of
          x[y, x] * exp(-2*pi*i * (u*y/H + v*x/W))
```

u、v 分别是垂直、水平频率下标，值域 0..H-1、0..W-1；低频在四角（未做
fftshift）。逆向变换：

```
x[y, x] = (1/(H*W)) * sum over u, v of
          X[u, v] * exp(+2*pi*i * (u*y/H + v*x/W))
```

本算子把 X 拆成同形状的两个实张量：`ComplexImage::real` 存 Re(X)，
`ComplexImage::imag` 存 Im(X)。没有单独定义复数数据类型，是为了让频谱能直接复用
既有的 Tensor/Shape/后端基础设施（拷贝、布局、设备迁移都不需要新代码）。

### 1.2 可分离性

二维核 exp(-2*pi*i*(u*y/H + v*x/W)) 可以拆成两个一维核的乘积：

```
exp(-2*pi*i*(u*y/H + v*x/W))
  = exp(-2*pi*i*u*y/H) * exp(-2*pi*i*v*x/W)
```

因此二维 DFT 等价于“先沿行做 W 点一维 DFT，再沿列做 H 点一维 DFT”（顺序可交换）。
本实现先做行、再做列，一维变换复用同一个 radix-2 内核。

### 1.3 Parseval 定理

未归一化的正向变换满足：

```
sum |X[u, v]|^2 = H*W * sum |x[y, x]|^2
```

单元测试用它做整体正确性校验（不依赖具体相位约定）。

### 1.4 Cooley–Tukey（radix-2）

长度 n 为 2 的幂时，按奇偶下标拆分成两个 n/2 点变换，递归/迭代合并：

```
X[k] = E[k] + w^k * O[k]
X[k + n/2] = E[k] - w^k * O[k],   w = exp(-2*pi*i/n)
```

实现上先做位反转置换，再自底向上按 len = 2, 4, ..., n 做蝶形合并，复杂度
O(n log n)。长度为 n 的变换只需一张 n 项的 twiddle 表（cos/sin），正向与逆向
共用（逆向取正弦的相反数）。

## 2. 适用场景

| 场景 | 用法 |
| --- | --- |
| 频域滤波 | 正向变换后在频域乘掩膜，再逆向变换回图像域 |
| 频谱可视化 / 分析 | 取实部/虚部自行计算幅度 sqrt(re^2 + im^2) 与相位 atan2(im, re) |
| 卷积定理加速 | 大核卷积：正向变换 → 逐点相乘 → 逆向变换 |
| 图像配准 | 相位相关（互功率谱 + 逆向变换找峰值） |
| 教学 / 对拍 | 作为其他 FFT 实现（FFTW、cuFFT、pocketfft）的参考基线 |

不适合：需要非周期边界（本算子不含 padding/窗函数，边界即循环卷积语义）、
超大尺寸的实时处理（主机实现，未做 SIMD 与 GPU）。

## 3. 实现算法有哪几种

| 算法 | 复杂度 | 适用长度 | 说明 |
| --- | --- | --- | --- |
| 朴素 DFT | O(n^2) | 任意 | 直接按定义求和，最易验证，作为非 2 的幂长度的回退 |
| radix-2 Cooley–Tukey | O(n log n) | 2 的幂 | 位反转 + 迭代蝶形，工程上最常用的入门实现 |
| 混合基 / radix-4 / split-radix | O(n log n) | 2^k·3^m·5^p 等 | 减少乘法与循环层数，性能优于纯 radix-2 |
| Bluestein (chirp-z) | O(n log n) | 任意 | 把任意长度补成 2 的幂再做卷积，适合大素数长度 |
| FFTW / pocketfft | O(n log n) | 任意 | 自动选择算法与向量化，工业级性能上限 |
| cuFFT | O(n log n) | 任意 | GPU 批处理，适合大规模/多通道吞吐 |

本算子只依赖标准库，不引入第三方 FFT 库，因此选择“radix-2 + 朴素 DFT 回退”：
2 的幂长度走 radix-2，其余长度走 O(n^2) 保证正确性（README §6 给出复杂度代价）。

## 4. 本项目采用的实现

### 4.1 平面组织与布局

- 支持 rank-2 HW、rank-3 HWC/CHW、rank-4 NHWC/NCHW；逐 (n, c) 平面独立变换。
- 输入可以是非连续视图：读取用 `ImageByteOffset` 逐元素寻址；两个输出由入口
  统一分配为连续张量。
- 输出与输入同 shape、同 dtype、同 backend、同 device。

### 4.2 算法选择

```
n 是 2 的幂  → radix-2（位反转 + 迭代蝶形）
否则          → 直接 DFT（O(n^2) 求和，逐 k 累加）
```

行、列两次一维变换都用同一套 `FftPlan1D`（长度、是否 radix-2、twiddle 表）。
每个平面只构造一次表，所有行/列共享，避免重复三角函数计算。

### 4.3 精度约定

- 所有累加在 double 中完成，最后一步才窄化到张量 dtype（f32 或 f64），
  与 gaussian_blur 的“double 累加”策略一致。
- twiddle 表按 `-2*pi*k/n` 在 double 中计算；逆向变换只翻转正弦符号。
- 逆向归一化因子 1/(H*W) 在写回时统一乘一次，避免每级单独缩放。

### 4.4 并行级别与位精确

三个 CPU 级别共享同一套一维内核，只改变“哪一行/哪一列交给哪个线程”：

| 级别 | 实现文件 | 并行方式 |
| --- | --- | --- |
| naive | `fourier_cpu_naive.cpp` | 顺序执行所有行、列 |
| OpenMP | `fourier_cpu_omp.cpp` | `#pragma omp parallel for` 遍历行/列 |
| TBB | `fourier_cpu_tbb.cpp` | `tbb::parallel_for` 遍历行/列 |
| CUDA | `fourier_cuda.cu` | 每行/每列一个 block，块内 `__syncthreads()` 同步蝶形 |

由于每一行/列的运算顺序与线程无关，三个级别的输出逐字节相同（单元测试用
memcmp 校验）。OpenCV 未注册：请求该后端时注册表返回 `kNotImplemented`，不会
悄悄用主机代码跑设备内存。

### 4.5 CUDA 实现

`fourier_cuda.cu` 用设备内核完成整条流水线，主机侧只做发射：

| 内核 | 职责 |
| --- | --- |
| `FourierLoadPlaneKernel` | 把一个 (n, c) 平面按 `ImageElementOffset` 取到 double 暂存区，无虚部时填 0 |
| `FourierRadix2Kernel` | 一行/一列的位反转 + 迭代蝶形，就地完成；`stride` 区分行（1）与列（W） |
| `FourierDftKernel` | 非 2 的幂长度用 out-of-place O(n^2) DFT，写入乒乓缓冲 |
| `FourierStorePlaneKernel` | 按布局线性索引写回，逆向时乘 1/(H*W) |

- 一行/一列固定交给一个 block（`grid = 行数或列数`），块内线程用 `blockIdx`
  之外的一维索引分工，蝶形前后各一次 `__syncthreads()`，数据始终留在
  global memory，不需要共享内存分块。
- twiddle 表在主机端按 `-2*pi*k/n` 用 double 生成，一次 `cudaMemcpyAsync`
  上传到 device 的临时缓冲（H 与 W 各一张），整张图共享。
- 暂存区为 `2*H*W` 个 f64（实部/虚部）；只要 H 或 W 不是 2 的幂，再加一对
  `2*H*W` 的乒乓缓冲供 DFT 读写分离，全部通过 `Tensor::Empty` 复用
  内存池。
- 多通道时逐平面发射（load → rows → cols → store），所有发射串行进入同一
  stream，最后统一 `FinishLaunch` 同步一次。
- 设备端 FMA 收缩与主机不同，跨后端对拍用 `1e-4` 相对容差；同一后端内部仍然
  确定性可复现。

## 5. 复杂度

设单平面尺寸 H × W，通道数 c，批大小 n：

| 情形 | 每个一维变换 | 每个平面 | 整张图 |
| --- | --- | --- | --- |
| H、W 均为 2 的幂 | O(H log H)、O(W log W) | O(H log H + W log W) | O(n*c*(H log H + W log W)) |
| 含非 2 的幂长度 | O(n^2) | O(H*W*(H + W)) | O(n*c*H*W*(H + W)) |

空间：每个平面额外两份 double 缓冲（实部/虚部），即 16*H*W 字节，与通道数无关；
CUDA 在 H 或 W 不是 2 的幂时还要一对乒乓缓冲，即 32*H*W 字节。

## 6. 接口与示例

```cpp
#include "hpcplatform/hpcplatform.h"

hpcplatform::OpContext ctx;              // context / stream / cpu_level
auto spectrum = hpcplatform::FourierTransform(ctx, image);   // ComplexImage{real, imag}
auto magnitude = /* sqrt(real^2 + imag^2) 由调用方决定 */;

auto restored = hpcplatform::InverseFourierTransform(ctx, spectrum.value().real,
                                              spectrum.value().imag);
// restored 与 image 在 1e-5（f32）/ 1e-9（f64）内一致
```

参数校验：只接受 f32/f64（u8 无法表示有符号频谱，返回 `kUnsupported`）；
rank 必须为 2/3/4 且与 `layout` 匹配（否则 `kInvalidArgument`）；逆向要求
real/imag 的 shape、dtype、backend、device 完全一致。

## 7. 测试

[`tests/test_fourier.cpp`](tests/test_fourier.cpp) 覆盖：

| 测试 | 内容 |
| --- | --- |
| `FourierHandComputedFourPoint` | 1×4 输入 {1,2,3,4} → {10, -2+2i, -2, -2-2i}，手算对拍 |
| `FourierConstantImageIsImpulse` | 常数图能量集中在 DC（每通道 F[0,0] = H*W*0.25） |
| `FourierImpulseIsFlatSpectrum` | 冲激 → 全 1 的平坦谱 |
| `FourierSatisfiesParseval` | 7×5 非 2 的幂尺寸满足 sum|F|^2 = N * sum|x|^2 |
| `FourierRoundTripIsIdentity` | f32/f64、NHWC/NCHW、非 2 的幂尺寸的正逆往返 |
| `FourierLayoutsAgree` | 同一逻辑图像在 NHWC/NCHW 下频谱逐元素一致 |
| `FourierCpuLevelsAreBitExact` | naive / OpenMP / TBB 逐字节相同 |
| `FourierRejectsInvalidParameters` | u8、rank5、空 context、real/imag 形状或 dtype 不匹配 |
| `FourierOpenCvIsNotImplemented` | OpenCV 后端返回 `kNotImplemented`（按宏启用） |
| `FourierCudaMatchesCpu` | radix-2（16×12）、非 2 的幂（9×7）、NCHW（8×10×2）三种尺寸与 CPU 频谱对拍，`1e-4` |
| `FourierCudaRoundTripIsIdentity` | CUDA 上正向 + 逆向往返回到原图（`1e-4`） |

CUDA 用例在 `hpcplatform_WITH_CUDA` 且设备可用时执行；设备不可用时打印 SKIP 而不是失败。

全特性构建树（`build_debug`，GoogleTest + CTest）：

```bash
ctest --test-dir build_debug -R test_fourier --output-on-failure   # 11/11
```

## 8. 基准

[`benchmark/fourier_benchmark.cpp`](benchmark/fourier_benchmark.cpp) 输出 CSV：

```
op,backend,level,dtype,direction,in_h,in_w,c,ms_per_call,mpix_per_s,speedup
```

speedup 相对同尺寸、同通道数、同方向的 naive 行。测量方法：warmup 1 次，
每次 5 迭代；加速后端（CUDA）先于 CPU 行测量、跑两轮取较快值，CPU 三级按
级别分组测量，避免数秒 CPU 满载压低 GPU 时钟（与 laplacian_pyramid 基准同一
纪律）。

实测环境：Linux，24 逻辑 CPU，RTX 4060 Ti（CUDA arch 89），Release（-O2），
f32，图像为确定性填充。

| H×W | c | 方向 | naive ms | OpenMP ms (×) | TBB ms (×) | CUDA ms (×) |
| --- | --- | --- | --- | --- | --- | --- |
| 256² | 1 | forward | 3.30 | 3.05 (1.08) | 1.93 (1.70) | 0.18 (18.3) |
| 256² | 1 | inverse | 3.51 | 2.39 (1.47) | 2.28 (1.54) | 0.18 (19.6) |
| 256² | 3 | forward | 9.98 | 4.83 (2.06) | 5.33 (1.87) | 0.48 (20.6) |
| 256² | 3 | inverse | 10.19 | 5.42 (1.88) | 5.60 (1.82) | 0.49 (20.8) |
| 512² | 1 | forward | 23.08 | 7.51 (3.07) | 5.48 (4.21) | 0.65 (35.7) |
| 512² | 1 | inverse | 22.82 | 6.38 (3.58) | 5.58 (4.09) | 0.65 (35.0) |
| 512² | 3 | forward | 64.41 | 22.92 (2.81) | 16.76 (3.84) | 1.87 (34.4) |
| 512² | 3 | inverse | 66.40 | 22.01 (3.02) | 17.16 (3.87) | 1.91 (34.8) |
| 1024² | 1 | forward | 106.57 | 30.98 (3.44) | 29.40 (3.62) | 2.74 (39.0) |
| 1024² | 1 | inverse | 107.97 | 30.39 (3.55) | 29.32 (3.68) | 2.80 (38.5) |
| 1024² | 3 | forward | 311.56 | 87.90 (3.54) | 80.29 (3.88) | 8.73 (35.7) |
| 1024² | 3 | inverse | 313.33 | 87.68 (3.57) | 79.21 (3.96) | 7.89 (39.7) |

结论：**CUDA 是全部配置的最优后端**（比 naive 快 18–40 倍，比 TBB 快 8–11
倍），且随尺寸增大优势更稳定；CPU 侧 TBB 仍是最优级别（1.7–4.2 倍），OpenMP
紧随其后。小尺寸（256²、c=3）并行收益有限，主要是单次调用工作量太小、线程
唤醒与内存带宽占比上升。CUDA 每行/列一个 block + 就地蝶形的布局没有共享内存
分块，全部读写落在 global memory：1024²×1 约 383 Mpx/s，NHWC 三通道时因平面
加载/写回按通道步长交错，降到约 120 Mpx/s（见 §9）。

## 9. 已知限制与后续方向

- 非 2 的幂长度回退到 O(n^2) DFT；大尺寸非 2 的幂会明显变慢（可加 Bluestein）。
- 未做 SIMD / AVX2，单线程性能与 FFTW 仍有差距。
- CUDA 版本仍是“每行/列一个 block + 就地蝶形”的入门布局：没有共享内存分块，
  1024² 时受 global memory 带宽限制；非 2 的幂仍走 O(n^2) DFT（可换 Bluestein
  或 cuFFT 风格的混合基）。
- OpenCV 后端未注册；`cv::dft` 可作为后续跨后端对拍对象。
- 逆变换只返回实部，适用于实值图像；需要完整复数输出的场景需扩展接口。
- 频谱未做 fftshift，低频在四角；这是刻意的约定，便于与 1/(H*W) 归一化配合。
