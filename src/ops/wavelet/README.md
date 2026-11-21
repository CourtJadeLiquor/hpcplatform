# wavelet —— 单层正交二维离散小波变换算子

`WaveletTransform` / `InverseWaveletTransform` 对每个 (n, c) 平面做一层可分离的
正交二维离散小波变换，把 H × W 图像（H、W 必须为偶数）拆成四个 H/2 × W/2 子带，
并能从四子带完美重建原图。支持 Haar 与 Daubechies-4 两种小波，周期边界。

- 公共头：[`include/mycv/ops/wavelet.h`](../../../include/mycv/ops/wavelet.h)
- 入口与注册：[`wavelet.cpp`](wavelet.cpp)
- 共享 CPU 骨架：[`wavelet_cpu.h`](wavelet_cpu.h)
- 测试：[`tests/test_wavelet.cpp`](tests/test_wavelet.cpp)
- 基准：[`benchmark/wavelet_benchmark.cpp`](benchmark/wavelet_benchmark.cpp)

## 1. 原理

### 1.1 一维分析与合成

给定长度为 N（偶数）的序列 s 和低通分解滤波器 h[0..taps-1]、高通滤波器
g[0..taps-1]，单层分析（降采样 2）为：

```
a[k] = sum j = 0..taps-1 of h[j] * s[2k + j]      k = 0..N/2-1
d[k] = sum j = 0..taps-1 of g[j] * s[2k + j]
```

正交小波满足“合成滤波器等于分析滤波器”，因此合成为：

```
x[n] = sum over k of h[n - 2k] * a[k] + g[n - 2k] * d[k]
```

下标按 N 取模（周期边界）。对 Haar（h = {1/sqrt(2), 1/sqrt(2)}，
g = {1/sqrt(2), -1/sqrt(2)}）代入可验证 a = (s0+s1)/sqrt(2)、
d = (s0-s1)/sqrt(2)、x0 = (a+d)/sqrt(2)、x1 = (a-d)/sqrt(2)，即完美重建。

### 1.2 滤波器系数

| 小波 | taps | 低通 h | 高通 g |
| --- | --- | --- | --- |
| Haar | 2 | {1/sqrt2, 1/sqrt2} | {1/sqrt2, -1/sqrt2} |
| Db4 | 4 | {(1+sqrt3)/(4*sqrt2), (3+sqrt3)/(4*sqrt2), (3-sqrt3)/(4*sqrt2), (1-sqrt3)/(4*sqrt2)} | g[k] = (-1)^k * h[3-k] |

两者都满足 sum(h) = sqrt(2)、sum(h^2) = 1；高通由
g[k] = (-1)^k * h[taps-1-k] 得到，构成正交镜像滤波器对（QMF）。

### 1.3 二维可分离与四带语义

二维小波变换 = 行方向一维分析 + 列方向一维分析，于是产生四个子带：

```
            列低通            列高通
行低通       LL（近似）        LH（水平细节）
行高通       HL（垂直细节）    HH（对角细节）
```

- LL：两方向低通，是下一级分解的输入（多级金字塔只需对 LL 迭代调用本算子）。
- LH：行低通、列高通，对水平方向的高频（竖直边缘）敏感。
- HL：行高通、列低通，对垂直方向的高频（水平边缘）敏感。
- HH：两方向高通，对角落/纹理敏感。

本项目的 HH/LH/HL 命名以“行/列”为准，单元测试用 2×2 手算值固定该约定。

### 1.4 完美重建与能量守恒

因为分析/合成使用同一组正交系数（paraunitary），所以：

- 重建：合成后再分析等于恒等变换，误差只来自浮点舍入。
- 能量：sum x^2 = sum (LL^2 + LH^2 + HL^2 + HH^2)，即变换是正交的。

这两条都在单元测试里做数值校验（见 §7）。

## 2. 适用场景

| 场景 | 用法 |
| --- | --- |
| 图像压缩 | 对 LL 递归分解，对细节带做量化/阈值（JPEG2000 同源思路） |
| 去噪 | 细节带软阈值后重建；周期边界 + 正交基避免边界振铃 |
| 边缘/纹理分析 | LH/HL/HH 分别对应水平、垂直、对角高频 |
| 多尺度金字塔 | 只对 LL 重复调用本算子，得到多级分解 |
| 特征工程 | 子带能量作为纹理描述子 |
| 教学 / 对拍 | 与 PyWavelets、MATLAB dwt2 的 `mode='periodization'` 对拍 |

不适合：需要平移不变性的分析（单层 + 降采样不是平移不变的，应用平稳小波
à trous）、需要整数无损压缩（应使用整数提升/5-3 小波）。

## 3. 实现算法有哪几种

| 算法 | 优点 | 代价 | 说明 |
| --- | --- | --- | --- |
| 卷积滤波器组（直接实现） | 公式直白、易验证、可任意 taps | 每个输出 taps 次乘加 | 本算子采用的方式 |
| 提升方案 (lifting) | 原地、约省一半乘法、可整数可逆 | 需要滤波器分解为提升步 | Haar/Db4 都有标准分解 |
| à trous / 平稳小波 | 平移不变、无下采样 | 输出与原图同尺寸，内存翻倍 | 分析型任务常用 |
| 小波包 | 对高频也继续分解，频带划分更细 | 需要最优基搜索 | 压缩/特征任务 |
| 双正交 CDF 9/7 | 线性相位、压缩性能好 | 非正交，重建有容差 | JPEG2000 有损路径 |
| 多级金字塔 DWT | 多尺度表示 | 需多次调用/递归 | 对 LL 迭代本算子即可 |

选型理由：本项目已有“单算子 + 可组合”的架构，单层正交卷积滤波器组最贴合
Tensor 接口，且能同时覆盖 Haar/Db4、给出位精确的 CPU 级别矩阵；提升方案
留作后续优化。

## 4. 本项目采用的实现

### 4.1 平面组织与布局

- 支持 rank-2 HW、rank-3 HWC/CHW、rank-4 NHWC/NCHW；逐 (n, c) 平面独立变换。
- 四个输出子带由入口统一分配为连续张量，保持输入的 dtype、layout、backend、
  device；形状为 H/2 × W/2（对应布局的维度顺序）。
- 输入可以是非连续视图（读取走 `ImageByteOffset`）。

### 4.2 行/列两趟

```
行分析：x[y, 0..W-1]        → low_rows[y, 0..W/2-1]、high_rows[y, 0..W/2-1]
列分析：low_rows[:, x]      → LL[:, x]、LH[:, x]
        high_rows[:, x]     → HL[:, x]、HH[:, x]
```

合成严格逆序：先列合成得到 low_rows/high_rows，再行合成得到图像。三个 CPU
级别都用“按 j 升序、按 k 升序”的固定累加顺序；CUDA 合成改为按 taps 求解系数
下标（见 §4.5），累加顺序略有不同，跨后端因此用 `1e-4` 相对容差对拍。

### 4.3 边界与尺寸

- 周期边界：`(2k + j) mod N`，因此 H、W 必须为偶数，否则子带尺寸无定义，
  入口返回 `kInvalidArgument`。
- 单层、无 padding、无边界延拓；需要对称延拓时应由调用方先 padding。

### 4.4 精度与并行

- 所有累加在 double 中完成，最后窄化到 f32/f64 张量。
- 三个 CPU 级别共享同一套逐元素内核，只改变“哪一行/哪一列交给哪个线程”：

| 级别 | 实现文件 | 并行方式 |
| --- | --- | --- |
| naive | `wavelet_cpu_naive.cpp` | 顺序执行所有行、列 |
| OpenMP | `wavelet_cpu_omp.cpp` | `#pragma omp parallel for` 遍历行/列 |
| TBB | `wavelet_cpu_tbb.cpp` | `tbb::parallel_for` 遍历行/列 |
| CUDA | `wavelet_cuda.cu` | 每个 (行, 子带列) / (行, 输出列) 一个线程 |

因此三级 CPU 输出逐字节相同（单元测试 memcmp）。OpenCV 级别未注册，请求时
返回 `kNotImplemented`。

### 4.5 CUDA 实现

`wavelet_cuda.cu` 把“逐元素可并行”的特点用到极致：滤波器组没有跨样本依赖，
所以每个输出样本由一个线程独立算出，不需要共享内存或块内同步。

| 方向 | 内核顺序 |
| --- | --- |
| 正向 | `WaveletLoadPlaneKernel` → `WaveletAnalyzeRowsKernel` → `WaveletAnalyzeColsKernel`（直接写四个子带张量） |
| 逆向 | `WaveletLoadBandsKernel`（一次读四带）→ `WaveletSynthesizeColsKernel` → `WaveletSynthesizeRowsKernel` → `WaveletStorePlaneKernel` |

- 正/逆各 3–4 次内核发射；每个内核用一维扁平索引 + `ThreadStride()` 网格跨步，
  线程数按 `BlockCount` 封顶到 65535 个 block。
- 滤波器系数只有 2 或 4 个，随内核参数按值传入（`WaveletFilter`），不需要
  device 端常量表；系数来自与 CPU 共用的 `WaveletLowPass`/`WaveletHighPass`
  （在 `wavelet_internal.h` 中，`1/sqrt2`、`sqrt3` 用 double 计算）。
- 合成不再遍历所有 k：由 `2k + j ≡ y (mod N)` 反解，对每个 j 只需判断
  `(y - j)` 是否为偶数，因此每个输出仅 taps 次乘加，与 CPU 同阶。
- 暂存区为 `2*H*W` 个 f64（正向：平面 + 两条行缓冲；逆向：四条子带 + 两条行
  缓冲），经 `Context::AllocateTensor` 复用内存池；周期边界用一次减法代替取模。
- 所有发射进入同一 stream，末尾一次 `FinishLaunch` 同步；设备端 FMA 收缩与主机
  不同，跨后端对拍用 `1e-4` 相对容差。

## 5. 复杂度

单平面 H × W，taps 为滤波器长度（Haar 2、Db4 4）：

| 阶段 | 运算量 |
| --- | --- |
| 行分析 | H * (W/2) * taps * 2 次乘加 |
| 列分析 | (W/2) * (H/2) * taps * 2 * 2 次乘加 |
| 合计 | O(H*W*taps)，即线性复杂度 |

空间：每平面需要两块 (H*W/2) 的 double 行缓冲（low_rows / high_rows），
外加四带输出，合计 O(H*W)。Haar 最便宜（taps=2），Db4 约为其两倍。

## 6. 接口与示例

```cpp
#include "mycv/mycv.h"

mycv::OpContext ctx;

// 单层分解（默认 Haar；指定 kDaubechies4 使用 Db4）
auto bands = mycv::WaveletTransform(ctx, image, mycv::WaveletKind::kHaar);
// bands.ll / bands.lh / bands.hl / bands.hh 均为 H/2 x W/2

// 完美重建（kind 必须与分解时一致）
auto restored = mycv::InverseWaveletTransform(ctx, bands.value(),
                                              mycv::WaveletKind::kHaar);
```

参数校验：f32/f64（u8 返回 `kUnsupported`）；H、W 必须为偶数
（否则 `kInvalidArgument`）；rank 与 `layout` 必须匹配；逆变换要求四带
shape、dtype、backend、device 完全一致。

## 7. 测试

[`tests/test_wavelet.cpp`](tests/test_wavelet.cpp) 覆盖：

| 测试 | 内容 |
| --- | --- |
| `WaveletShapesAndBands` | NHWC/NCHW 下四带形状、dtype、连续性 |
| `WaveletHaarHandComputedTwoByTwo` | 2×2 输入手算 LL/LH/HL/HH |
| `WaveletConstantImageHasZeroDetails` | 常数图细节带为 0，LL = 2c（Haar 与 Db4） |
| `WaveletRoundTripIsIdentity` | Haar/Db4 × f32/f64 的完美重建（1e-6 / 1e-12） |
| `WaveletConservesEnergy` | 四带平方和等于输入平方和（正交性） |
| `WaveletCpuLevelsAreBitExact` | naive / OpenMP / TBB 逐字节相同 |
| `WaveletRejectsInvalidParameters` | 奇数尺寸、u8、rank5、空 context、四带不匹配 |
| `WaveletOpenCvIsNotImplemented` | OpenCV 后端返回 `kNotImplemented`（按宏启用） |
| `WaveletCudaMatchesCpu` | Haar/Db4 四带逐带与 CPU 对拍（`1e-4`） |
| `WaveletCudaRoundTripIsIdentity` | CUDA 上 Haar/Db4 正逆往返回到原图（`1e-4`） |

CUDA 用例在 `MYCV_WITH_CUDA` 且设备可用时执行；设备不可用时打印 SKIP 而不是失败。

三棵构建树的验收结果：

```
build-default        11/11
build-release-full   13/13
build-asan           12/12
```

## 8. 基准

[`benchmark/wavelet_benchmark.cpp`](benchmark/wavelet_benchmark.cpp) 输出 CSV：

```
op,backend,level,dtype,wavelet,direction,in_h,in_w,c,ms_per_call,mpix_per_s,speedup
```

speedup 相对同尺寸、同通道、同小波、同方向的 naive 行。测量方法：warmup 1 次、
每次 5 迭代；加速后端（CUDA）先于 CPU 行测量、跑两轮取较快值，CPU 三级按级别
分组测量（与 laplacian_pyramid 基准同一纪律，避免 CPU 满载压低 GPU 时钟）。

实测环境：Linux，24 逻辑 CPU，RTX 4060 Ti（CUDA arch 89），Release（-O2），f32。

| H×W | c | 小波 | 方向 | naive ms | OpenMP ms (×) | TBB ms (×) | CUDA ms (×) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512² | 1 | haar | forward | 3.79 | 8.64 (0.44) | 2.48 (1.53) | 0.06 (60.7) |
| 512² | 1 | haar | inverse | 4.76 | 3.58 (1.33) | 2.75 (1.73) | 0.07 (64.4) |
| 512² | 1 | db4 | forward | 4.06 | 4.06 (1.18) | 2.62 (1.44) | 0.07 (60.9) |
| 512² | 1 | db4 | inverse | 5.03 | 4.05 (1.43) | 2.72 (1.85) | 0.08 (65.5) |
| 512² | 3 | haar | forward | 9.09 | 11.97 (0.81) | 7.93 (1.24) | 0.16 (59.2) |
| 512² | 3 | haar | inverse | 12.19 | 14.16 (0.84) | 9.02 (1.34) | 0.19 (63.1) |
| 512² | 3 | db4 | forward | 12.88 | 11.99 (1.10) | 8.15 (1.60) | 0.17 (73.0) |
| 512² | 3 | db4 | inverse | 14.90 | 12.45 (1.16) | 9.14 (1.68) | 0.21 (70.5) |
| 1024² | 1 | haar | forward | 21.41 | 16.62 (1.29) | 13.72 (1.56) | 0.17 (126.8) |
| 1024² | 1 | haar | inverse | 26.70 | 20.00 (1.34) | 13.81 (1.93) | 0.21 (129.6) |
| 1024² | 1 | db4 | forward | 23.20 | 20.62 (1.13) | 13.09 (1.77) | 0.17 (136.5) |
| 1024² | 1 | db4 | inverse | 27.60 | 18.01 (1.53) | 13.67 (2.02) | 0.22 (127.8) |
| 1024² | 3 | haar | forward | 54.90 | 48.71 (1.13) | 30.33 (1.81) | 0.70 (79.0) |
| 1024² | 3 | haar | inverse | 66.79 | 44.55 (1.50) | 31.26 (2.14) | 0.88 (76.2) |
| 1024² | 3 | db4 | forward | 61.17 | 44.80 (1.37) | 30.20 (2.03) | 0.73 (84.3) |
| 1024² | 3 | db4 | inverse | 71.24 | 46.02 (1.55) | 32.68 (2.18) | 0.90 (78.9) |

结论：**CUDA 在全部 16 个配置上都是最优组合**（比 naive 快 59–137 倍，比 TBB
快约 40–80 倍）；数据量与并行度足够时 1024²×1 可到约 6.2 Gpx/s。CPU 侧 TBB
仍是最优级别（1.2–2.2 倍）。并行扩展比 FFT 弱，原因有两点：单层的运算量只有
O(H*W*taps)，每行工作量很小，线程调度占比高；分析/合成的列方向都是跨步访问
（CPU 侧是 stride = W/2 的行缓冲，GPU 侧是 NHWC 平面里通道步长为 c 的读写），
缓存与合并访问效率低。因此 1024²×3 时 CUDA 从约 6.2 Gpx/s 降到约 1.5 Gpx/s。
OpenMP 在 512² 的小任务上会低于 naive（0.44–0.84 倍），且共享机器上多线程行有
2 倍以内的波动，不是正确性问题（三级位精确测试仍通过）。

## 9. 已知限制与后续方向

- 只做单层；多级分解由调用方对 LL 迭代调用。
- 周期边界；不支持对称/零填充模式（可加参数）。
- 列方向跨步访问未做转置优化，是当前并行扩展的主要瓶颈（CPU 与 GPU 同样受影响）。
- 未做提升方案（可省一半乘法）与 SIMD；OpenCV 后端未提供。
- Db4 采用标准正交系数；与双正交 CDF 9/7 不兼容。
