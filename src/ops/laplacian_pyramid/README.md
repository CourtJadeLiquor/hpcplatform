# laplacian_pyramid

基于 `resize` 与 `gaussian_blur` 的**组合算子**：不引入新的像素级数学，而是逐级复用两个子算子的全部版本。
因此它的“优化版本”就是转发给子算子的 `CpuKernelLevel`，它的“执行后端”就是 `OpContext` 的后端。

> 排版约定：本文所有公式都用纯文本或代码块书写（如 `G[i+1]`、`ceil(H/2)`），不依赖 LaTeX/MathJax，
> 在任何 Markdown 预览器中都能直接显示。

---

## 1. 原理

### 1.1 为什么要金字塔

自然图像的信息分布在多个尺度上：纹理和噪声在高频，物体轮廓在中频，光照和整体结构在低频。
单一分辨率的卷积核只能覆盖一个尺度。**图像金字塔**通过对图像反复平滑、降采样，得到一组分辨率递减的图像：

```
G[0] = 输入图像                      尺寸 H0 x W0
G[1] = 降采样(平滑(G[0]))            尺寸 H1 x W1 ≈ H0/2 x W0/2
G[2] = 降采样(平滑(G[1]))            尺寸 H2 x W2 ≈ H1/2 x W1/2
...
G[n] = 最粗层                        尺寸 Hn x Wn
```

G[i] 称为**高斯金字塔**（Gaussian pyramid）的第 i 层。层数越多，能分析的尺度范围越宽。

### 1.2 拉普拉斯带通

高斯金字塔本身只保留低频。把相邻两层相减，就能提取“某一尺度上的细节”：

```
Up(G[i+1])  = 把 G[i+1] 放大回 G[i] 的尺寸
L[i]        = G[i] - Up(G[i+1])        （i = 0 .. levels-1）
```

- `Up(·)` 是上采样（expand）操作；
- `L[i]` 是第 i 层的**拉普拉斯带通图**（Laplacian band），正负值表示该尺度上的明暗变化；
- 最粗层 `G[levels]` 没有下一层，直接作为**低频残差**保留。

这样 `L[0], L[1], ..., L[levels-1], G[levels]` 构成一个**拉普拉斯金字塔**，把原图分解成
“细尺度细节 + 中尺度细节 + ... + 低频底座”。典型两层分解的尺度关系：

```
G[0]  H x W        ──► L[0] = G[0] - Up(G[1])     细节（最细）
  │ blur+down
  ▼
G[1]  H/2 x W/2    ──► L[1] = G[1] - Up(G[2])     细节（中）
  │ blur+down
  ▼
G[2]  H/4 x W/4    ──► 直接作为低频残差输出
```

### 1.3 可逆性与重建

由定义 `L[i] = G[i] - Up(G[i+1])` 可以直接得到：

```
G[i] = L[i] + Up(G[i+1])
```

因此只要使用**同一个** `Up(·)`，就能从金字塔逐级重建原始图像：

```
G[levels] = 最粗层
G[i]      = L[i] + Up(G[i+1])        （从粗到细）
G[0]      = 重建图像
```

本项目保证该关系**精确成立**（浮点舍入除外）：分解与重建共用 `Resize` 的 bilinear 上采样。
这也意味着分解阶段本身不丢信息——真正不可逆的只有“降采样时丢弃奇数行/列”这一步，
而它已经体现在下一层 `G[i+1]` 里。

### 1.4 与“高斯差分”的关系

对高斯核做差分（Difference of Gaussians, DoG）可以近似拉普拉斯算子：

```
DoG = G(σ1) - G(σ2) ≈ -(σ2 - σ1) * ∇²G
```

所以拉普拉斯金字塔的每一层在概念上接近该尺度的 LoG（Laplacian of Gaussian）响应，
但**本项目不是严格的 LoG**：上采样滤波器（bilinear）与分解用的高斯核不构成经典
Burt–Adelson 的分析/合成滤波器对，详情见 §4.5。

---

## 2. 适用场景

| 场景 | 如何使用金字塔 |
| --- | --- |
| 多曝光 / 多焦距图像融合 | 每张图分解后按清晰度/曝光权重逐带融合，再重建，避免单图过曝或失焦 |
| 多频段图像混合、接缝消除 | 把两张图分别分解，低频平滑过渡、高频保留各自纹理（multiband blending） |
| 图像压缩 / 渐进传输 | 低频层先传输，细节层按码率裁剪；解码端逐级重建 |
| 超分辨率 / 上采样重建 | 学习或构造各频带的高频补偿，再逐级重建出高分辨率图 |
| 光流 / 立体匹配的由粗到细求解 | 在最粗层估计大位移，逐级上采样作为下一层初值 |
| 纹理分析 / 图像合成 | 统计各频带能量，做纹理分类、风格迁移、材质合成 |
| 缺陷检测 / 异常检测 | 高频带能量异常、带通统计偏移往往对应划痕、污渍、局部缺陷 |
| 医学 / 遥感图像增强 | 对不同频带做不同增益（tone mapping、边缘增强）后再重建 |

**不适合**的场景：需要严格平移不变、方向选择性（steerable pyramid / 小波）、或严格正交分解的任务；
这类需求应使用对应变换，而不是普通的高斯/拉普拉斯金字塔。

---

## 3. 实现算法有哪几种

### 3.1 经典 Burt–Adelson（1983）

最常见的标准实现，使用固定的 5×5 二项式核：

```
w = [1, 4, 6, 4, 1] / 16        （可分离：先水平后垂直）
reduce(G) = 先卷积 w，再隔行隔列取样
expand(G) = 先零插值放大，再用 4*w 卷积
L = G - expand(reduce(G))
```

- 优点：分析/合成滤波器严格配对，重建误差可控，是后续大量金字塔工作的基准；
- 缺点：核与尺寸固定，边界处理、扩展核（4*w）需要单独实现。

### 3.2 OpenCV `pyrDown` / `pyrUp`

`cv::pyrDown` 把“5×5 高斯 + 隔行取样”合并成一次调用，`cv::pyrUp` 做“零插值 + 5×5×4 卷积”。
`cv::buildPyramid` 可一次构造高斯金字塔，拉普拉斯层则用两层相减得到。
边界默认 `BORDER_REFLECT_101`，与 Burt–Adelson 的常见变体接近。

### 3.3 DoG 近似

用两个不同 σ 的高斯模糊相减近似 LoG，常用于 SIFT 关键点检测、边缘增强。
它计算便宜，但**不是**真正的金字塔分解，重建能力弱，通常只取响应不取金字塔结构。

### 3.4 频域 / 小波方法

在 FFT 域做低通/带通滤波，或用离散小波变换（DWT）做多分辨率分解。
频域方法对整幅图做全局变换，内存与延迟较高；小波方向选择性更好，但不再是“高斯/拉普拉斯金字塔”。

### 3.5 本项目采用：可分离高斯 + nearest 抽取 + bilinear 上采样

实现位于 `laplacian_pyramid_host.cpp` / `laplacian_pyramid_cuda.cu`，每一级只做三件事：

```
G[i+1] = ResizeNearest( GaussianBlur(G[i], kernel_size, sigma),
                        ceil(H[i]/2), ceil(W[i]/2) )
Up(·)  = ResizeBilinear( G[i+1], H[i], W[i] )
L[i]   = G[i] - Up(G[i+1])
```

选择这种组合的原因：

1. **复用现有算子**：高斯模糊和缩放已经实现了 naive/OpenMP/TBB/OpenCV/CUDA 全部版本，
   金字塔只需组合，不必重复写像素级代码；
2. **参数开放**：核大小、σ、层数都可调，不局限于固定的 5×5 二项式核；
3. **多后端一致**：组合链让 CPU、OpenCV、CUDA 共用同一套语义，跨后端结果可用单元测试对拍；
4. **确定性**：同一后端、同一级别下所有中间结果逐位可复现。

与 §3.1/§3.2 的差异见 §4.5。

### 3.6 算法对比

| 方法 | 降采样 | 上采样 | 严格 LoG | 可逆性 | 本项目 |
| --- | --- | --- | --- | --- | --- |
| Burt–Adelson | 5×5 二项式核 + 取样 | 零插值 + 4w 卷积 | 近似 | 分析/合成核配对，重建稳定 | 否 |
| OpenCV pyrDown/pyrUp | 内置 5×5 高斯 + 取样 | 零插值 + 5×5×4 | 近似 | 配对 | 仅作为子算子（OpenCV 后端） |
| DoG | 两个 σ 模糊相减 | 无金字塔结构 | 近似 LoG | 弱 | 否 |
| FFT / 小波 | 频域低通 | 频域上采样 | 依变换而定 | 依变换而定 | 否 |
| **本项目** | **可分离高斯 + nearest 抽样** | **bilinear** | **否（见 §4.5）** | **同一 Up 下精确重建** | **是** |

---

## 4. 本项目实现细节

### 4.1 逐级公式

```
输入：I，层数 levels，核大小 k（奇数 >= 3），标准差 sigma > 0

G[0] = I
for i = 0 .. levels-1:
    G[i+1] = resize_nearest( gaussian_blur(G[i], k, sigma),
                             ceil(H[i] / 2), ceil(W[i] / 2) )
    L[i]   = G[i] - resize_bilinear( G[i+1], H[i], W[i] )
输出：[L[0], L[1], ..., L[levels-1], G[levels]]
```

- `gaussian_blur`：可分离卷积，核系数归一化，边界 **REPLICATE**（复制边缘像素）；
- `resize_nearest`：源坐标 `floor(y * H[i] / H[i+1])`。偶数尺寸时恰好是隔行/隔列取样；
- `resize_bilinear`：半像素坐标 `(y + 0.5) * H[i] / H[i+1] - 0.5`，越界钳制（等价 align_corners=false）；
- `ceil` 尺寸让奇数边长也能分解，但此时 nearest 的取样点不再是严格隔行（见 §4.5）。

### 4.2 返回值与尺寸

`BuildLaplacianPyramid` 返回 `levels + 1` 个张量：

| 下标 | 内容 | 尺寸 |
| --- | --- | --- |
| `0 .. levels-1` | 带通 `L[i]`，由细到粗 | 与 `G[i]` 相同 |
| `levels` | 最粗高斯残差 `G[levels]` | `ceil(H[levels-1]/2) x ceil(W[levels-1]/2)` |

所有层保持输入的 dtype、layout（NHWC/NCHW）和后端。

### 4.3 dtype 与参数约束

- **只接受 f32/f64**：拉普拉斯带是有符号差值，u8 无法表示负值，传入 u8 返回 `kUnsupported`；
- `levels >= 1`；每一级降采样的**输入**必须是 `>= 2x2`，否则返回 `kInvalidArgument`
  （最粗层本身可以小到 1x1）；
- `kernel_size` 为 `>= 3` 的奇数；`sigma` 必须有限且 `> 0`；
- 输入 rank 必须是 2/3/4（HW、HWC/CHW、NHWC/NCHW），由 `ParseImage` 校验。

### 4.4 后端与版本矩阵

| 后端 / 级别 | 实现文件 | 说明 |
| --- | --- | --- |
| CPU `kNaive` | `laplacian_pyramid_host.cpp` | 组合子算子 naive 内核 + host 逐元素相减 |
| CPU `kOpenMP` | 同上 | 级别透传给子算子 |
| CPU `kTbb` | 同上 | 级别透传给子算子 |
| CPU `kSimdGeneric` / `kSimdAvx2` | — | **未注册**：resize 无 SIMD 实现，组合链无法成立，查表即返回 `kUnsupported` |
| OpenCV | `laplacian_pyramid_host.cpp` | 子算子走 `cv::GaussianBlur` / `cv::resize`，相减在 host 内存 |
| CUDA | `laplacian_pyramid_cuda.cu` | 子算子走 CUDA 内核，相减为 device kernel |

组合算子的一个阶段只写两个张量：`output` 为 `L[i]`，`params->next_gaussian` 为 `G[i+1]`
（由内核 move 赋值），因此公共入口仍是“校验 → 查注册表 → 调内核”的固定骨架，只是逐级循环调用。

### 4.5 与经典实现的差异（重要）

本项目的“拉普拉斯金字塔”是一个**可逆带通分解**，但不是教科书意义上的 LoG 金字塔：

1. **上采样不是 matched filter**：经典 expand 使用 `4 * w`（w 为 5×5 二项式核），
   本实现用 bilinear。因此 `L[i]` 更接近“原图减去双线性放大的低频图”，
   而不是严格的高斯-拉普拉斯响应。
2. **重建仍然精确**：因为 `L[i]` 就是按同一个 `Up` 定义的差值，
   `G[i] = L[i] + Up(G[i+1])` 恒成立，逐级重建可无损恢复 `G[0]`（浮点舍入除外）。
3. **奇数尺寸**：`ceil(H/2)` 配合 `floor(y * H / H')` 取样，奇数边长时取样点不均匀，
   与 OpenCV `pyrDown` 的 `(H+1)/2` 尺寸一致，但滤波器不同。
4. **边界**：高斯用 REPLICATE，bilinear 用坐标钳制；OpenCV 后端子算子仍用 OpenCV 自己的边界规则，
   所以跨后端只有 `1e-4` 量级一致，而非逐位一致。

如果任务要求严格 LoG 响应或与 OpenCV 金字塔逐位一致，应直接使用 `cv::pyrDown`/`cv::pyrUp`，
或按 Burt–Adelson 的 matched filter 另写实现。

### 4.6 复杂度

每一级的高斯模糊是可分离卷积：`O(N[i] * k)`，降采样与上采样是 `O(N[i])`，相减是 `O(N[i])`。
总计算量约为 `N * (1 + 1/4 + 1/16 + ...) ≈ 4/3 N`（常数倍 k），层数增加不会改变量级。
内存需要同时保留输出金字塔，约 `4/3 N` 个元素（f64 时注意显存/内存占用）。

---

## 5. 接口示例

```cpp
#include "mycv/mycv.h"

mycv::OpContext ctx;  // context / stream / cpu_level
auto pyramid = mycv::BuildLaplacianPyramid(ctx, image, /*levels=*/3,
                                           /*kernel_size=*/5, /*sigma=*/1.2);
if (pyramid.ok()) {
  for (const mycv::Tensor& band : pyramid.value().levels) {
    // 前 levels 个是带通 L[i]，最后一个是低频残差 G[levels]
  }
}

// 重建（与分解共用同一上采样）：
// G[i] = L[i] + ResizeBilinear(G[i+1], H[i], W[i])
```

---

## 6. 测试

```bash
ctest --test-dir build-default -R test_laplacian_pyramid --output-on-failure
```

| 用例 | 覆盖点 |
| --- | --- |
| `ShapesAndLevelCount` | NHWC/NCHW、9x11→5x6→3x3→2x2、共 levels+1 个张量、dtype/后端/连续性 |
| `ConstantImageHasZeroBands` | 常数图所有带通 ≈ 0（1e-6），最粗层保持常数 |
| `MatchesReferenceComposition` | 用公共 `GaussianBlur`/`Resize` 独立重算，逐级 **memcmp 位精确** |
| `CpuLevelsAreBitExact` | naive vs OpenMP/TBB 逐级 memcmp；未注册级别按 `kUnsupported` 跳过 |
| `RejectsInvalidParameters` | levels=0、偶数核、sigma=0、u8、rank5、空 context、1x1 输入层数过深 |
| `OpenCvMatchesCpuFloat` | 逐级与 CPU 对拍，容差 `1e-4`（相对 `max(1,|a|)`） |
| `CudaMatchesCpuFloat` | 同上 |

default 树 5 个用例，full/ASAN 树 7 个（OpenCV/CUDA 用例真实执行）。

---

## 7. 基准与实验结果

```bash
./build-release-full/src/ops/laplacian_pyramid/laplacian_pyramid_benchmark
```

CSV 列：`op,backend,level,dtype,levels,kernel,sigma,in_h,in_w,c,ms_per_call,mpix_per_s,speedup`；
`speedup` 相对同配置的 CPU naive 行。加速后端行**先于** CPU 行测量，避免数秒 CPU 满载压低 GPU 时钟
导致末尾配置被系统性高估；CUDA 行跑两轮取较快值。

Release + OpenCV + CUDA + TBB 构建树，f32，k=5，sigma=1.2（2026-11-21）：

| 输入 / 层数 | naive | OpenMP | TBB | OpenCV | CUDA |
| --- | --- | --- | --- | --- | --- |
| 512×512×3, 2 | 85.17 ms | 22.19 ms (3.84×) | 16.06 ms (5.30×) | 8.65 ms (9.85×) | **0.79 ms (107.6×)** |
| 512×512×3, 3 | 89.32 ms | 24.45 ms (3.65×) | 16.51 ms (5.41×) | 8.95 ms (9.98×) | **0.91 ms (98.3×)** |
| 1024×1024×3, 2 | 342.14 ms | 73.11 ms (4.68×) | 56.97 ms (6.01×) | 32.33 ms (10.58×) | **2.70 ms (126.9×)** |
| 1024×1024×3, 3 | 359.12 ms | 78.36 ms (4.58×) | 60.50 ms (5.94×) | 34.03 ms (10.55×) | **2.89 ms (124.1×)** |

**最高性能组合：`cuda` 后端 + `kNaive` 级别**（全部四组配置，相对 CPU naive 约 98–127×）。
CPU 内部最优为 TBB（约 5.3–6.0×），OpenCV 稳定在 10× 左右；由于组合链包含 resize，
`kSimdGeneric`/`kSimdAvx2` 组合不可用。吞吐（`mpix_per_s`）在 512² 约 331 MPix/s、1024² 约 389 MPix/s（CUDA）。

> 绝对数值随机器/GPU 时钟变化；跨后端一致性由单元测试保证，表中只用于相对比较。

---

## 8. 已知限制

- 不支持 u8；需要 8 位结果时应先在外部转成有符号/浮点，或只使用带通图做分析、不做重建；
- 上采样为 bilinear，带通不是严格 LoG；对滤波器形状敏感的任务应换用 §3.1 的 matched filter；
- `kSimdGeneric` / `kSimdAvx2` 不可用（resize 尚无 SIMD 内核）；
- 奇数尺寸的取样点不均匀；极小图受限：1x1 无法分解，2x2 只能分解一层（得到 1x1）；
- 跨后端只有数值级一致（`1e-4`），不是逐位一致。
