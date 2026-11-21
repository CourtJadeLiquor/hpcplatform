# gaussian_blur 算子：可分离高斯平滑（v2 重点算子）

> 本文件夹是**完备解决方案**（[架构设计 §6.1](../../../docs/architecture.md#61-算子文件夹契约新增)）：入口、全部版本实现、构建脚本、单元测试、基准、本文档。
> 公共 API：[`include/mycv/ops/gaussian_blur.h`](../../../include/mycv/ops/gaussian_blur.h)。

## 语义

- 可分离两趟实现：水平方向 → 中间缓冲（f64，线性索引）→ 垂直方向 → 输出。
- `kernel_size` 必须为奇数且 ≥ 3；`sigma` 必须有限且 > 0；违例返回 `kInvalidArgument`。
- 边界模式 **REPLICATE**（边缘复制），与 `cv::BORDER_REPLICATE` 对齐——OpenCV 后端因此可直接对拍。
- dtype：u8 / f32 / f64；输出形状与输入相同。
- **u8 舍入规则**：两趟均以 double 累加，最终 `std::round`（half away from zero）回舍。系数恒正、和为 1，累加值非负且 ≤ 255，无需饱和。
- layout：HW / HWC / CHW / NHWC / NCHW 全部支持（OpenCV 仅 HW/HWC/NHWC，见 `RejectNchw`）。

## 版本矩阵

| 后端 | CpuKernelLevel | 实现文件 | 对拍方式 |
| --- | --- | --- | --- |
| native_cpu | kNaive | `gaussian_blur_cpu_naive.cpp`（真值基准） | — |
| native_cpu | kOpenMP | `gaussian_blur_cpu_omp.cpp`（行级并行，像素内顺序不变） | 与 naive `memcmp` 位精确 |
| native_cpu | kSimdGeneric | `gaussian_blur_cpu_simd.cpp`（类型化加载 + `MYCV_CPU_IVDEP`） | 与 naive `memcmp` 位精确 |
| native_cpu | kSimdAvx2 | `gaussian_blur_cpu_avx2.cpp`（4 通道块，c≥4 且 NHWC/HWC；其余回退标量骨架） | 与 naive `memcmp` 位精确 |
| native_cpu | kTbb | `gaussian_blur_cpu_tbb.cpp`（像素级并行） | 与 naive `memcmp` 位精确 |
| CUDA | kNaive | `gaussian_blur_cuda.cu`（两趟 kernel，中间缓冲 f64 Tensor） | 与 CPU 容差 1e-4 相对误差 |
| OpenCV | kNaive | `gaussian_blur_opencv.cpp`（`cv::GaussianBlur`，`BORDER_REPLICATE`） | 与 CPU 容差：f32/f64 1e-4 相对；u8 ±1 绝对 |

位精确纪律：所有 CPU 级别按 kernel 系数序（`i = 0..k-1`，即偏移 `-r..+r` 升序）用 double 单累加器累加；AVX2 用 mul+add（**不用 FMA**），每条 lane 与标量舍入一致。系数由 [`BuildGaussianKernel`](gaussian_blur_internal.h) 统一生成（归一化求和顺序同样固定），任何级别都不允许换序、换精度、换舍入——想放宽，先走评审（[架构设计 §6.5](../../../docs/architecture.md)），不要在测试里私自改容差。

SIMD/AVX2 级拒绝跨步（非连续）输入（`kUnsupported`，沿用 v1 元素级算子惯例），typed 加载假定连续内存。

## 测试与基准

```bash
# 全部 CPU 级别位精确 + 校验规则 + 布局一致性 + 常数图不变性
ctest --test-dir <build> -R test_gaussian_blur --output-on-failure

# 跨后端（full 配置才有）
ctest --test-dir build -R test_gaussian_blur --output-on-failure

# 基准
./build/src/ops/gaussian_blur/gaussian_blur_benchmark
```

## Profiling 数据

基准 stdout 为 CSV（列：`op,backend,level,dtype,kernel,sigma,in_h,in_w,c,ms_per_call,gb_per_s,speedup`，
`speedup` 相对同配置 CPU naive）：

```bash
./build-default/src/ops/gaussian_blur/gaussian_blur_benchmark
```

default Release 树，512×512，k=7，sigma=2.0（2026-11-21）：

| 配置 | level | u8 ms/call | u8 GB/s | f32 ms/call | f32 GB/s |
| --- | --- | --- | --- | --- | --- |
| c=3（AVX2 回退） | naive | 38.44 | 0.205 | 34.35 | 0.366 |
| c=3 | openmp | 4.44 | 1.770 | 5.05 | 2.492 |
| c=3 | simd_generic | 20.19 | 0.390 | 17.09 | 0.736 |
| c=3 | simd_avx2 | 38.19 | 0.206 | 34.82 | 0.361 |
| c=4（AVX2 生效） | naive | 51.10 | 0.205 | 46.40 | 0.362 |
| c=4 | openmp | 6.14 | 1.707 | 5.53 | 3.036 |
| c=4 | simd_generic | 26.74 | 0.392 | 22.91 | 0.732 |
| c=4 | simd_avx2 | 6.04 | 1.735 | 3.76 | 4.468 |

注：c=3 时 AVX2 的 4 通道块按设计不命中、回退标量骨架，故与 naive 同速；
c=4 的 NHWC 输入下 AVX2 相对 naive 为 u8 8.5×、f32 12.4×。
TBB/CUDA/OpenCV 行需在含对应后端的 Release 树复测后登记（`build` 为 Debug，数值不可比）。
需要更细粒度剖析时使用 perf / Nsight；仓库内不再保留无调用方的 profiling 宏层。
