# resize

`Result<Tensor> Resize(const OpContext& context, const Tensor& input,
int64_t output_height, int64_t output_width,
InterpolationMode interpolation = InterpolationMode::kBilinear,
ImageLayout layout = ImageLayout::kNHWC)`

双线性/最近邻图像缩放。rank2（HW）、rank3（NHWC={h,w,c} / NCHW={c,h,w}）与
rank4（NHWC / NCHW）均支持；布局是显式参数，不做隐式转置。

## 版本矩阵

| 版本 | 文件 | 关键点 |
| --- | --- | --- |
| naive | `resize_cpu_naive.cpp` | 双线性按 long double 累加（`value_cast.h`），u8 舍入 half-away-from-zero |
| OpenMP | `resize_cpu_omp.cpp` | 按输出行并行；每像素累加顺序与 naive 相同 → 位精确 |
| CUDA | `resize_cuda.cu` | grid-stride kernel，`Describe`/`ImageElementOffset` 感知 stride；只接受 `kNaive` 级别 |
| OpenCV | `resize_opencv.cpp` | `cv::resize`；跨后端对拍容差见下 |

SIMD（`kSimdGeneric`/`kSimdAvx2`）当前无注册实现：`SupportsKernelLevel` 返回
false，分发层自动跳过；后续补齐时走各算子文件夹的 OBJECT 库模式
（参见 gaussian_blur 的 `-mavx2` 方案）。

## 容差约定

- CPU 各级别（naive/OpenMP）：位精确（`memcmp`），对拍在 `tests/test_resize`
  与算子内测试中执行。
- 跨后端 f32/f64：相对容差 `1e-4`（`|a-b| <= tol*max(1,|a|)`）。
- u8 跨后端（OpenCV/CUDA）：绝对容差 ±1（定点/舍入实现差异）。

## 基准（build_release 全特性树）

```bash
./build_release/src/ops/resize/resize_benchmark   # stdout 为 CSV
```

CSV 列：`op,backend,level,dtype,mode,in_h,in_w,c,out_h,out_w,ms_per_call,gb_per_s,speedup`；
`speedup` 相对同一 (mode, dtype) 的 CPU naive 行。GB/s 按 输入+输出 字节数计。

### bilinear_down 1024×1024×3 → 512×768×3

| level | u8 ms/call | u8 GB/s | f32 ms/call | f32 GB/s | f64 ms/call | f64 GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| naive | 63.50 | 0.068 | 52.07 | 0.332 | 51.91 | 0.667 |
| openmp | 7.72 | 0.560 | 8.29 | 2.087 | 8.40 | 4.120 |

### 其余配置（ms/call）

| mode（输入→输出） | level | u8 | f32 | f64 |
| --- | --- | --- | --- | --- |
| bilinear_up 512×768×3→1024×1024×3 | naive | 168.58 | 138.09 | 137.69 |
| bilinear_up 512×768×3→1024×1024×3 | openmp | 18.35 | 19.82 | 19.64 |
| nearest_down 1024×1024×3→512×768×3 | naive | 23.55 | 14.36 | 14.20 |
| nearest_down 1024×1024×3→512×768×3 | openmp | 3.31 | 2.93 | 2.78 |

OpenCV / CUDA 行只在对应后端构建树输出；`build` 为 Debug，其数值不能与上表比较，
需另建 Release + 加速后端树复测后登记。
