# 测试与验证：怎么跑、怎么写、容差怎么定

> 相关代码：[`tests/test_utils.h`](../../tests/test_utils.h)、[`tests/CMakeLists.txt`](../../tests/CMakeLists.txt)、根 [`CMakeLists.txt`](../../CMakeLists.txt)、各算子文件夹内的 `CMakeLists.txt`。
> 测试框架是 **GoogleTest**，由 **CTest** 统一注册与调度；`gtest_main` 提供 `main()`，测试文件只写 `TEST(...)`。

---

## 1. 单一全特性构建树

所有 `hpcplatform_*` 开关默认 ON：OpenCV、CUDA、yaml-cpp（配置驱动算子）、TBB、OpenMP、SIMD、单测、样例与基准。**不再为每个可选后端维护独立构建树**——旧的 `default` / `full` / `asan` 三棵树验证矩阵已废弃。

构建类型由目录名推断，也不用手写 `-DCMAKE_BUILD_TYPE`：

| 目录 | 构建类型 | Sanitizer |
| --- | --- | --- |
| `build_debug` | Debug | 默认 ON（ASan + UBSan） |
| `build_release` | Release | 默认 OFF |

```bash
cmake -S . -B build_debug
cmake --build build_debug -j
ctest --test-dir build_debug --output-on-failure
```

Release 构建同理：

```bash
cmake -S . -B build_release
cmake --build build_release -j
ctest --test-dir build_release --output-on-failure
```

常用选项：

```bash
ctest --test-dir build_debug -R test_resize          # 只跑匹配的用例
ctest --test-dir build_debug --rerun-failed          # 重跑上次失败的用例
ctest --test-dir build_debug -N                      # 只列出用例
```

明确需要裁剪特性时仍可单独关掉开关（例如没有 CUDA 的机器 `-Dhpcplatform_WITH_CUDA=OFF`），但这属于例外，不再是默认验证路径。

### 1.1 本环境的已知坑

- **`env -u LD_LIBRARY_PATH`**：IDE 可能往 `LD_LIBRARY_PATH` 里塞一个旧版 `libstdc++`，导致测试运行时报符号错误。解法：跑 `ctest` 前加 `env -u LD_LIBRARY_PATH`。
- **CUDA 与 LeakSanitizer**：NVIDIA 驱动在 `cuDriverGetVersion()` 后会保留进程级状态，LSan 会把它当成泄漏。`hpcplatform_ENABLE_SANITIZERS` 且 `hpcplatform_WITH_CUDA` 时，`hpcplatform_add_test` 会自动给测试设置 `ASAN_OPTIONS=detect_leaks=0`；ASan/UBSan 的错误检测仍然开启。
- **没有 GPU / 内存不足**：CUDA 用例会打印 SKIP 并把用例判为通过，不会把整份 `ctest` 搞红，见第 3 节。

---

## 2. GoogleTest 约定

### 2.1 文件与用例

```cpp
#include "test_utils.h"   // 已包含 <gtest/gtest.h> 与 hpcplatform_EXPECT_OK

#include "hpcplatform/hpcplatform.h"

namespace {

using namespace hpcplatform;

TEST(hpcplatform, ShapeRankAndNumel) {
  const Shape shape{2, 3, 4};
  EXPECT_EQ(shape.Rank(), 3);
  EXPECT_EQ(shape.Numel(), static_cast<int64_t>(24));
}

}  // namespace
```

- 套件名统一用 `hpcplatform`，用例名用原测试函数名（PascalCase），例如 `TEST(hpcplatform, ResizeU8LevelsAreBitExact)`。
- 每个测试文件编成一个独立可执行文件，由 `gtest_main` 提供入口，**不再写 `main()`**。
- `EXPECT_*` 非致命；`ASSERT_*` 致命。hpcplatform 不用异常，`Result::value()` 在失败时返回默认构造值，所以**用 `.value()` 前先判 `ok()`**，拿不准时用 `ASSERT_TRUE(result.ok())` 直接短路。

### 2.2 Status 断言

`tests/test_utils.h` 只保留一个项目侧宏：

```cpp
hpcplatform_EXPECT_OK(expr);   // 等价于 EXPECT_TRUE(expr.ok())，失败时附带 Status 消息
```

其余断言全部使用 GoogleTest 原生宏（`EXPECT_TRUE/FALSE/EQ/NE/...`、`EXPECT_THAT` 等）。

### 2.3 新增测试

1. 新建 `tests/core/test_xxx.cpp`（核心层）、`tests/backends/test_xxx.cpp`（后端）或 `src/ops/<op>/tests/test_xxx.cpp`（算子内）。
2. 引入 `test_utils.h`，写若干 `TEST(hpcplatform, ...)`。
3. 登记构建：
   - 核心层：把文件名加进 [`tests/core/CMakeLists.txt`](../../tests/core/CMakeLists.txt) 的 `hpcplatform_CORE_TEST_SOURCES`；
   - 后端层：仿照 [`tests/backends/CMakeLists.txt`](../../tests/backends/CMakeLists.txt) 用 `if(hpcplatform_WITH_xxx)` 包住 `hpcplatform_add_test(...)`；
   - 算子内：在算子 `CMakeLists.txt` 的 `if(hpcplatform_BUILD_TESTS)` 里加一行 `hpcplatform_add_test(...)`。
4. `hpcplatform_add_test(target sources...)` 负责链接 `hpcplatform` 与 `GTest::gtest_main`、添加 include 路径并调用 `add_test()`。

---

## 3. 没有 GPU / 后端未编译：跳过

可选后端不可用时，测试必须自己跳过并返回成功：

```cpp
TEST(hpcplatform, CudaSomething) {
  auto ctx = GetCudaContext();
  if (!ctx) return;   // 辅助函数已打印 SKIP，用例仍算 PASS
  ...
}
```

新写的用例更推荐 GoogleTest 原生的跳过方式，CTest 会显示为 `Skipped`：

```cpp
TEST(hpcplatform, CudaSomething) {
  auto ctx = GetCudaContext();
  if (!ctx) GTEST_SKIP() << "CUDA is unavailable";
  ...
}
```

要点：

- `Context::Create` 失败时返回 `kUnsupported`（不抛异常），用 `ok()` 判断。
- 跳过要让人在日志里看得见，不要静默返回。
- 后端未编译进来时，对应测试目标本身就不构建（CMake 门控），而不是运行时跳过。

---

## 4. 容差约定：什么时候位精确，什么时候放容差

### 4.1 CPU 各版本之间：位精确（bit-exact）

同一算子的不同 CPU 实现（naive / OpenMP / TBB / SIMD / AVX2）必须逐字节相同，对拍直接 `memcmp`：

```cpp
EXPECT_EQ(std::memcmp(actual.data(), reference.data(), reference.ByteSize()), 0);
```

- 整数与浮点都要求位精确，这强制 SIMD/并行版本的计算顺序与 naive 一致。
- 想放宽？先按[架构设计 §6.2](../architecture.md) 的位精确纪律走评审决策，不能在测试里私自改容差。
- 现有例子：`ResizeU8LevelsAreBitExact`、`GaussianBlurCpuLevelsAreBitExact`、`FourierCpuLevelsAreBitExact`、`WaveletCpuLevelsAreBitExact`。

### 4.2 跨后端：有限容差

CPU ↔ CUDA / OpenCV 之间，因浮点运算顺序、硬件融合乘加等差异，不要求位精确。两个后端测试文件各自定义 `ExpectEquivalent(...)`：CUDA 见 [`test_cuda_backend.cpp`](../../tests/backends/test_cuda_backend.cpp)，OpenCV 见 [`test_opencv_backend.cpp`](../../tests/backends/test_opencv_backend.cpp)。

实际取值规则（新增同类测试照此对齐，不要自己发明新数值）：

| 场景 | 容差 | 说明 |
| --- | --- | --- |
| f32 / f64 跨后端 | `1e-4` 相对误差 | `ExpectEquivalent` 的默认值，resize/warp 也用默认 |
| u8 跨后端（OpenCV） | 允许 ±1；显式传 `0.0` 则位精确 | `allowed_error = tolerance == 0.0 ? 0 : 1` |
| u8 / 整数 跨后端（CUDA） | 位精确（`memcmp`） | CUDA 版 `ExpectEquivalent` 只对 f32/f64 走容差分支 |
| 原始字节拷贝往返 | 显式传 `0.0` → 位精确 | |
| f16 转换往返 | `0.001` | f16 精度有限，放宽到 1e-3 |

> 相对误差指 `|a-b| <= tol * max(1.0, |a|)`；`max(1.0, ...)` 让接近 0 的值退化为绝对误差，避免除以极小值放大误报。

### 4.3 不确定就从严

拿不准该用多少容差时，先按位精确写，跑不过再分析是否确属硬件差异。放松容差要在测试注释里写清理由，不能为了让测试变绿而调大。

---

## 5. 基准（benchmark）

`hpcplatform_BUILD_BENCHMARKS=ON`（默认开）时，各算子文件夹内的 `benchmark/` 会编出各自的基准可执行：

- `src/ops/resize/benchmark/` → `resize_benchmark`
- `src/ops/gaussian_blur/benchmark/` → `gaussian_blur_benchmark`
- `src/ops/laplacian_pyramid/benchmark/` → `laplacian_pyramid_benchmark`
- `src/ops/fourier/benchmark/` → `fourier_benchmark`
- `src/ops/wavelet/benchmark/` → `wavelet_benchmark`

基准是**手工性能工具，不注册进 CTest**，不影响 `ctest` 的通过判定；用 Release 树跑才有意义：

```bash
./build_release/src/ops/resize/resize_benchmark
./build_release/src/ops/gaussian_blur/gaussian_blur_benchmark
./build_release/src/ops/laplacian_pyramid/laplacian_pyramid_benchmark
./build_release/src/ops/fourier/fourier_benchmark
./build_release/src/ops/wavelet/wavelet_benchmark
```

基准的解读纪律（怎么对比才算公平、怎么避免 `-march=native` 污染结论）见[架构设计 §8.2](../architecture.md) 的剖析与基准约定。
