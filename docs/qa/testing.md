# 测试与验证：怎么跑、怎么写、容差怎么定

> 相关代码：[`tests/test_framework.h`](../../tests/test_framework.h)、[`tests/CMakeLists.txt`](../../tests/CMakeLists.txt)、根 [`CMakeLists.txt`](../../CMakeLists.txt)
> mycv 没有引入 GoogleTest，用的是一个零依赖的迷你测试框架——因为研究平台要能离线、在裸工具链上编出来。换框架时测试体不用改，见下文。

---

## 1. 三棵构建树：验证矩阵

mycv 的正确性要在**多种编译配置**下都成立，因为可选后端和 CPU 优化都是编译期开关（`#ifdef`）。一份代码可能在 `default` 配置下编不过、却在 `full` 下漏过去（或者反过来）。所以固定用三棵构建树交叉验证。

> 构建配置完全由根 [`CMakeLists.txt#L8-L16`](../../CMakeLists.txt#L8-L16) 的 `option()` 决定，下面的命令就是这些开关的组合。`CMAKE_BUILD_TYPE` 不指定时默认 `Release`（[`CMakeLists.txt#L18-L20`](../../CMakeLists.txt#L18-L20)）。

### 1.1 default：最小可移植配置

只开 CPU + OpenMP + 可移植 SIMD（AVX2 在编译器支持时自动开），**不依赖任何可选第三方**，验证核心层和算子骨架在最干净的环境下成立。

```bash
cmake -S . -B build-default
cmake --build build-default -j
ctest --test-dir build-default --output-on-failure
```

### 1.2 full：全开配置

打开 OpenCV、CUDA、TBB，验证所有后端和所有 CPU level 共存时不冲突、分发正确。

```bash
cmake -S . -B build \
  -DMYCV_WITH_OPENCV=ON \
  -DMYCV_WITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DMYCV_ENABLE_TBB=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

> `MYCV_WITH_CUDA=ON` 时**必须**显式给 `CMAKE_CUDA_ARCHITECTURES`，否则配置阶段直接 `FATAL_ERROR`（[`CMakeLists.txt#L27-L36`](../../CMakeLists.txt#L27-L36)）。这是故意的——不让你编出一个不知道跑在哪代 GPU 上的库。

### 1.3 asan：内存与 UB 检查

开 AddressSanitizer + UndefinedBehaviorSanitizer（[`CMakeLists.txt#L84-L90`](../../CMakeLists.txt#L84-L90)）。一般配 OpenCV 但不配 CUDA（CUDA runtime 与 ASan 同进程容易打架）。

```bash
cmake -S . -B build-asan -DMYCV_ENABLE_SANITIZERS=ON -DMYCV_WITH_OPENCV=ON
cmake --build build-asan -j
ASAN_OPTIONS=verify_asan_link_order=0 ctest --test-dir build-asan --output-on-failure
```

> 三棵树都通过才算"这改动没问题"。只跑 default 就合并，等于没测 OpenCV/CUDA/TBB 路径。

### 1.4 本环境的已知坑（遇到再看）

以下是当前开发环境（IDE 注入了环境变量）观察到的、与库本身无关的运行时坑：

- **`env -u LD_LIBRARY_PATH`**：IDE 可能往 `LD_LIBRARY_PATH` 里塞了一个旧版 `libstdc++`，导致运行测试时符号不对。表现是一运行就 `abort` 或找不到符号。解法：跑 `ctest` 前 `env -u LD_LIBRARY_PATH ctest ...`。
- **`ASAN_OPTIONS=verify_asan_link_order=0`**：ASan 启动时校验自身链接顺序，和预装的 `libstdc++` 冲突会直接报错退出。加上这个选项让它跳过该校验。

---

## 2. 测试框架：零依赖、非致命

框架在 [`tests/test_framework.h`](../../tests/test_framework.h)，只有几个宏：

| 宏 | 作用 | 定义位置 |
| --- | --- | --- |
| `MYCV_TEST(name) { ... }` | 定义一个测试用例，自动注册 | [test_framework.h#L82-L85](../../tests/test_framework.h#L82-L85) |
| `MYCV_EXPECT_TRUE(cond)` | 断言为真 | [#L87-L92](../../tests/test_framework.h#L87-L92) |
| `MYCV_EXPECT_FALSE(cond)` | 断言为假 | [#L94-L99](../../tests/test_framework.h#L94-L99) |
| `MYCV_EXPECT_EQ(a, b)` | 断言相等，**返回 bool** | [#L101-L102](../../tests/test_framework.h#L101-L102) |
| `MYCV_EXPECT_OK(expr)` | 断言 `Status` 为 OK | [#L104-L111](../../tests/test_framework.h#L104-L111) |
| `MYCV_TEST_MAIN()` | 放在文件末尾，生成 `main()` | [#L113-L114](../../tests/test_framework.h#L113-L114) |

### 2.1 最关键的语义：断言是"非致命"的

`MYCV_EXPECT_*` 失败时**只记录、不中断**当前测试函数（[`RunAll` 实现](../../tests/test_framework.h#L60-L78)：失败数累加，测试体继续跑完，最后统一判这个用例 PASS/FAIL）。

这和 GoogleTest 的 `EXPECT_*` 一样，但后果是：**失败后后面的代码还会继续跑**。如果你紧接着对一个失败的 `Result` 调 `.value()`，因为 mycv 不用异常，`.value()` 不会抛，只会返回一个默认构造的 `T`（见 [`error_model.md`](./error_model.md)）——然后你就拿着空对象往下跑，要么段错误，要么报一堆误导性的二次失败。

所以写法上必须手动短路：

```cpp
auto out = Add(ctx, a, b);
MYCV_EXPECT_TRUE(out.ok());
if (!out.ok()) return;          // ← 失败就赶紧返回，别往下走
// 现在才能安全用 out.value()
```

这个模式在现有测试里到处都是，例如 [`test_cuda_backend.cpp#L49`](../../tests/consistency/test_cuda_backend.cpp#L49) 和 [#L51-L54](../../tests/consistency/test_cuda_backend.cpp#L51-L54)。

> 记住：mycv 里"断言失败"≠"程序停止"。任何 `Result::value()` 之前都要先 `ok()`，测试里也不例外。

### 2.2 怎么加一个新测试

1. 新建 `tests/unit/test_xxx.cpp`（单元）或 `tests/consistency/test_xxx.cpp`（对拍）。
2. 文件头：
   ```cpp
   #include "test_framework.h"
   #include "mycv/mycv.h"
   using namespace mycv;
   ```
3. 写若干 `MYCV_TEST(名字) { ... }`。
4. **文件最后一行**写 `MYCV_TEST_MAIN();`——每个测试文件是独立可执行文件，都需要自己的 `main`。
5. 登记到 [`tests/CMakeLists.txt`](../../tests/CMakeLists.txt)：
   - 单元/基础对拍加进 `MYCV_TEST_SOURCES` 列表（[#L1-L18](../../tests/CMakeLists.txt#L1-L18)）；
   - 依赖可选后端的测试（CUDA/OpenCV）仿照 [#L20-L32](../../tests/CMakeLists.txt#L20-L32) 用 `if(MYCV_WITH_xxx)` 包住，这样没开该后端时不会编。

---

## 3. 没有 GPU / 后端没编进来：SKIP 模式

可选后端的测试不能因为机器没 GPU 就把整个 `ctest` 搞红。约定：**测试自己探测、自己跳过，返回 0**。

看 CUDA 测试的写法（[`test_cuda_backend.cpp#L20-L28`](../../tests/consistency/test_cuda_backend.cpp#L20-L28)）：

```cpp
std::shared_ptr<Context> GetCudaContext() {
  static auto result = Context::Create(BackendType::kCUDA);
  static const bool reported = [] {
    if (!result.ok())
      std::cout << "SKIP CUDA tests: " << result.status().ToString() << '\n';
    return true;
  }();
  (void)reported;
  return result.ok() ? result.value() : nullptr;
}
```

每个用例开头：

```cpp
MYCV_TEST(CudaSomething) {
  auto ctx = GetCudaContext();
  if (!ctx) return;          // ← 没 GPU 就打印 SKIP 并返回，用例仍算 PASS
  ...
}
```

要点：

- `Context::Create` 失败时返回 `kUnsupported`（不是抛异常），所以用 `ok()` 判断。
- 跳过要**打印一行 `SKIP ...`**，让人在日志里看得见"这不是没跑，是没条件跑"。
- 因为框架是非致命的，`return` 后当前失败数为 0，用例判 PASS——正是我们想要的"跳过即通过"。

---

## 4. 容差约定：什么时候位精确，什么时候放容差

这是最容易写出"看起来对其实放水"或"太严导致误报"的地方。mycv 的规则分两层：

### 4.1 CPU 各版本之间：位精确（bit-exact）

同一个算子的不同 CPU 实现（naive / OpenMP / TBB / SIMD / AVX2）之间必须**逐字节相同**。对拍直接 `memcmp`：

```cpp
MYCV_EXPECT_EQ(std::memcmp(added.value().data(), reference_add.data(),
                           reference_add.ByteSize()), 0);
```

见 [`test_cpu_kernels.cpp#L52-L53`](../../tests/consistency/test_cpu_kernels.cpp#L52-L53)。

- 整数：当然位精确。
- **浮点也要求位精确**——不是"误差很小就行"。这强制 SIMD/并行版本的计算顺序和 naive 完全一致。想放宽？先走 [`../architecture/P3.md`](../architecture/P3.md) 的决策流程，不能在测试里私自改容差。
- 对拍会遍历所有"当前构建支持的 level"（[`AvailableLevels`](../../tests/consistency/test_cpu_kernels.cpp#L22-L30)，靠 `SupportsKernelLevel` 探测），所以新增 level 只要注册对了就自动被覆盖。

### 4.2 跨后端：有限容差

CPU ↔ CUDA / OpenCV 之间，因为浮点运算顺序、硬件融合乘加等差异，不要求位精确。两个一致性测试文件各自定义了 `ExpectEquivalent(expected, actual, native, tolerance = 1e-4)`（[CUDA](../../tests/consistency/test_cuda_backend.cpp#L45-L46) / [OpenCV](../../tests/consistency/test_opencv_backend.cpp#L32-L33)），**几乎所有调用都不传第四个参数**，即默认 `1e-4` 相对误差——包括 resize / warp，并没有单独的插值容差。

实际取值规则（全部来自现有测试，新增同类测试照此对齐，不要自己发明新数值）：

| 场景 | 容差 | 依据 |
| --- | --- | --- |
| f32 / f64 跨后端 | `1e-4` 相对误差 | 两个 `ExpectEquivalent` 的默认值，绝大多数调用处（含 resize/warp）都用默认 |
| u8 跨后端（OpenCV） | 允许 ±1；若显式传 `0.0` 则位精确 | [`test_opencv_backend.cpp#L49-L53`](../../tests/consistency/test_opencv_backend.cpp#L49-L53) 的 `allowed_error = tolerance == 0.0 ? 0 : 1` |
| u8 / 整数 跨后端（CUDA） | 位精确（`memcmp`） | CUDA 版 `ExpectEquivalent` 无 u8 分支，非 f32/f64 走 `else` 做 `memcmp`，[`test_cuda_backend.cpp#L62-L64`](../../tests/consistency/test_cuda_backend.cpp#L62-L64) |
| 原始字节拷贝往返 | 显式传 `0.0` → 位精确 | [`test_cuda_backend.cpp#L435`](../../tests/consistency/test_cuda_backend.cpp#L435)、[`test_opencv_backend.cpp#L197`](../../tests/consistency/test_opencv_backend.cpp#L197) |
| f16 转换往返 | `0.001` | [`test_cuda_backend.cpp#L442`](../../tests/consistency/test_cuda_backend.cpp#L442)（f16 精度有限，放宽到 1e-3） |

> "相对误差"指 `|a-b| <= tol * max(1.0, |a|)`，见 [`test_cuda_backend.cpp#L57`](../../tests/consistency/test_cuda_backend.cpp#L57) 和 [`test_opencv_backend.cpp#L44`](../../tests/consistency/test_opencv_backend.cpp#L44)。`max(1.0, ...)` 让接近 0 的值退化为绝对误差，避免除以极小值放大误报。

### 4.3 不确定就从严

拿不准该用多少容差时，**先按位精确写**，跑不过再分析是不是真的硬件差异。放松容差要有理由（在测试注释里写清"因为 XX 导致末位不同"），不能为了让测试绿而调大。

---

## 5. 基准（benchmark）

`MYCV_BUILD_BENCHMARKS=ON`（默认开，[`CMakeLists.txt#L10`](../../CMakeLists.txt#L10)）时会编两个基准程序，定义在 [`benchmarks/CMakeLists.txt`](../../benchmarks/CMakeLists.txt)：

- `operator_benchmark`：单算子测速。
- `image_pipeline_benchmark`：图像流水线端到端测速。

跑完直接在构建目录执行即可：

```bash
./build/benchmarks/operator_benchmark
```

基准的解读纪律（怎么对比才算公平、怎么避免 `-march=native` 污染结论）见 [`../architecture/P3.md`](../architecture/P3.md)。
