# HPCPlatform

> **⚠️ 个人项目声明**：本项目是**个人兴趣使然的学习与研究项目**，用于探索高性能算子库的架构设计与工程实践，源于上一份工作中的某个需求，当时只做到0.1.0，没有深入，导致后面一个面试没过...。它**未经过生产环境验证，不建议直接用于生产**。如需在生产中使用，请自行完成安全审计、稳定性验证与长期维护评估。

HPCPlatform 是一个面向高性能算子学习、研究与开发的 C++17 平台。项目以图像与变换类算子为研究对象，提供统一的 Tensor/Context 抽象、可插拔后端、CPU 多版本优化、CUDA 加速、配置驱动调用，以及与算子一一对应的测试和基准工具。

平台强调**显式选择、结果可复现、算子自治**：调用方明确指定后端、内核级别和数据布局；后端不可用或未实现时返回错误，不做静默降级；新增算子以扩展自治目录为主，避免修改无关模块。

<br />

um改为cuda统一内存，关注kernel而不是cuda api，但是其在 WSL2 缺少 HMM 内核支持的配置，因此无法按需页迁移加速。如果仅限于研究kernel的话，在visual studio 2022/2026上配合统一内存接口是最佳的

## 特性

- **统一数据与执行抽象**：通过 `Tensor`、`Context`、`Stream` 和 `OpContext` 管理数据、设备、内存与执行环境。
- **CPU 多版本优化**：支持 naive、OpenMP、oneTBB、可移植 SIMD 与 AVX2 等实现路径；具体支持矩阵以各算子文档为准。
- **GPU 与 OpenCV 后端**：提供 CUDA 内核，以及部分算子的 OpenCV 4 实现，便于跨后端对拍和性能比较。
- **算子自治与注册分发**：每个算子独立维护公共入口、各版本实现、测试、基准和文档；通过 `(OpKind, BackendType, CpuKernelLevel)` 注册表分发。
- **YAML 配置驱动**：从 YAML 创建已绑定参数和执行环境的算子对象，可修改配置或环境变量切换后端，无需改动业务代码。
- **明确的内存语义**：支持 RAII、零拷贝视图、外部内存包装、跨后端拷贝和内存统计。
- **可验证、可比较**：使用 `Status`/`Result<T>` 传递错误，不依赖异常；提供 GoogleTest/CTest 测试、CPU 位精确对拍和独立基准程序。

> YAML 加载器只负责创建算子对象，不构建计算图；算子的执行顺序和数据流由调用方编排。某个后端或优化级别未实现时返回 `kNotImplemented`，不会自动改用其他实现。

## 当前算子

| 算子                  | 主要能力              | 说明文档                                                      |
| ------------------- | ----------------- | --------------------------------------------------------- |
| `resize`            | 最近邻、双线性缩放         | [resize](src/ops/resize/README.md)                        |
| `gaussian_blur`     | 可分离高斯平滑           | [gaussian\_blur](src/ops/gaussian_blur/README.md)         |
| `laplacian_pyramid` | 拉普拉斯金字塔分解与重建      | [laplacian\_pyramid](src/ops/laplacian_pyramid/README.md) |
| `fourier_transform` | 二维 DFT/FFT 及逆变换   | [fourier](src/ops/fourier/README.md)                      |
| `wavelet_transform` | Haar、Db4 二维小波及逆变换 | [wavelet](src/ops/wavelet/README.md)                      |

## 项目环境

| 项目      | 要求或当前约定                                          |
| ------- | ------------------------------------------------ |
| 操作系统    | 主要开发与验证环境为 Linux                                 |
| C++ 编译器 | 当前记录使用 Clang 21.1.8；GNU 编译器尚未验证                  |
| 语言标准    | C++17；CUDA 源文件同样使用 C++17                         |
| 构建系统    | CMake 3.20 或更高版本                                 |
| OpenCV  | OpenCV 4，使用 `core` 与 `imgproc` 组件                |
| CUDA    | 需要兼容的 NVIDIA 驱动、CUDA Toolkit 与 GPU；默认架构为 `sm_89` |
| 测试      | GoogleTest + CTest                               |

非 `sm_89` GPU 应在配置时显式设置 `-DCMAKE_CUDA_ARCHITECTURES=<架构号>`。没有 NVIDIA GPU 的开发机可关闭 CUDA 后端，仍可研究和运行 CPU/OpenCV 路径。

## 依赖与安装

所有可选后端和特性默认开启；开启的依赖由 CMake 以 `REQUIRED` 方式查找，缺失时配置会直接失败。

| 依赖           | 用途                       | 对应 CMake 开关（默认）                |
| ------------ | ------------------------ | ------------------------------ |
| OpenCV 4     | OpenCV 后端及 `cv::Mat` 互操作 | `hpcplatform_WITH_OPENCV=ON`   |
| CUDA Toolkit | CUDA 后端与设备内核             | `hpcplatform_WITH_CUDA=ON`     |
| yaml-cpp     | YAML 配置解析和算子对象加载         | `hpcplatform_WITH_YAML=ON`     |
| oneTBB       | TBB CPU 并行实现             | `hpcplatform_ENABLE_TBB=ON`    |
| OpenMP       | OpenMP CPU 并行实现          | `hpcplatform_ENABLE_OPENMP=ON` |
| 编译器 SIMD 能力  | 可移植 SIMD、AVX2 等内核        | `hpcplatform_ENABLE_SIMD=ON`   |
| GoogleTest   | 单元测试；仅构建测试时需要            | `hpcplatform_BUILD_TESTS=ON`   |

Ubuntu/Debian 可安装基础依赖：

```bash
sudo apt update
sudo apt install -y \
  build-essential \
  cmake \
  clang \
  libopencv-dev \
  libyaml-cpp-dev \
  libtbb-dev \
  libomp-dev \
  libgtest-dev
```

`clang` 软件包版本由发行版决定；如需复现当前记录环境，请安装 Clang 21.1.8。SIMD 不需要单独安装软件包，但 AVX2 内核要求编译器和目标 CPU 支持 AVX2。

CUDA 需要额外安装 NVIDIA 驱动和 CUDA Toolkit。可使用发行版软件源中的工具包，或按照 [NVIDIA CUDA 下载页](https://developer.nvidia.com/cuda-downloads) 的说明安装：

```bash
# 可选：使用 Ubuntu 软件源中的 CUDA Toolkit；也可用 NVIDIA 官方源
sudo apt install -y nvidia-cuda-toolkit

# 安装后检查驱动与编译器
nvidia-smi
nvcc --version
```

## 构建、测试、安装与打包

### Debug

`build_debug` 目录名会自动推断为 Debug，默认开启 ASan/UBSan。

```bash
cmake -S . -B build_debug
cmake --build build_debug -j
ctest --test-dir build_debug --output-on-failure
```

### Release

`build_release` 目录名会自动推断为 Release，默认关闭 Sanitizer，用于性能测试和日常使用。

```bash
cmake -S . -B build_release
cmake --build build_release -j
ctest --test-dir build_release --output-on-failure
```

也可以显式指定构建类型和 CUDA 架构：

```bash
cmake -S . -B build_release \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build_release -j
```

### 裁剪可选特性

默认推荐全特性构建。依赖或硬件不可用时，可显式关闭对应功能：

```bash
# 无 NVIDIA GPU：保留其他功能，关闭 CUDA
cmake -S . -B build_release -Dhpcplatform_WITH_CUDA=OFF

# 不需要 YAML 配置驱动时
cmake -S . -B build_release -Dhpcplatform_WITH_YAML=OFF
```

其他开关包括 `hpcplatform_BUILD_SAMPLES`、`hpcplatform_BUILD_BENCHMARKS`、`hpcplatform_ENABLE_SANITIZERS`、`hpcplatform_WITH_OPENCV`、`hpcplatform_ENABLE_OPENMP`、`hpcplatform_ENABLE_TBB` 和 `hpcplatform_ENABLE_SIMD`。

### 常用测试与基准命令

```bash
# 只运行名称匹配的测试
ctest --test-dir build_debug -R test_resize --output-on-failure

# 重跑上次失败的测试
ctest --test-dir build_debug --rerun-failed --output-on-failure

# 基准程序不注册到 CTest，应使用 Release 构建运行
./build_release/src/ops/resize/resize_benchmark
./build_release/src/ops/gaussian_blur/gaussian_blur_benchmark
```

如果 IDE 注入的旧版 `libstdc++` 导致构建或测试出现符号错误，可分别使用 `env -u LD_LIBRARY_PATH cmake --build build_debug -j` 和 `env -u LD_LIBRARY_PATH ctest --test-dir build_debug --output-on-failure`。完整测试约定见[测试与验证](docs/qa/testing.md)。

### 安装与打包

```bash
# 安装静态库、公共头文件和 CMake targets；可替换为目标前缀
cmake --install build_release --prefix /your/install/prefix

# 在构建目录中生成二进制包，布局与 install() 规则一致
cd build_release
cpack -G TGZ
cpack -G DEB
```

## 如何使用

公共 API 统一从伞形头 `hpcplatform/hpcplatform.h` 引入，并链接 CMake 目标 `hpcplatform`。

### 方式一：直接调用 API

直接调用适合需要显式控制 `Context`、内核级别、内存与执行流程的场景。下面创建 CPU 输入，依次执行 resize 和 Gaussian blur：

```cpp
#include <algorithm>
#include <iostream>

#include "hpcplatform/hpcplatform.h"

int main() {
  using namespace hpcplatform;

  const auto context_result = Context::Create(BackendType::kNativeCpu);
  if (!context_result.ok()) {
    std::cerr << context_result.status().ToString() << '\n';
    return 1;
  }
  const auto context = context_result.value();

  const auto input_result =
      Tensor::Empty(*context, Shape{1, 4, 4, 3}, DataType::kFloat32);
  if (!input_result.ok()) {
    std::cerr << input_result.status().ToString() << '\n';
    return 1;
  }
  auto input = input_result.value();
  std::fill_n(input.Data<float>(), input.Numel(), 1.0F);

  OpContext op_context(*context);
  op_context.set_cpu_level(CpuKernelLevel::kNaive);

  const auto resized = Resize(op_context, input, 2, 2);
  if (!resized.ok()) {
    std::cerr << resized.status().ToString() << '\n';
    return 1;
  }

  const auto blurred = GaussianBlur(op_context, resized.value(), 3, 1.0);
  if (!blurred.ok()) {
    std::cerr << blurred.status().ToString() << '\n';
    return 1;
  }

  std::cout << blurred.value().shape().ToString() << '\n';
  return 0;
}
```

可直接运行仓库中的完整示例：

```bash
./build_release/samples/00_hello_tensor
./build_release/samples/01_memory_copy
./build_release/samples/02_ops_pipeline
./build_release/samples/03_opencv_backend  # 需要 OpenCV
./build_release/samples/04_cuda_pipeline   # 需要 CUDA 和可用 GPU
./build_release/samples/07_data_transfer   # 数据搬运 API 全览（自动跳过不可用后端）
```

### 方式二：通过 YAML 配置调用

先创建配置文件，例如 `operators.yaml`。`name` 是算子实例名，`op` 是已注册的算子类型，`params` 是该算子的参数：

```yaml
operators:
  - name: downscale
    op: resize
    backend: native_cpu
    cpu_level: naive
    device_index: 0
    params:
      output_height: 2
      output_width: 2
      interpolation: bilinear
      layout: nhwc

  - name: smooth
    op: gaussian_blur
    backend: native_cpu
    cpu_level: naive
    params:
      kernel_size: 3
      sigma: 1.0
      layout: nhwc
```

通过 `LoadOperators` 一次性加载配置，再用 `GetOperator<T>` 获取强类型对象。对象可重复调用，调用方自行组织执行顺序：

```cpp
#include <algorithm>
#include <iostream>
#include <utility>

#include "hpcplatform/hpcplatform.h"

int main() {
  using namespace hpcplatform;

  auto loaded = LoadOperators("operators.yaml");
  if (!loaded.ok()) {
    std::cerr << loaded.status().ToString() << '\n';
    return 1;
  }
  const OperatorMap operators = std::move(loaded.value());

  auto* const downscale = GetOperator<ResizeOperator>(operators, "downscale");
  auto* const smooth = GetOperator<GaussianBlurOperator>(operators, "smooth");
  if (downscale == nullptr || smooth == nullptr) {
    std::cerr << "Missing operator or operator type mismatch\n";
    return 1;
  }

  const auto input_result = Tensor::Empty(
      *downscale->context(), Shape{1, 4, 4, 3}, DataType::kFloat32);
  if (!input_result.ok()) {
    std::cerr << input_result.status().ToString() << '\n';
    return 1;
  }
  auto input = input_result.value();
  // The configuration above uses native_cpu, so host initialization is valid.
  std::fill_n(input.Data<float>(), input.Numel(), 1.0F);

  const auto resized = downscale->Run(input);
  if (!resized.ok()) {
    std::cerr << resized.status().ToString() << '\n';
    return 1;
  }
  const auto blurred = smooth->Run(resized.value());
  if (!blurred.ok()) {
    std::cerr << blurred.status().ToString() << '\n';
    return 1;
  }

  std::cout << blurred.value().shape().ToString() << '\n';
  return 0;
}
```

仓库已提供可直接运行的配置示例：

```bash
# 不传路径时使用示例源码中的内联 YAML
./build_release/samples/05_config_operator

# 从文件加载
./build_release/samples/05_config_operator samples/configs/operators.yaml

# 使用环境变量统一覆盖后端和 CPU 内核级别
hpcplatform_BACKEND=native_cpu \
hpcplatform_CPU_LEVEL=openmp \
  ./build_release/samples/05_config_operator samples/configs/operators.yaml

# 切换到 CUDA（需要 CUDA 构建和可用设备）
hpcplatform_BACKEND=cuda \
hpcplatform_DEVICE_INDEX=0 \
  ./build_release/samples/05_config_operator samples/configs/operators.yaml
```

支持的环境变量为 `hpcplatform_BACKEND`、`hpcplatform_CPU_LEVEL` 和 `hpcplatform_DEVICE_INDEX`。设置 `hpcplatform_BACKEND` 后，所有算子共享同一个 `Context`，YAML 中的 `backend`、`cpu_level`、`device_index` 会被忽略；请求的环境不可用时直接报错。字段约束、环境优先级、参数表和错误语义详见[配置驱动算子对象](docs/operator_config.md)。

## 从哪里开始

1. **理解数据模型**：运行并阅读 [00\_hello\_tensor.cpp](samples/00_hello_tensor.cpp)，了解 `Shape`、`Tensor`、`Context` 和 `Result<T>`。
2. **理解内存与拷贝**：阅读 [01\_memory\_copy.cpp](samples/01_memory_copy.cpp) 与 [07\_data\_transfer.cpp](samples/07_data_transfer.cpp)，再参考[内存生命周期](docs/qa/memory_lifecycle.md)和[数据搬运](docs/qa/context_copy.md)。
3. **运行第一条算子链路**：阅读 [02\_ops\_pipeline.cpp](samples/02_ops_pipeline.cpp)，掌握 `OpContext` 和直接 API 调用。
4. **比较后端实现**：按需阅读 [03\_opencv\_backend.cpp](samples/03_opencv_backend.cpp) 和 [04\_cuda\_pipeline.cpp](samples/04_cuda_pipeline.cpp)。
5. **学习配置驱动**：运行 [05\_config\_operator.cpp](samples/05_config_operator.cpp)，修改 [operators.yaml](samples/configs/operators.yaml) 中的参数、后端和内核级别。
6. **深入研究单个算子**：从上方“当前算子”表进入对应文档，查看算法原理、版本矩阵、正确性测试和基准。
7. **开发新算子或优化版本**：先读[架构设计](docs/architecture.md)中的“算子自治结构”和“扩展指南”，再复制最接近的算子目录作为参考；不要从修改核心框架开始。

## 项目结构

```text
.
├── include/                 # 对外公共 API
├── src/
│   ├── core/                # Tensor、内存、拷贝和内部后端契约
│   ├── framework/           # 注册表、描述符、布局、dtype 分发等工具
│   ├── backends/            # CPU、CUDA、OpenCV 等后端服务
│   ├── ops/                 # 自治算子目录：实现、测试、基准、文档
│   └── operator/            # YAML 配置解析与算子对象加载
├── samples/                 # 渐进式公共 API 示例与 YAML 配置
├── tests/                   # core、backend、operator、性能测试
├── docs/                    # 架构、配置、QA 和版本演进文档
└── CMakeLists.txt           # 构建、安装与 CPack 配置
```

## 项目文档

| 文档                                  | 内容                                 |
| ----------------------------------- | ---------------------------------- |
| [架构设计](docs/architecture.md)        | 分层、依赖规则、数据模型、注册分发、算子自治和扩展指南        |
| [配置驱动算子对象](docs/operator_config.md) | YAML 契约、参数表、环境变量、对象模型和错误语义         |
| [QA 与易错点索引](docs/qa/README.md)      | Tensor 视图、内存生命周期、分发、错误模型、CUDA 流等专题 |
| [测试与验证](docs/qa/testing.md)         | 构建树、GoogleTest/CTest、位精确与容差、基准程序   |
| 各算子 `README.md`                     | 算法原理、适用场景、实现版本、测试命令和性能数据           |

## License

本项目采用 [GNU General Public License v3.0](LICENSE)，SPDX 标识为 `GPL-3.0-only`。

选择 GPLv3 是为了保证本项目及其衍生作品持续开放：**基于本项目形成并对外分发的衍生作品，包括通过静态或动态链接本库形成的组合作品，必须以 GPL-3.0 开源并提供对应源代码，同时保留版权和许可声明。** 仅内部使用且不对外分发时，不触发源代码公开义务。完整法律条款以 [LICENSE](LICENSE) 为准。

## 架构评审

- [架构评审报告（2026-11-25）](docs/architecture_review_2026-11-25.md)：SOLID 五原则符合性、分层与依赖方向、限界上下文、适配度函数、风险登记与 CI 门禁的完整评审结论。
