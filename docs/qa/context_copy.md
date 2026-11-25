# 数据搬运：Copy / Upload / Download 与 From / Into

> 相关代码：[transfer.h](../../include/hpcplatform/core/transfer.h)、[copy.cpp](../../src/core/copy.cpp)、[conversion.h](../../include/hpcplatform/interop/conversion.h)、[opencv_convert.cpp](../../src/backends/opencv/opencv_convert.cpp)
> 配套样例：[samples/07_data_transfer.cpp](../../samples/07_data_transfer.cpp)（每一步都打印用了哪个入口）、[03](../../samples/03_opencv_backend.cpp)、[04](../../samples/04_cuda_pipeline.cpp)、[06](../../samples/06_image_cuda_pipeline.cpp)

---

## 0. 四层结构：只有一个实现，上面是两种"说法"

```
L3 适配层   Tensor::From<T>(外部对象, ctx)     第三方类型 -> Tensor，内含 staging
            Tensor::Into<T>(tensor)            Tensor -> 第三方类型，内含 staging
L2 语义层   Upload(host_src, &device_dst)      方向已知时的自文档化入口
            Download(device_src, &host_dst)    = Copy + 两个内存域断言
L2 原语层   Copy(src, &dst)                    唯一的搬运实现：任意方向、按 stride 搬逻辑元素
L1 内部     CopyTensorData(backend, dst, src)  只做校验 + 执行者选路 + 连续/非连续分发，不对外
L0 字节层   Backend::CopyBuffer(连续字节区间)   每后端实现；碰不到的内存返回 kUnsupported
            Backend::CopyStrided(任意 stride)   每后端决策搬运策略；CUDA 单 kernel launch，host 逐元素循环

（另：View / Slice / Permute / Reshape 只改元数据，不搬运任何字节——它们不是搬运 API。）
```

关键事实：**`Upload` 和 `Download` 不实现任何新能力**，它们只是给 `Copy` 加上"源必须是 host / 目标必须是 device"这类断言。所以不存在"用哪个更快"的问题，只存在"用哪个更能表达意图"的问题。

---

## 1. 唯一的选择规则：方向在调用点是否已知

| 场景 | 入口 | 理由 |
| --- | --- | --- |
| 第三方类型 ↔ Tensor（cv::Mat 等） | `From` / `Into` | 适配器自己负责 H2D/D2H，调用方不写传输代码 |
| 显式 H2D，目标已预分配 | `Upload` | 方向固定，断言能在早期抓住"忘了分配 device 张量"这类错误 |
| 显式 D2H，目标已预分配 | `Download` | 同上 |
| host ↔ host（可跨后端） | `Copy` | 没有方向可言 |
| device ↔ device（同一设备） | `Copy` | 没有方向可言 |
| 方向由数据决定（流水线、模板代码） | `Copy` | 传 device 张量就是 H2D，传 host 就是 host 拷贝，无需分支 |
| 同后端深拷贝一份 | `Tensor::Clone()` | 连目标分配都省了，等价于"分配 + Copy" |

一句话：**方向已知就写方向（Upload/Download），方向未知就用 Copy；拿到的是外部类型就用 From/Into。**

---

## 2. 角色分工

### 2.1 使用者（写业务流程）

```cpp
// 推荐：外部数据进来，适配器负责上卡
auto device = Tensor::From(image, *cuda);        // cv::Mat -> device 张量
auto result = Resize(ops, device.value(), h, w); // 算子全程在 GPU
auto out = Tensor::Into<cv::Mat>(result.value());// 自动 D2H

// 已有 host 张量、要显式上卡
auto device_tensor = Tensor::Empty(*cuda, host.shape(), host.dtype()).value();
hpcplatform_RETURN_IF_ERROR(Upload(host, &device_tensor));

// 方向不固定的通用工具函数
hpcplatform_RETURN_IF_ERROR(Copy(src, dst));      // 传什么域就搬什么域
```

职责边界：**由使用者决定数据放在哪个域**（选 CPU Context 还是 CUDA Context，选 `From(mat, *cpu)` 还是 `From(mat, *cuda)`）。

### 2.2 算子开发者（实现 `Resize` / `GaussianBlur` 这类算子）

**算子内部不调用 `Copy` / `Upload` / `Download`。**

- 输入张量在进入算子前已经在执行后端的内存域内（调用方用 `From` 或 `Upload` 放好）；算子直接读 `Data<T>()` 或交给 kernel。
- 输出用 `AllocateOutput(context, shape, dtype)`（内部就是 `Tensor::Empty`）分配，kernel 直接写入，不需要"先算到临时区再拷出来"。
- 需要把非连续视图落地时用 `Tensor::Clone()`，而不是自己拼 `Copy`。
- 需要跨界（例如 CPU 算子想读 GPU 输入）是**设计错误**，应该在调用方解决，不要在算子里偷搬。

判据：如果算子里出现了 `Upload` / `Download`，说明域的选择漏到了算子层，这是分层错误。

### 2.3 后端开发者（新增 CUDA / NPU / 新加速器）

只需正确实现 `Backend::CopyBuffer(dst_buffer, dst_offset, src_buffer, src_offset, bytes, stream)`（连续区间）；非连续拷贝由 `Backend::CopyStrided(dst, dst_strides, src, src_strides, shape, element_size, stream)` 承接，host 后端可直接继承默认的逐元素循环实现：

- 支持的方向自己实现；碰不到的内存（例如纯 host 后端遇到 device 源、CUDA 遇到跨设备源）**必须返回 `kUnsupported`**，禁止在内部用 host 中转"帮忙"——那会引入隐藏拷贝，破坏性能预期。
- `Copy` / `Upload` / `Download` / `From` / `Into` 会自动复用你的实现，不需要为上层的每种入口各写一份。
- 能否拷贝（可达性）、怎么拷贝（策略）都由后端决策；上层 `CopyTensorData` 只做校验和路由，不做任何能力判断。
- 跨设备（peer-to-peer）当前明确 `kUnsupported`，调用方需要时自己分两段拷。

---

## 3. 契约细节

### 3.1 统一签名

三个函数的参数顺序一致：**源在前、目标在后，目标用指针写出**。

```cpp
Copy(const Tensor& src,     Tensor* dst,     Stream* stream = nullptr);
Upload(const Tensor& host_src, Tensor* device_dst, Stream* stream = nullptr);
Download(const Tensor& device_src, Tensor* host_dst, Stream* stream = nullptr);
```

配套项目编码规范：不可变参数在前并带 `const`，可变输出在后。

### 3.2 执行者由张量所有权推断

调用方**永远不需要**传 Context：

- `Copy` / `Upload` / `Download`：由**目标**张量的 owning backend 驱动；若源是 device 张量，则由源所在设备后端执行（D2H）。
- `From`：由传入的 Context 决定目标域。
- `Into`：由张量自身所在的域决定。

因此"用错 Context"这类错误在类型层面就不存在。

### 3.3 同步契约：返回即可用

三者返回前**同步完成**（CUDA 路径内部 `cudaMemcpyAsync` + `cudaStreamSynchronize`）：

- 返回后 `dst` 立即可读，`src` 可以立即析构；
- 代价是没有异步流水；未来若加异步，会是一个**新的显式入口**，不会悄悄改变现有语义。

### 3.4 搬的是逻辑元素，不是裸字节

- 源可以是 `Slice` / `Permute` 后的非连续视图：按源 stride 逐元素读、按目标 stride 逐元素写，空隙里的垃圾不会被带过去。
- 要求 shape / dtype / 元素数一致；**不做 dtype 转换、不做 layout 转换**（NCHW↔NHWC 需要显式 `Permute` + `Clone`）。
- 连续 host↔host 走单次 `memcpy`；连续 host↔device 走单次 `CopyBuffer`。
- 非连续视图交给 `Backend::CopyStrided`：CUDA 后端单次 strided-copy kernel launch（任意 stride、任意 rank，无逐元素传输）；host 后端继承默认的逐元素循环。stride 感知是后端能力，上层不做策略决策。

---

## 4. 设计自评

| 维度 | 结论 |
| --- | --- |
| 分层 | 清晰：字节层 / 张量层 / 语义层 / 适配层各司其职，上层只加约束不加机制 |
| 是否有两个真源 | 否：`Copy` 是唯一实现，`Upload`/`Download` 是纯断言包装 |
| 心智模型 | 单一轴：**方向是否已知**。已知写方向，未知用 Copy，外部类型用适配器 |
| 参数一致性 | 三个函数统一 `(源, 目标指针)`，与项目"输出参数在后"规范一致 |
| 可扩展性 | 新后端实现 `CopyBuffer` + `CopyStrided`（host 后端可直接继承默认逐元素实现）；新第三方类型只加 `TensorConvert<T>` 特化，都不改传输 API |
| 已知取舍 | 全部同步，无异步流水；跨设备 P2P 未实现（后端显式 `kUnsupported`，不偷偷经 host）；非连续拷贝的策略由各后端自选（CUDA 单 kernel，host 逐元素循环） |

---

## 5. 常见误用

| 误用 | 后果 | 正确做法 |
| --- | --- | --- |
| 在算子里调用 `Upload` / `Download` | 域选择漏到算子层，算子被绑死在特定部署形态 | 由调用方决定域，算子里用 `Clone` 落地视图 |
| 期望 `Copy` 自动转换 dtype | 直接报错（类型不匹配） | 输出张量分配成目标 dtype，或先走 dtype 转换算子 |
| 期望 `Copy` 自动换 layout | 不换 | `Permute` 后用 `Clone` 落地 |
| 对非连续 device 张量直接 `Copy` / `Into` | 正常支持：CUDA 单 kernel launch 完成，无需先落地 | 仅跨设备 P2P 会 `kUnsupported`，需分两段拷 |
| 手工 host 中转做 D2D | 双倍流量 + 隐藏延迟 | 直接用 `Copy(device_src, &device_dst)` |
| 传 `nullptr` 目标 | 返回 `kInvalidArgument` | 先 `Tensor::Empty` |

---

## 6. 一句话总结

> 字节层只有 `CopyBuffer`（连续）和 `CopyStrided`（任意 stride），张量层只有 `Copy`；`Upload`/`Download` 是带方向断言的 `Copy`，`From`/`Into` 是把 staging 包起来的类型适配器。选型只有一个问题——方向是否已知——而算子内部永远不做搬运。
