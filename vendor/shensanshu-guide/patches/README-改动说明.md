# 三元 Bonsai → ninfer · 源码改动包

> **本包是改动的可移植快照**（按目录名覆盖回对应 fork 根目录 → **必须重编**，见 §D → 按 §E 做验证）。
> **基座树 = `Ambolio/ninfer-4090-windows` @ `6eb70a07`（`v1.0.8` 线）**；目录结构与上游 fork 一致，见 `changed-files/`。
>
> ⚠️ **§F 是"中段快照"**（当时 mma 张量核快路径尚未接入，decode 31.0 / prefill 48.8 t/s）。
> **最终结果**以仓库根 `README.md` 为准：decode 62–64 t/s、MTP K=2 96.7–130.8 t/s、PPL 6.445。
> **其中 prefill 356–439 tok/s 所依赖的那条 mma 快路径（`ternary_rowsplit_mma_small_t.cuh`，走 8-token tile、同时覆盖 verify 与 prefill）现已随本包发布**（本行原文曾写"含 mma 快路径"而包里没有 —— 已补齐）。
>
> 根因与教训（两个真 bug + 验证方法学）：见仓库 `docs/04-工程实录-坑与方法学.md`。

## A. 本包包含

| 目录 | 内容 |
|---|---|
| `ninfer-4090w-ternary\**` | v1.0.8 线（我们的开发载体）的全部相关改动：**38 个文件** |
| `ternary-pack\{pack.py, MAPPING.json}` | 打包器 + 权威映射规格（**含 `gdn_value_z` 头级置换修复**）|
| `verify\*.py / *.cmd` | 本轮新增的验证脚本（见 §C）|
| `verify\harness\{rot_test.cu, gemm_test.cu}` | 脱离 ninfer 构建树的 nvcc 测试台（编的是**引擎同一份真代码**）|

## B. 改动清单（本轮）

### B1 新增（三元折叠基旋转的 op 侧接线，M2）
| 文件 | 作用 |
|---|---|
| `src/ops/linear/ternary/ternary_rotation.h/.cpp/.cu` | 旋转的公开入口 + 工作区尺寸 + 开关 |
| `src/ops/linear/ternary/ternary_rotation_kernels.cuh` | **自包含 device 内核**（D1024 归一化 SWHT + 显式符号 + P 置换；正向 / 逆向就地），供独立 nvcc 单测复用 |
| `src/ops/linear/ternary/ternary_row_view.h` | 三元父权重的行切片（PQ2_0 = 32+0 B/组，PTQ1_0 = 24+2 B/组，**按 qtype 推几何**）|

### B2 修改（M3 各 fused 家族 + 工作区 + 布局）
| 文件 | 改动 |
|---|---|
| `src/ops/linear/ternary/ternary_dispatch.{h,cpp}` | 旋转编排（`folded_activation`）+ `ternary_dispatch_basis[_strided]` |
| `src/ops/linear/ternary/ternary_launch.h` / `ternary_rowsplit_gemm.{cuh,cu}` | **激活布局改为 token 主序（ne[0] 连续）**；输出行步长参数；新增 decode 用 warp-per-row GEMV |
| `src/ops/linear/linear.cpp` | 三元分支传工作区；容量函数返回旋转 scratch |
| `src/ops/wrapper/gdn_input_proj.cpp` | 三元分支：plain / conv-snapshot / conv-record 三条路径 |
| `src/ops/wrapper/attn_input_proj.cpp` | 三元分支（4 个投影共用一次旋转）|
| `src/ops/wrapper/linear_add.cpp` | 三元 = GEMM 到 scratch + `residual_add` |
| `src/ops/wrapper/linear_swiglu.cpp` | 三元 = GEMM gate_up + 现成 `silu_mul` |
| `src/ops/wrapper/embedding.cpp` | 三元表查表后施加**逆变换**（就地，无需工作区）+ 诊断 dump 钩子 |
| `include/ninfer/ops/{gdn_input_proj,attn_input_proj}.h` | 带工作区的重载声明 |
| `src/targets/qwen3_6_27b/impl/variant.cpp` | 传工作区 + groupwise-int profile 预留旋转 scratch |
| `src/targets/qwen3_6/impl/runtime/{text_context_impl,text_prefill_impl,program_impl,dflash_impl}.h` | LM head 改用带工作区的 `linear`（折叠后的 output_head 需要旋转 scratch）|
| `src/CMakeLists.txt` | +2 源文件 |
| `ternary-pack/pack.py` | **`gdn_value_z` 头级置换修复**（`perm_row`）|

> 上一棒的改动（格式注册、三元解码、M5 拼头、绑定层、`embed_gather` 三元内核、`tensor.h` 的 `hadamard_*` 字段）
> 也随这些文件一并包含（本包是**当前磁盘状态**，不是增量 diff）。

### B3 新增（mma 张量核快路径 · T = 2..8）—— **本次补齐的那一块**

| 文件 | 作用 |
|---|---|
| `src/ops/linear/ternary/ternary_rowsplit_mma_small_t.cuh` | **按 8-token tile 走序列的 PQ2_0 张量核路径**：一个 warp 管 16 输出行、CTA 4 warp（=128 行），K 按 512 宽 chunk 双缓冲 `cp.async`，权重经 256 项 shared LUT 解包（1 个打包字节 → 4 个三值权重 = 一条 `LDS.64`）。**同时覆盖 verify（T=2..8，即投机验证轮）与 prefill（T>8，付 ceil(T/8) 趟权重）** |
| `ternary_rowsplit_gemm.cu`（修改） | 加 `mma_admits()` 与 `launch_pq2_mma_small_t()`，并把它插进 `launch_ternary_gemm_t8` 的 GEMV 分支**之前**。**回退开关 `NINFER_TERNARY_MMA=0`** |
| `ternary_rowsplit_gemm.cuh`（修改） | 头注释更正：原文写 *"an MMA path are deliberately left out until the numerics are pinned down"*，现已接入 |

> ⚠️ **本包只补这一条**：更宽的 token tile、int8 档（及其 activation scratch 头）、T 分档阈值、解包派发探针，
> 都是这些文档写成**之后**的进展，**不在本包内**（本包的判据是"让已发布的数字可复现"，不是"把最新工作树倒出来"）。
>
> 验证（本次做的）：在 **`<基座 v1.0.8> + changed-files + 本提交`** 搭出的树里，该 TU 用 nvcc（`sm_89` / CUDA 13.3）
> **编译通过 `exit=0`**（唯一输出是既有的 `launch_pq2_gemv declared but never referenced` 警告）。

## C. 验证脚本（都可直接复跑）

| 脚本 | 判据 |
|---|---|
| `verify\check_payload_order.py` | 三元载荷是 row-split 三平面还是交错块（**分布指纹**：zero_share 0.3278）|
| `verify\check_signs.py` | artifact 符号表 vs GGUF 元数据逐值相同 + 宽度前缀和 |
| `verify\check_row_order.py` | **行级指纹的多重集 + 规则逐行比对**（抓 `gdn_value_z` 那个 bug 的杀手判据）|
| `verify\check_assembly.py` | **全量装配审计**：15/15（含 `mlp/gate_up` concat 顺序、`gdn/value_z` 头级置换）|
| `verify\check_embedding.py` + `verify\harness\` 的 dump 钩子 | **引擎侧注入式验证** embedding（gather + 逆变），T=1 与 T>1 都要跑 |
| `verify\oracle_rot.py` + `rot_test.cu` | 旋转内核 vs numpy 显式矩阵，6/6 + 6 类负控 |
| `verify\gemm_oracle.py` + `gemm_test.cu` | 解码+GEMM+旋转组合 vs 纯 numpy，rel_l2 2.8e-03 + 3 类负控 |
| `verify\loadtest.cmd` / `gentest.cmd` | 可复现的装载 / 生成探针（写 `load-*.log` / `gen-*.log`）|

## D. 重建

```powershell
# v1.0.8 线（b1 树）
<BUILD_ROOT>\buildb1inc2.cmd          # 增量；日志 <BUILD_ROOT>\b1inc2.log
# 三元 artifact（改了 pack.py 之后必须重打，产物用新文件名）
<PYTHON>\python.exe <WORKSPACE>\ternary-pack\pack.py build <out.ninfer>
```

## E. ⚠️ 三项必须做的验证（否则会踩同样的坑）

1. **⛔ 这个构建树的 MSVC 头文件依赖扫描是坏的**：改 `.h` **不会触发重编**，必须 `touch` 所有包含它的 `.cpp`。
   （这正是上一棒 `/GS` 栈金丝雀崩溃 `0xC0000409` + 异常数据 `2` 的根因：`bindings.h` 改了、`registry.cpp` 没重编。）
2. **`_tu_check.py` 对同名 `.cpp` 与 `.cu` 会按 stem 去重、静默少编一个** ⇒ 用带扩展名的 needle。
3. **任何 GEMM/变换类改动必须有 T>1 的用例，并在引擎侧验证**（注入式 dump），不能只在独立 harness 里自证：
   T=1 时 token 主序与行主序**完全重合**，所以 T=1 全绿 ≠ 正确。

## F. 该改动的实测结果（2026-09-19 15:2x）

| 指标 | 值 |
|---|---|
| 输出 | **通顺**（`User asks: "The capital of France is". Need answer concise. Final: Paris.`）|
| **PPL** | **6.43557**（黄金参照 6.8196 ± 0.53650 ⇒ 同量级且略优 ✅）|
| decode | 31.0 t/s（等效带宽 249 GB/s，**待优化**：GEMV 快路径已加入但尚未接线）|
| prefill | 48.8 t/s（同上）|
