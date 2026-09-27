# 03 · 把「三元 + Hadamard 旋转基」模型移植进 NInfer

> 目标：让 NInfer 能**原生读/跑**三元（1.58-bit）权重模型——本项目用的是
> **Ternary Bonsai 2 27B**（底座 Qwen3.8-27B，架构未改，权重存于 Hadamard 旋转基）。
>
> 与 [02](02-微调模型转-NInfer.md) 的本质区别：**02 是"换权重"，本文是"扩引擎"**——
> 三元是 NInfer **原本不存在**的格式，必须先给引擎加格式 + 运行时旋转，再谈打包。

---

## 0. 为什么难：三元不是一种"编码"，是一次引擎能力扩展

三元的权重值域是 `{−1, 0, +1}`（加每 128 个一组共享的 FP16 scale），且**存于旋转基**：
训练/量化时把每行做了一次块内正交旋转再取值，运行时**必须对激活施加匹配的逆变换**，
否则"形状全对、输出乱码"。所以要做三件事：

1. **格式**：注册新的 QType（`PQ2_0_G128` / `PTQ1_0_G128`）+ 解码语义；
2. **运行时**：激活侧施加 `P → s → H` 变换（正向），embedding 查表后施加逆向；
3. **接线**：所有 fused 家族（attn/gdn/swiglu/add/LM head/embedding）都要能走三元。

---

## 1. 数学与布局（照此实现即可）

### 1.1 三元格式

| 格式 | group | base/组 | high/组 | scale/组 | 合计 |
|---|---:|---:|---:|---:|---:|
| `PTQ1_0_G128` | 128 | 24 B (`qs[24]`) | 2 B (`qh[2]`) | 2 B (`d`) | **28 B/128** |
| `PQ2_0_G128` | 128 | 32 B (`qs[32]`) | 0 | 2 B (`d`) | **34 B/128** |

解码（与 `ggml-quants.c` **逐字一致**）：

- `PQ2_0`：`code = (qs[j>>2] >> (2*(j&3))) & 3`，值 `= (code − 1) · d`（码本 {−1,0,+1,+2}）；
- `PTQ1_0`：`stages{32,16,8}` 游走 + **`&0xFF` 回绕** + `((q*3)>>8) − 1`，尾部 `qh` 覆盖最后 8 个（n 慢变 h 快变）。

尺寸反验：`[248320,5120]` → PTQ1_0 **278,118,400 B** / PQ2_0 **337,715,200 B**。

### 1.2 Hadamard 旋转（M2 的完整序列是 **P → s → H**）

- `H` = 归一化 Sylvester–Walsh–Hadamard，块宽 **1024**，`H[i][j] = (popcount(i&j)&1) ? −1 : +1`，再乘 `1/√1024`（`0x1p-5f`）；
- **蝶形约定**（与 in-tree 实现逐字一致）：低元 `x+y`、高元 `x−y`；
- `s` = 每输入宽度一份的 ±1 向量：权重输入宽 `K` 拥有 `n_blk = K/1024` 行 × 1024 个符号，激活第 `b` 个 1024 块用第 `b % n_blk` 行；
- **正向**：`y = W'·(H·(s ⊙ (P·x)))`（先符号、再旋转）；
- **逆向**（仅 `token_embd` 查表后）：`h = s ⊙ (H·z)`（先旋转、再符号）；
- **P 置换**：只作用于 `gdn/output` 一个权重（K=6144）。**本项目的打包器已把 GDN 张量统一成 grouped，运行时不再施加 P**（实测证实）。

### 1.3 ⛔⛔ 头号坑：激活布局是 **`ne[0]` 连续（token 主序）**

NInfer 的 Tensor 与 ggml 同源：`ne[0]` 是连续轴。所以 `ne=(K,T)` 的激活元素 `(k,t)` 在内存里是

```
x[k, t]  →  t*K + k        （不是行主序 k*T+t）
out[n,t] →  t*out_row_stride + n
```

**T=1 时两种排布完全重合** ⇒ 只测 T=1 的解码/oracle **永远发现不了这个错**。
必须**至少有一个 T>1 用例，且在引擎侧验证**（见 §4.2）。

---

## 2. 引擎侧改动

新增 / 修改集中在三元相关文件：

```
src/ops/linear/ternary/
  ternary_rotation.{h,cpp,cu}          ← 正向/逆向旋转入口 + 工作区
  ternary_rotation_kernels.cuh         ← 自包含 device 内核（D1024 SWHT + 显式符号 + P）
  ternary_rowsplit_gemv.cuh            ← decode 快路径（T=1，warp-per-row）
  ternary_rowsplit_mma_small_t.cuh     ← T≥2 张量核快路径（mma.m16n8k16）
  ternary_rowsplit_gemm.{cuh,cu}       ← GEMM 主路径 + 布局（token 主序）+ 路由
  ternary_dispatch.{h,cpp}             ← 需要时旋转再跑 GEMM；basis 变体供"旋转一次喂多权重"
  ternary_launch.h / ternary_row_view.h / ternary_rowsplit_storage.cuh
```

配套修改：

| 文件 | 改动 |
|---|---|
| `src/artifact/{binder,reader,storage_layouts,typed_binding}` | 注册新格式/布局；绑定层读**artifact 声明值** |
| `src/core/tensor.h` | `Weight` 加符号表字段（`hadamard_signs` / `hadamard_n_blk`）|
| `src/ops/linear/linear.cpp` | 三元分支传工作区 |
| `src/ops/wrapper/{embedding,attn_input_proj,gdn_input_proj,linear_swiglu,linear_add}.cpp` | 各 fused 家族的三元分支（**能组合就不写新 kernel**）|
| `src/targets/qwen3_6_27b/impl/**` | 传工作区；LM head 用带工作区的 linear |
| `src/CMakeLists.txt` | +ternary_rotation.cu/.cpp |

**不需要改**（实测结论）：`gdn_gating_proj`（a/b 是 BF16，非折叠）、`sparse_moe`（27B dense 不走）。
⚠️ 若目标是 **MoE** 模型，则 `sparse_moe` **必须**补三元 expert 内核（本项目未做）。

### 2.1 两条速度路径

- **decode（T=1）**：`warp-per-row GEMV`。PQ2_0 的打包使 lane `l` 的 4 列 `4l..4l+3` 全落在**字节 `l`** ⇒ 每组每 warp 只读 1 B codes + 2 B scale，**零 `__syncthreads`**，5 级 shuffle 归约；
- **T≥2（验证轮 + prefill）**：`mma.m16n8k16`，一个 mma 承载 **16 行 × 8 token × 16 k** ⇒ tile 内所有 token **共用同一次权重读取**。关键调参：**K 分块 512**（不是 256，本轮最大单项）、**4 warp/CTA**（CTA 数是决定性变量）。

---

## 3. 打包器：GGUF → `.ninfer`

因为 NInfer 全树**没有 GGUF 读取器**，我们自写打包器 `tools/pack.py`：
自写 GGUF 解析 → **把 402 个三元矩阵的 2-bit 码字逐字节搬运**（绝不反量化再量化）→ 按 row-split 三平面装配。

逐张量规则（权威在 `tools/MAPPING.json`）：

1. **`norm_shift`**：凡 `*.norm.weight` 值 = `gguf − 1.0`；⛔ **唯独 `gdn/norm`（`ssm_norm`）原样**（它的 raw 已 ≈+1）；
2. **`gdn_v` tiled→grouped**：48 头轴，**必须按头粒度**施加：`perm48(head)*128 + inner`（⚠️ 直接套 `perm48(row)` 是**非双射**——见 §4.1 bug②）；
3. **`attn_q` 逐 head 交错**：N=12288 视作 48 个 256 行块，偶数块 = 24 个 query 头、奇数块 = 24 个 gate 头；
4. **`mlp/gate_up`** = `concat(ffn_gate, ffn_up)`（沿行）；
5. **三元码逐字节搬运**（row-split 三平面，group=128）；
6. **globals / MTP / vision**：与源同族，按需借模板载荷。

> ⚠️ **模板（`--template`）不只是"载荷来源"，它是骨架/清单。**
> `pack.py` 会**遍历模板自己的对象名表**逐个映射；只有映射表**以外**的对象（`vision/*`、`mtp/*`、`frontend/*`、`text/draft_head*`）才走 `borrow`（源码里 `return None` 那一支）。
> ⇒ **模板必须与目标制品同 schema**，即 **`identity.weights_id == "groupwise-int"`**。
>
> **同族的 `nvfp4` 制品用不了**：它把投影**融合**了 —— `gdn/a_b_projection`、`gdn/query_key_value_z`、`attention/query_key_gate_value` —— 这些名字不在映射表里，一进来就 `unmapped gdn object` 中止。
>
> **怎么产一份正确的模板**（基树自带两个转换器，**schema 由"跑哪一个"决定**）：
> ```bash
> # ✅ groupwise-int —— pack.py 要的这份
> python3 -m tools.convert.qwen3_8_27b.convert \
>   --model /path/to/Qwen3.8-27B \
>   --dflash2-model /path/to/Qwen3.8-27B-DFlash2 \
>   --out out/qwen3_8_27b.ninfer
> # ❌ python3 -m tools.convert.qwen3_8_27b.convert_nvfp4 ...  → 产 nvfp4，schema 不匹配
> ```
> **自检（任一条满足即为正确）**：① `identity.weights_id == "groupwise-int"`；
> ② `text/layers/3/` 下是 `attention/query_key` 与 `attention/gate_value` **两个**对象（若是单个 `attention/query_key_gate_value` ⇒ nvfp4，用不了）。
> 详见 README「常见问题」。两个路径也可用 `--gguf` / `--template`（或环境变量 `NINFER_TERNARY_GGUF` / `NINFER_TERNARY_TEMPLATE`）覆盖，**不必再手改源码常量**。

---

## 4. 验证方法学（**本项目最值钱的部分**）

"搬运无损"类判据对**偏移/置换错误完全盲**。必须换判据。

### 4.1 两个真 bug（都不是猜测，都有可复跑判据）

**① 头号：激活布局**（§1.3）。判据 = **引擎侧注入式 dump**：
`T=1 cos +0.999997 ✓ / T=55 cos +0.006 ❌`。T=1 全绿、T>1 全乱 ⇒ 只能是布局。

**② 第二：打包器 `gdn_value_z` 头级置换漏 ×128**。`perm48` 套在 6144 行上而非 48 头上 ⇒ 非双射 ⇒ 重复行 + 丢失行。
该 bug **守恒一切可数之物**（尺寸/行数/字节/往返无损全绿）。判据 = **行级指纹的多重集比对**：
v1 `6144 行 → 2080 distinct、4064 缺失、2048 重复`；v2 `6144/6144 全对`。

### 4.2 方法学（会给出"自洽的假绿"的那类）

| # | 判据 | 能证伪 |
|---|---|---|
| 1 | **分布特征自证**（尺度中位数/全正/`zero_share`）| 载荷偏移或平面错位。`zero_share` 精确 = `0.3278`（PQ2_0 理论零码占比）|
| 2 | **跨实现互验**（两个独立实现解同一份数据）| 单一实现的系统性误读 |
| 3 | **负控必须存在** | "判据太松"导致的假通过 |
| 4 | **行级指纹比「多重集」+ 直接比对** | 行置换/重复/丢失（尺寸守恒）|
| 5 | **T>1 用例 + 引擎侧验证** | 布局/步长类错误（T=1 全绿也测不出）|
| 6 | **端到端数值口径用 PPL** | 采样温度/模板带来的错觉 |

### 4.3 四个验证脚本（缺一不可）

```
python check_payload_order.py <art> 'text/layers/0/gdn/output'   # 是否 row-split 三平面（分布指纹）
python check_signs.py         <art> <hadamard-meta.json>          # 符号表 vs 元数据逐值
python check_row_order.py     <art> <source.gguf>                 # 行级指纹多重集
python check_assembly.py      <art> <source.gguf>                 # 全量装配审计（目标 15/15）
```

---

## 4.5 ★ MTP 拼头（只做一次，但**决定成败**）

三元模型**不自带 MTP**（包内无 NextN 张量）。要把投机解码用起来，必须**把 MTP 头拼进 artifact**：

- **来源**：Qwen3.8-27B 自带的 **NextN 单层**（与本体同族、**同一份**）；
- **规格**：按 NInfer 的 `mtp/*` 12 张量搬——含 `qkv 14336` / `gate_up 34816` 融合 + `W8G32` 量化；
- **能直接借的前提**：MTP 块**不做 Hadamard 旋转**（残差流在**原始基**）；
- **精度判据**：重建头 vs artifact 自带头 **12/12 余弦 ≥ 0.99966**、**7 个 norm 逐字节相同** ⇒ 数值精确。

**接受率与收益**（实测）：

| 项 | 值 |
|---|---|
| **K=2 接受率** | **≈ 59.4%**（**强依赖任务**：英文代码 **40.8%** → 中文短答 **80%**）|
| **tok/轮** | **2.19**（实测 1.82 ~ 2.38）|
| **窗口** | **N=2 最优**；N=3/4/5 崩到 **40.9 / 33.2 / 22.9%**（N≥6 本线不支持）|
| 长度效应 | 说明文 71 token 接受 66.7% → 120 token 54.8% |
| **收益** | 纯解码 **62 → 96.7~130.8 t/s** |

> ⚠️ **没有这一步就只有 62 t/s 的裸解码，KPI 必挂。**
> 已否掉的替代：`--lm-head-draft`（更差，说明文接受率 63.5% → 55.0%）。

---

## 5. 结果（实测）

| 项 | 结果 | 判据 |
|---|---|---|
| 端到端正确性 | **PPL 6.43557**（旋转线）/ **6.445008**（mma 线）| 黄金参照 6.8196 ± 0.5365；错 ⇒ 几百 |
| 纯解码（T=1） | **62–64 t/s** | 等效带宽 ~447 GB/s > llama.cpp 同卡 356 GB/s |
| 单流（MTP K=2） | **96.7 – 130.8 t/s** | 4/5 负载 ≥101 |
| prefill | **356–439 tok/s** | 追平并超过非三元 |
| 制品 | 7.74 GB（文本 6.696 GiB）| 红线 5.5–7.2 GiB |
| 基准 | **19/20 = 95%** | 同引擎全精度 27B = 19/20（**打平**） |

---

## 6. 复现清单（从零到跑通）

| # | 步骤 | 判据 |
|---|---|---|
| 1 | 环境自检 | 4080S / sm_89 / CUDA 13.3 |
| 2 | 读 GGUF 元数据 | `block_size=1024`、`Σsign_widths=28672`、`weight_names=401`、`inverse=[token_embd]` |
| 3 | 打包 | `pack.py build <out>` → 尺寸/对象数符合 |
| 4 | **验 artifact** | 4 脚本全绿（尤其 `check_assembly` 15/15）|
| 5 | 改引擎（或用 `patches/`）| 见 §2 |
| 6 | 构建 | `BUILD OK`（改 `.h` 先 touch `.cpp`，见 doc 01 §4.1）|
| 7 | 端到端 | 输出通顺；PPL ≈ 6.4–6.8 |
| 8 | 速度 | T=1 62~64；MTP K=2 ≥101 |

> 更细的逐步命令与"测量方法学会骗你"（L2 flush、nsys 图内 kernel、idle 时钟…）见 [04-工程实录](04-工程实录-坑与方法学.md)。
