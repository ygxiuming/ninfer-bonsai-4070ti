# 02 · 把「普通（微调）模型」转成 NInfer 原生 `.ninfer`

> 目标：把一个 **HuggingFace safetensors** 格式的模型（含各种**微调/去审版**），
> 用 NInfer 自带的转换框架，转成可直接加载运行的 `.ninfer`。
>
> 特点：**零引擎改动**——所有目标格式都在引擎里已注册，转换只是"读权重 → 量化 → 装对象"。

---

## 1. 原理

NInfer 的转换框架在 `tools/convert/`，结构是**按架构分目录的"配方（recipe）"**：

```
tools/convert/
  common/
    safetensors.py     ← 读 HF safetensors 分片
    quantize.py        ← 量化器（把 fp16/bf16 权重压成目标格式）
  qwen3_6_27b/         ← 一套架构 = 一套 convert + recipe + inventory + verify
    convert.py
    recipe.py          ← 逐张量 → 目标格式（quantization recipe）
    inventory.py       ← 源张量清单
  qwen3_6_35b_a3b/
  qwen3_8_27b/
    dflash2_recipe.py  ← 可选：同时装 DFlash2 草稿头
    fp8_embedding.py
```

**配方决定每个张量用什么格式**（如 attention/MLP 投影 → `Q4G64`，embedding → `BF16`，某些 → `Q5G64`）。
转换器把这些张量写成 `.ninfer` 的**对象 + 布局**，并绑到目标架构。

---

## 2. 支持的架构

转换框架目前覆盖这几套骨架（**必须是其中之一**，否则要先补一套 recipe）：

| 架构目录 | 底座 |
|---|---|
| `qwen3_6_27b` | Qwen3.6-27B（也承载 Qwen3.8-27B 的主体） |
| `qwen3_8_27b` | Qwen3.8-27B（含可选 DFlash2 草稿头） |
| `qwen3_6_35b_a3b` | Qwen3.6-35B-A3B（MoE，含可选 DFlash 草稿头） |

> **关键**：只要你的模型是**这些骨架的微调/去审/融合版**（架构不变、只是权重不同），就能直接用对应 recipe 转。

---

## 3. 步骤

### 3.1 准备源

把 HF 模型（safetensors + config.json + tokenizer）下载/准备到一个目录，例如 `<SRC>`。

### 3.2 跑转换

命令是 `python -m tools.convert.<arch>.convert`，**必须在源码树根目录**运行：

```bash
# 27B（Qwen3.6 / 3.8 主体）
python -m tools.convert.qwen3_8_27b.convert \
  --model <SRC> \
  --dflash2-model <DFLASH2_SRC> \      # 若无 DFlash2 草稿，给一个占位/跳过
  --out <OUT>.ninfer \
  --device cuda
```

```bash
# 35B-A3B（MoE）
python -m tools.convert.qwen3_6_35b_a3b.convert \
  --model <SRC_35B> \
  --dflash-model <DFLASH_SRC> \
  --out <OUT_35B>.ninfer
```

### 3.3 产物

- `<OUT>.ninfer` —— 完整制品
- `<OUT>.ninfer.conversion.json` —— **转换记录**：源路径、recipe id、格式统计、对象数、耗时、环境（torch/GPU）

### 3.4 检查

```bash
python -m tools.artifact.inspect <OUT>.ninfer --objects
```

---

## 4. 本机实测（可复现的证据）

本机在一张 **RTX 4080 SUPER / Windows / torch 2.12+cu130** 上转过一个 **27B 去审微调版**：

| 项 | 值 |
|---|---|
| recipe | `qwen3_8_27b-v1` |
| 耗时 | **114.2 秒** |
| 产物 | **18.21 GB**（1124 对象 / 1199 张量 / 18 分片）|
| 格式分布 | BF16 ×582、FP32 ×96、**Q4G64 ×183**、**Q5G64 ×246**、Q6G64 ×1、W8G32 ×9 |
| 布局 | `contiguous-le-v1` ×679、`row-split-k128-v1` ×439 |
| 环境 | Windows-11 / RTX 4080 SUPER / CUDA 13.0 |

> 也就是说：**一张消费级 32 GB 卡在 Windows 上，两分钟级别就能把一个 27B 压成 `.ninfer`。**

---

## 5. 判据

1. 转换器正常结束，打印 `complete: <bytes> bytes in <s>s`；
2. 生成 `.conversion.json`，且 `objects.count` 合理（27B ≈ 1124）；
3. `tools.artifact.inspect` 能列出对象；
4. **能装载**：`ninfer.exe <OUT>.ninfer --prompt "hi" --max-new 1` 不报格式/尺寸/未消费对象类错误。

---

## 6. 坑

| # | 症状 | 对策 |
|---|---|---|
| 6.1 | `unsupported tensor` / 形状不符 | 源模型架构与 recipe 不匹配，或用了错误的 revision |
| 6.2 | 内存爆 | 转换要把分片读进内存；大模型需足够 RAM（27B 建议 ≥64 GB）|
| 6.3 | CUDA 不可用 | `--device cuda` 需与引擎同代的 torch+CUDA |
| 6.4 | 制品版本 | 本线产出 **v2**；上游主线是 **v3**（两者引擎不通用，看 magic 版本选引擎）|

---

## 7. 附：制品版本（v2 / v3）与引擎的匹配

**选引擎的唯一判据 = artifact 的 magic 版本**：

- **v2** → 本线 / 1.2.0 引擎可读；
- **v1**（老格式）→ **只有 1.2.0 能读**，本线报 `artifact magic is not NInfer v2` 拒载。

跨版本升级：上游提供离线升级器（`docs/weight-conversion.md` 里的 "upgrade an existing v2 artifact"），**不必重下权重**。
