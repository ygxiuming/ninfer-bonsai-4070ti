# 路线 A「省显存档」（llama.cpp + KVMem）· RTX 3060 12GB 实测

> 实测日期 **2026-10-03** ｜ 机器：**RTX 3060 12GB（sm_86 / 28 SM / ~360 GB/s）**，
> Ubuntu 26.04，驱动 595.91.07（驱动侧 CUDA 13.2），i5-12400F，30 GB 内存。
> 引擎：`kvmem/kvmem-llama.cpp` 的 `kvmem-bonsai-llama.cpp` 分支（Bonsai/prism.3 线），
> **本机自编 CUDA 13.2.86 / `-DCMAKE_CUDA_ARCHITECTURES=86-real`**（官方 Linux 预编译不含三元，见 §2）。

---

## 0. 一句话

在 12GB 的 30 系卡上，这条线拿到 **128K 逻辑上下文**（KVMem 把历史 KV 放主机内存、GPU 只驻留窗口），
**decode 34~41 t/s、prefill 206~233 t/s、峰值显存 8.0~8.8 GiB / 12 GB**，
并把一条藏在 **108,699 token** 提示里的信息准确召回。

---

## 1. 实测速览

| 项 | 实测 | 备注 |
|---|---|---|
| 逻辑上下文 | **131072（128K）** | KVMem：GPU 只驻留 24K 检索窗口 + 10K 生成预留 |
| decode（无投机） | **34.1 t/s**（中文 505 tok）/ 33.9（英文 700 tok） | 600 词长输出，贪心 |
| decode（MTP1） | **39.3 / 41.2 t/s**（+15~22%） | 社区 r3 MTP 头 → Q4_0 合并，草稿 KV f16 |
| decode（MTP2） | 34.6 / 38.6 t/s | **本卡上比 MTP1 慢**，不推荐 |
| prefill | 232.7 t/s @6.8k ｜ 208.9 @36k ｜ **205.7 @109k** | 长提示不明显衰减 |
| 长上下文召回 | **109k 提示 → 正确**（代码藏在 40k 与 56k 段落之间） | 单样本验证，不是矩阵 |
| 多轮前缀复用 | ✅ 命中（`prompt_cache_hit_tokens` 30+） | KVMem 核心机制 |
| 思考模式 | ✅ 正常（`reasoning_budget` 4096，答案正确） | 请求级可关 |
| 峰值显存 | **8.0 GiB**（无 MTP）/ **8.6~8.8 GiB**（MTP1） | `nvidia-smi` 实读，含桌面 1.2 GiB |
| 模型加载 | ~1.3 s（页缓存热）/ 6.95 GiB 权重 | PQT1_0 5.95 GB |
| 编译 | 525/525，**0 错误**，6 核 **7.7 分钟**（ninja 计时，只编 sm_86 一个架构） | 见 §3 |

基准脚本：**`scripts/bench_kvmem.py`**（本仓库自带，`python3 scripts/bench_kvmem.py` 直接出上表数字）。

### 补充实测：256K 档（`long` profile，2026-10-08 验证）

长会话会撞"提示 token + 预留 > `-c`"的硬墙（见 §7 坑 8）。把 `-c` 顶到**模型原生上限 262144** 后可解，
代价只有**主机内存**（历史 KV 不在显存里）：

| 项 | 实测（RTX 3060 12GB） |
|---|---|
| 引擎自报 | `n_ctx=262144`（work 档 131072 的 2 倍） |
| 提示上限 | `262144 − 10240 = 251904` token（原 120832） |
| **12.1 万 token 真请求** | ✅ 通过：预填充 **571 s / 120851 token（211.6 t/s）**，无 OOM |
| 重复同一请求（缓存热） | ✅ **2.9 s**，`cache = 120753 / 120851`，只补算 98 token |
| 显存 | 8.06 GiB（空载）→ 8.58 GiB（跑完 12 万 token 后）—— 与 128K 档基本相同 |
| 主机内存 | 30 GB 机器：9 GiB（空载）→ 15 GiB（装载 12 万 token KV 后） |

启动：`./scripts/start-kvmem.sh long`。注意超长上下文的召回质量是官方标注的实验特性，能装下 ≠ 都记得住。

---

## 2. 为什么路线 A 在本机也必须"自编"一次

这一点容易误判，先说清楚：

- 官方 **rc3 Linux 预编译包**（`kvmem-v0.16.0-rc3-linux-x86_64-cuda13.2.86.tar.gz`）**确实编了 sm_86**，
  但那套是**通用 Qwen3.8 线**（IQ3/IQ4），**不含 Bonsai 三元格式**
  （`general.file_type = 143` = `PTQ1_0`，外加 `prism.hadamard.*` 激活变换）。
- 带三元支持的 **prism.3 分支**，官方**只发了 Windows 预编译**
  （`...-windows-x86_64-cuda12.9-rtx20-30-40-50.zip`）。
- ⇒ Linux + 30 系 + 三元 = **只能自编**（本页做的就是这个）。好消息：编译产物跑起来后
  **不需要系统装 CUDA Toolkit**（动态库都在工作目录里，见 §6 的 `LD_LIBRARY_PATH`）。

---

## 3. 免 root 工具链（本机没有 sudo 密码，全部装在用户目录）

CUDA 13.2 Update 2（`13.2.86`，与官方 rc3/prism.3 构建基线一致）用 **NVIDIA apt 仓库的 .deb 直接解包**，
不装系统包、不改 `/usr/local`：

```bash
# 关键：libnvvm 那个包不能漏，否则 nvcc 报 "nvvm/bin/cicc: not found"
# 清单：nvcc / crt / cudart(-dev) / cccl / driver-dev / culibos(-dev) / nvrtc(-dev)
#       libcublas(-dev) / libnvvm / cuda-tileiras / libnvjitlink(-dev) / libnvfatbin
BASE=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64
for f in cuda-nvcc-13-2_13.2.86-1_amd64.deb libnvvm-13-2_13.2.86-1_amd64.deb ...; do
  curl -fsSL --noproxy '*' -o "debs/$f" "$BASE/$f" && dpkg -x "debs/$f" cuda-stage/
done
# 结果：cuda-stage/usr/local/cuda-13.2/ 就是一棵完整、可用的工具链
```

验证（本机通过）：

```bash
nvcc --version           # Cuda compilation tools, release 13.2, V13.2.86
nvcc -arch=sm_86 -o t t.cu && ./t    # → ok 20.085537（真在 3060 上跑）
```

- cmake 3.31.6（cmake.org tarball）+ ninja 1.12.1（GitHub via ghfast 镜像），同样只解压到 `tools/`。
- 宿主编译器用系统 **GCC 15.2**，nvcc 13.2 直接接受，无需 `-allow-unsupported-compiler`。

**完整一键脚本：`scripts/build-kvmem-sm86.sh`。**

---

## 4. 编译

```bash
git clone --branch kvmem-bonsai-llama.cpp --recurse-submodules --shallow-submodules \
  https://github.com/kvmem/kvmem-llama.cpp.git
cd kvmem-llama.cpp && ./scripts/apply-patches.sh      # 给 Prism 源打 KVMem 适配补丁（30 个文件）
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86-real \                # ★ 30 系就这一处
  -DGGML_CUDA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DKVMEM_BUILD_LLAMA=ON -DLLAMA_KVMEM=ON -DLLAMA_KVMEM_ROOT=$PWD
ninja -C build -j6 llama-kvmem-server llama-kvmem-cli
```

- `-j6` 不是保守，是**内存纪律**：nvcc 单作业峰值 1.5~3 GB，30 GB 内存机器上 `-j12` 会触顶
  （官方 Windows 侧就出过 `-j24` 打爆内存导致内核崩溃的事故）。
- `cmake` 配置输出里出现 `Found CUDAToolkit: ... 13.2.86` 与
  `CMAKE_CUDA_ARCHITECTURES=86-real` 即正确。
- ⚠️ **路径要对上**：上面是「看懂每一步」的手写流程，产物在 `./kvmem-llama.cpp/build/bin/`。
  而 `scripts/start-kvmem.sh` 找的是 **`$KVROOT/kvmem-src/build/bin/llama-kvmem-server`**
  （`KVROOT` 默认 `/mnt/D/code/kvmem-bonsai`）。想直接用启动脚本，就按
  `scripts/build-kvmem-sm86.sh` 的布局来（它克隆到 `$KVROOT/kvmem-src`），
  或者启动时 `KVROOT=` 指到你自己的树、并保证目录名是 `kvmem-src`。

---

## 5. 启动（3060 12GB 档位）

`scripts/start-kvmem.sh`（本仓库）：

```bash
./scripts/start-kvmem.sh work            # 128K 上下文，GPU 窗口 24K+10K，KV q8_0，思考开
KVMEM_MTP=1 ./scripts/start-kvmem.sh work   # 打开 MTP1（需先合并 MTP 头，见下）

# Web UI 可选（源码树的 ui/ 是 Svelte 源，需 Node 构建；没有则只提供 API）
curl -s http://127.0.0.1:18202/v1/models
```

关键参数（来自官方 `start-bonsai.ps1`，逐项对应）：

| 参数 | 值 | 说明 |
|---|---|---|
| `-c` | 131072 | 逻辑上下文（KVMem 把它"摊"到主机内存） |
| `--kvmem-budget` / `--kvmem-gen-reserve` | 24576 / 10240 | GPU 常驻窗口：24K 检索 + 10K 生成；**单轮输出不能超过 reserve** |
| `--kv-dtype` | `q8_0` | 30 系没有 FP8 张量核；q8_0 是这条线的默认档 |
| `--kvmem-block-tokens` | 128 | 检索块大小（budget/reserve 必须是它的整数倍） |
| `--flash-attn` | on | |
| `--reasoning-budget` | 4096 | 思考预算；请求级 `chat_template_kwargs.enable_thinking=false` 可关 |

**MTP（可选，+15~22% decode）**：头部来自社区 `ProCreations/Ternary-Bonsai-2-27B-MTP`
（固定 revision，非官方权重），按官方流程量化合并：

```bash
# 合并脚本只需要 numpy（不需要 pip/root）：wheel 本身就是 zip，解压后用 PYTHONPATH 指过去
curl -O https://pypi.tuna.tsinghua.edu.cn/simple/numpy/numpy-2.5.3-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
unzip -oq numpy-*.whl -d pylibs/

PYTHONPATH=$PWD/pylibs python3 scripts/bonsai-mtp-graft.py \
  --base models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --head model_mtp.safetensors \
  --output models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf --head-type Q4_0
# 脚本自带 SHA256 校验（base 53107f53…、head 7a4a18b2…），输出 6.19 GB
```

> ⚠️ MTP 头只在 `--spec-type draft-mtp` 真正生效时才被加载；否则日志会出现
> `model has unused tensor blk.64.nextn.*` 的告警（无害，说明头没被用上）。
> 本机实测 **MTP1 是甜点，MTP2 反而变慢**（34.6 vs 39.3 t/s）——3060 算力/带宽都低，
> 多草稿的额外计算吃掉了收益。

### 接 OpenAI 客户端（ZCode / Chatbox / Cherry Studio 等）

| 客户端里填 | 值 |
|---|---|
| Base URL | **`http://127.0.0.1:18210/v1`**（走上下文压缩代理；`18202` 是引擎直连，仅调试用） |
| API Key | 任意（服务端不校验，如 `ssss`） |
| 模型名 | 任意（如 `bonsai`）——服务端忽略该字段 |

**★ 必打兼容补丁：`reasoning_effort` 别名。** 模型自带聊天模板只认
`xhigh / medium / low`，而 OpenAI 系客户端默认发 **`high`**（ZCode 发
`{"reasoning_effort":"high","enable_thinking":true}`），模板会直接抛异常 → **HTTP 400**：

```
Jinja Exception: Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.
```

修法与 ninfer 线的 `patches/apply-compat.sh`（P1）同类，但**不用重编引擎**——挂一份模板副本即可：

```bash
./patches/apply-kvmem-compat.sh models/Ternary-Bonsai-2-27B-PTQ1_0.gguf   # 生成别名模板
# start-kvmem.sh 会自动带上 --chat-template-file patches/bonsai-chat-template-alias.jinja
```

**实测接 ZCode**（把本服务配成 provider，2026-10-03）：

| 项 | 实测 |
|---|---|
| 请求规模 | 系统提示 + **33 个工具** = **22,148 token** 提示 |
| 首轮（冷 prefill） | **~100 s**（prefill 231.6 t/s） |
| 第二轮起 | **1~2 s**（KVMem 前缀缓存命中 22147/22148 token） |
| 单轮生成上限 | 10240 token（= `--kvmem-gen-reserve`）；客户端要 32000 会被截断（`finish_reason: length`） |
| 工具调用 | 文本回复正常；27B 三元模型对 33 工具的调用能力**未经充分验证**，别期待能稳定当编码 agent |

> 一句话：接上去能用，**首轮慢、之后每轮 1~2 秒**；想让它当 agent 干活，建议先把
> ZCode 的思考关掉或调低（思考 token 会计入 10240 上限，且明显拖慢）。

---

## 6. 与 ninfer 线（报告里的"路线 B"）怎么选

同一张 3060 12GB 上，两条线是**速度 vs 上下文**的取舍（路线 B 数据引自作者本机的
ninfer 调优报告，本仓库未复测；路线 B 的部署见根 README 的 ninfer 章节）：

| | 路线 A：本页（llama.cpp + KVMem） | 路线 B：自编 ninfer sm_86 |
|---|---|---|
| decode | 34 t/s（MTP1: 39~41） | **108.7 t/s**（DFlash2 K=7，实测） |
| prefill | 206~233 t/s | 640 t/s |
| 上下文 | **128K**（109k 召回已验证） | 24K（其实测最终配置；调优单估算 int8 可到 64K，未实测） |
| 峰值显存 | 8.0~8.8 GiB | 约 8.2 GiB（6.70 权重 + 1.51 运行时，int8/32K） |
| 引擎改动 | 无需改源码（官方分支，只改编译架构） | 需改 4 处源码 + 补 FP8 stub |
| 引擎来源 | 公开仓库、可自编 | 交付包源码树（需自编） |

**一句话**：要**长上下文/多轮记忆/低显存**选 A；要**出字快**（对话、代码补全这类吃速度的场景）选 B。
两条线**互斥**（都吃满显存），`start-kvmem.sh` 与 `start-ninfer.sh` 不要同时起。

---

## 7. 本机踩到的坑（都已解决）

| # | 症状 | 原因 / 处置 |
|---|---|---|
| 1 | `nvvm/bin/cicc: not found` | CUDA 13 的 nvcc 需要单独的 `libnvvm-13-2` 包（apt 依赖列表里容易漏）→ 补包重解 |
| 2 | `--ui-dir must contain a readable index.html` | 源码树 `ui/` 是 Svelte **源码**不是静态页（要 Node 22 构建）→ 无 UI 时不要传 `--ui-dir` |
| 3 | 启动成功但日志 `spec=off`、`unused tensor blk.64.nextn.*` | MTP 头没启用：`--spec-type` 被写成了 `none`（脚本里做字符串替换时数组元素是分开的）→ 用独立变量拼 `SPEC_ARGS` |
| 4 | 走系统代理下载 6 GB 模型只有 ~390 KB/s | 国内镜像应**绕过代理直连**：`curl --noproxy '*'` → hf-mirror 实测 35 MB/s |
| 5 | `python3` 无 pip / 无 numpy（合并 MTP 头要用） | 不要装系统包：直接下 `numpy-*-cp314-*-manylinux*.whl`（wheel 就是 zip），`unzip` 到目录 + `PYTHONPATH` 即可 |
| 6 | 长 prompt 下 decode 偏低（27 t/s） | 短输出样本（8 token）被每轮固定开销主导 —— **测速要看 ≥300 token 的长输出**（报告里"短输出伪造低接受率"的同源教训） |
| 7 | 接 ZCode/客户端后整轮 400：`Unexpected reasoning effort high` | 客户端发 `reasoning_effort:"high"`，模板只认 xhigh/medium/low，模板抛异常 ⇒ 挂别名模板（`patches/apply-kvmem-compat.sh` + `--chat-template-file`），**不用重编**。注意：关掉思考时该字段不校验，所以只有"思考开"的客户端才会踩到 |
| 8 | 长会话某轮开始稳定 400：`{"error":"prompt + max_tokens exceeds n_ctx"}` | **上下文用满**：提示 token + 单轮预留（10240）> `-c`(131072)，上限其实是 **120832** 个提示 token。这是校验期秒拒（不是超时）。处置：新开会话；或 `KVMEM_RESERVE=4096`（让出约 6K 提示空间，输出上限同步降）；或 `KVMEM_CTX=262144` 顶到模型原生 256K（KV 在主机内存，显存不变，超长召回属官方实验特性）|
| 9 | 整轮 400（**53 ms 秒拒**）：`System message must be at the beginning.` | 客户端把系统提醒作为 **system 角色消息插在会话中间**（ZCode 的技能清单实测出现在 messages[3]），模板要求 system 只能在首条。处置：`patches/apply-kvmem-compat.sh`（补丁2）把非首条 system 按 user 轮渲染；**跟图片/长度/缓存无关**，别被带偏 |
| 10 | 带工具的会话整轮 400（**0.2 s 秒拒**）：`{"error":"MTP sampler could not initialize for this request"}`，引擎日志 `failed to parse grammar` | 某个工具的 JSON schema 生成了 llama.cpp 解析不了的**工具调用语法**（实测：ZCode 的 `ReplyToChannel` 参数 `text.maxLength: 2000`，撞上 `llama-grammar.cpp` 的 `MAX_REPETITION_THRESHOLD=2000` 保护：重复数 × 规则复杂度 ≥ 2000 即拒绝解析）。KVMem 服务端在"开视觉 / 用户检索策略"时**必须有 MTP 采样器**，语法挂 → 采样器初始化不了 → 整个请求被拒。处置：打 **`patches/llama-kvmem-server-grammar-fallback.patch`**（语法失败时丢弃语法约束继续跑、MTP 保持开启；工具调用仍可用，只是少一层格式强约束）→ `ninja llama-kvmem-server` 重编 + 重启引擎。**与工具数量无关**（只带这一个工具也秒挂；去掉它 250 个工具都正常）。另注：250 个工具时提示本身就 ~15 万 token（首次预填充 ≈11 分钟）——用不到的 MCP 建议关掉 |

---

## 8. 日常运维与排错速查

### 8.1 档位与覆盖变量

| 档位 | `-c` 上下文 | GPU 窗口（budget+reserve） | 单轮输出上限 | 提示上限 = ctx − reserve |
|---|---|---|---|---|
| `work`（默认） | 131072 | 24576 + 10240 | 10240 | 120832 |
| `long` | **262144** | 24576 + 10240 | 10240 | **251904** |
| `low` | 65536 | 12288 + 6144 | 6144 | 59392 |
| `fast` | 8192 | 4096 + 2048 | 2048 | 6144 |

```bash
./scripts/start-kvmem.sh work|long|low|fast [port]   # 启动
./scripts/start-kvmem.sh stop|status|logs [profile]  # 停止/状态/跟日志
```

环境变量覆盖默认值（长会话撞上限时用）：

| 变量 | 作用 |
|---|---|
| `KVMEM_CTX=262144` | 逻辑上下文顶到模型原生 256K（历史 KV 在主机内存，**显存不变**） |
| `KVMEM_RESERVE=4096` | 缩小单轮预留 → 给提示让出约 6K 空间（输出上限同步降到 4096） |
| `KVMEM_BUDGET=32768` | 加大 GPU 检索窗口（多吃显存） |
| `KVMEM_MTP=1` | 开 MTP1 投机解码（需 MTP 模型，+15~22% decode，显存 +0.6 GiB） |
| `KVMEM_DRAFT_N=2` | 草稿 2 个（**3060 上反而更慢**，别用） |
| `KVMEM_MODEL=<路径>` | 换模型（如 PQ2_0 打包版），不用重编 |

### 8.2 三条硬规则

1. **提示 token ≤ ctx − reserve**。超了**校验期秒拒**（不是超时、不是模板问题）：`prompt + max_tokens exceeds n_ctx`。
2. **单轮输出 ≤ reserve**（10240；**思考 token 计入**），超了截断：`finish_reason: length`。
3. **重启服务 = 清空 KVMem 缓存**（缓存在进程内）→ 长会话的下一个请求要全量重算
   （12 万 token ≈ 9.5 分钟）。所以别频繁重启；要重启就顺带换到 `long` 档。

### 8.3 症状 → 原因 → 处置

| 症状 | 原因 | 处置 |
|---|---|---|
| 整轮 400：`Unexpected reasoning effort high` | 客户端发 `high`，模板只认 xhigh/medium/low | 跑 `patches/apply-kvmem-compat.sh`（补丁1）；`start-kvmem.sh` 自动挂 |
| 整轮 400（**53 ms 秒拒**）：`System message must be at the beginning.` | 客户端把系统提醒当 **system 消息插在会话中间**（ZCode 的 skills 清单实测在 messages[3]） | 同上脚本（补丁2：中途 system 按 user 轮渲染）；**与图片、长度、缓存都无关** |
| 整轮 400：`prompt + max_tokens exceeds n_ctx` | 会话把上下文用满（提示 > ctx − reserve） | 新开会话 / `KVMEM_RESERVE=4096` / 用 `long` 档 |
| 带工具的会话整轮 400（**0.2 s 秒拒**）：`MTP sampler could not initialize for this request` | 某工具 schema 生成的工具调用语法解析失败（如 `maxLength: 2000` 撞 llama.cpp 的 `MAX_REPETITION_THRESHOLD=2000` 保护） | 已根治：`patches/llama-kvmem-server-grammar-fallback.patch`（引擎侧，重编后永久生效）；正常时日志出现 `dropping grammar constraint` 而非 400 |
| 界面长时间没动静 | ① 在预填充（日志 `progress` 在涨）② 思考 token 都在"内心独白"里 | 看日志；关客户端思考（`chat_template_kwargs.enable_thinking=false`） |
| 每轮都要好几分钟 | 缓存没命中（`cache = 0`），每次全量重算 | 若因重启导致：等它跑完一次即可恢复；若是会话过大：新开会话 |
| **一个 token 都不出**（客户端像死了） | ① 提示里**带了图片**——本部署没挂视觉塔，请求直接失败 ② 提示太长（十几万 token）正在预填充 | ① 日志出现 `multimodal_error: multimodal prefill failed or cancelled` ⇒ 别发图；要视觉见 §8.6 ② 看日志 `progress` 是否在涨，涨就是正常在算，别重试 |
| 显存不够 / 起不来 | 旧实例没退，或与 ninfer 线/游戏同时跑 | `start-kvmem.sh stop` + 等 `nvidia-smi` 回落；跑游戏/视频生成前先停服务 |

### 8.4 日志怎么读

**先找对文件**：日志名跟档位走（`work` → `kvmem-work.log`，`long` → `kvmem-long.log`）。
最省事的写法：

```bash
tail -3 "$(ls -t ~/kvmem-logs/kvmem-*.log | head -1)"   # 自动挑最近在写的那个
```

一行日志分三段（时间都是**毫秒**）：`prompt eval`（读输入，慢一般慢在这）/ `cache`（缓存命中）/ `eval`（写输出）。

```
prompt processing, n_tokens = 50560, progress = 0.64, t = 241 s / 209 t/s   ← 预填充中，progress 在涨=正常
prompt eval time = 385522 ms / 79513 tokens (206 t/s), cache = 0            ← 7.9 万 token 全量重算 → 6.4 分钟
prompt eval time = 1892 ms / 174 tokens (92 t/s), cache = 120069            ← 12 万 token 复用 → 1.9 秒
prompt eval time = 1384 ms / 274 tokens (198 t/s), cache = 0                ← 274 token 无缓存 → 1.4 秒（正常）
eval time = 236 ms / 9 tokens (38 t/s)                                      ← 写回答用了 0.24 秒
n_gen = 3820, tg = 26.8 t/s, tg_3s = 25.7 t/s                               ← 正在出字
total time = 1620 ms / 283 tokens                                           ← 整个请求 1.62 秒
```

> **别把 `cache = 0` 当故障**：它只表示"没有可复用的前缀"。几百 token 的输入没缓存也是秒回；
> 真正要等的是 **`cache = 0` 配上几万 token 输入**（那才是分钟级）。判断标准看 `total time`：
> 几秒内正常；反复分钟级 → 查是不是每轮都在全量重算；日志完全不动 → 才叫卡住。

### 8.5 资源账（3060 12GB / 30 GB RAM 实测）

- **显存**：空载 8.06 GiB，装载 12 万 token 后 8.58 GiB（历史 KV 不在显存里，**只跟 GPU 窗口大小走**）
- **内存**：约 **0.33 GB / 万 token**（q8_0 KV）；12 万 token ≈ 4 GB。256K 档 ≈ 8.5 GB，
  在 30 GB 机器上已接近上限，跑长会话时别同时开别的吃内存的程序
- **速度**：decode 34 t/s（MTP1 39~41）｜prefill 206~233 t/s（与上下文长度基本无关）

### 8.6 看图（视觉）—— 默认开启，已实测

视觉塔：`models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`（0.63 GB，官方 Q8_0 包，
sha256 `6807ede6…` 与 HF 一致）。**启动时自动挂上**（显存 +约 0.6 GB），贴图即用：

```bash
./scripts/start-kvmem.sh long                      # 默认带视觉
KVMEM_VISION=0 ./scripts/start-kvmem.sh long       # 关掉（显存紧张 / 只跑纯文本）
KVMEM_VISION_CPU=1 ./scripts/start-kvmem.sh long   # 保留视觉，编码器放 CPU
```

**本机实测（RTX 3060 12GB）**：

| 测试 | 结果 |
|---|---|
| 合成图（白底 + 左中下红圆 + 右上蓝方）首轮 | 76 s（含视觉编码器加载），描述**完全正确** |
| 同一张图第二轮 | **2.2 s**（缓存命中） |
| **中文表格截图**（1100×850，4 列） | **14.7 s**，4 列表头**逐字读对**（提示 2067 token） |
| 显存 | 挂视觉后服务约 8.7 GiB（比纯文本 +0.6 GiB） |

不挂视觉时发图会失败：`E srv KVMEM_TRACE multimodal_error error=multimodal prefill failed or cancelled`
（客户端表现为"一个 token 都不出"）。模型卡提示视觉是三元气化后差距较集中的类别
（MMMU-Pro 71.36），**读结构清晰的图表/截图可以，别指望精细图表推理**。

### 8.7 上下文自动压缩代理（默认启动，端口 18210）—— 已实测

**为什么需要它**：客户端（ZCode 等）每轮把**完整历史**重发。历史涨到十几万 token 后，冷缓存要全量
预填充十几分钟；而客户端等约 10 分钟就放弃，**中断的请求又不写缓存** ⇒ 永久重试死循环，永远不出字。
压缩代理把"较早的消息"替换成滚动摘要，给提示长度一个硬上限。

**链路**：客户端 → `http://127.0.0.1:18210/v1`（代理）→ `http://127.0.0.1:18202`（引擎）

```bash
./scripts/start-kvmem.sh long                 # 引擎 + 代理一起起（默认）
./scripts/start-kvmem.sh status               # 分别报告引擎与代理
KVMEM_PROXY=0 ./scripts/start-kvmem.sh long   # 只要引擎
```

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `KVMEM_PROXY_THRESHOLD` | 45000 | 估算 token 超此值触发压缩 |
| `KVMEM_PROXY_KEEP_RECENT` | 10 | 压缩时保留最近多少条原文 |
| `KVMEM_PROXY_KEEP_EST_MAX` | 15000 | 保留部分的估算 token 上限（超了继续往摘要挪）|
| `KVMEM_PROXY_SUMMARY_MAX_TOKENS` | 1600 | 摘要生成上限 |
| `KVMEM_PROXY_REBUILD_MIN` | 12000 | 滑出 keep 窗口的新消息攒到多少 est 才重建摘要；不够则沿用旧摘要原样追加 |

> 🔁 **2026-10-18 优化：只在攒够时才重建摘要（`mode=defer`）**。原实现每轮都把新滑出的
> 2~3 条消息重新摘要一遍：摘要文本一变，主请求的前缀就失效 ⇒ 每轮全量重算（实测 ~30K
> 提示 ≈ 2 分钟/轮，工具轮变成 ~3 分钟一轮）。现在未攒够 `REBUILD_MIN` 时沿用旧摘要、
> 把滑出消息**原样追加**在不动的摘要后面——前缀稳定 ⇒ KV 缓存命中，只算新增那几条（几秒）；
> 大约每 4~5 轮才重建一次（那一轮仍是 ~2 分钟）。代理日志出现 `摘要暂缓 滑出 N est` 即为
> 该模式；`mode=compress|incremental` 才是重建轮。**注意 `REBUILD_MIN` 别设得比 keep 窗口
> 还小**（keep 最多 15000 est，设小了会退化成每轮重建）。

**本机实测（3060）**：

| 场景 | 结果 |
|---|---|
| 1.4 万 token 会话（阈值降到 6000 强制触发） | 引擎只收 **1534 tok**；摘要 282 字；**埋在第 8 条的"项目代号 蓝鲸-7742"被摘要保住**，回答正确 |
| 同一会话第二轮（+2 条） | **增量**模式：只摘要新增 2 条，**8 秒**；提示 1292 tok |
| 流式（SSE）/ 普通 JSON 透传 | ✅ 都正常（chunked 转发，不缓冲）|
| 摘要失败时 | **原样透传**（宁可慢，不丢内容）|

> ⚠️ 摘要是**有损**的：久远细节可能被压掉，关键事实（代号/路径/决定）会被显式保留。
> 看代理日志 `est X→~Y` 与 `mode=compress|incremental|reuse` 判断是否生效；
> 状态在 `~/.cache/kvmem-proxy-state.json`（按会话存，删掉即重置）。
> 估算器对中文的偏差在 30% 以内（阈值已留余量）；拿不准就把阈值调小 20%。
>
> 🔧 **实现注意（别改回去）**：SSE 转发必须用 `resp.read1()` 而不是 `resp.read(4096)` ——
> 后者（BufferedReader 语义）会**攒满 4096 字节才返回**，在 ~200 B/s 的出字速度下等于"憋 20~40 秒
> 再一次性倒出来"：总耗时不变，但观感上明显变慢。已实测：`read1` 下经代理与直连的首事件
> 一致（模拟流 1 ms；真机首字 67 ms vs 直连 490 ms，差值只是缓存热度）。

---

## 9. 复现清单（TL;DR）

```bash
# 1) 引擎（免 root，见 build-kvmem-sm86.sh 全文）
./scripts/build-kvmem-sm86.sh                       # 解包 CUDA → clone 分支 → apply-patches → ninja

# 2) 模型（走国内镜像，直连不走代理；sha256 53107f530aa52eb0…）
curl -L --noproxy '*' -o models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  https://hf-mirror.com/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf

# 3) 起服务（128K）：引擎 18202，压缩代理 18210（客户端接 18210）
./scripts/start-kvmem.sh work

# 4) 冒烟（引擎直连；想连整条链路就把端口换成 18210）
curl -s --noproxy '*' http://127.0.0.1:18202/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"用一句话中文自我介绍。"}],"max_tokens":64,"temperature":0}'

# 5) 基准（本表全部数字由它产出）
python3 scripts/bench_kvmem.py
```

许可与出处：引擎 `kvmem-llama.cpp` / PrismML 的 `llama.cpp` fork（均在各自目录保留 LICENSE）；
模型 `prism-ml/Ternary-Bonsai-2-27B-gguf`（Apache-2.0，"Created using Bonsai by Prism ML."）；
MTP 头为社区产物（`ProCreations`），非官方权重。本页数据为本机实测，单卡单样本，不代表上限。
