# 3060（sm_86）容器镜像 —— llama.cpp + KVMem 线

> 给 **RTX 30 系（sm_86）**用的镜像，和 `docker/` 下那套 ninfer 镜像（**sm_89 / sm_120a = 40/50 系**）
> 是**两条不同的线**，互不影响：
>
> | | 本镜像（`docker/kvmem/`） | ninfer 镜像（`docker/`） |
> |---|---|---|
> | 引擎 | `kvmem/kvmem-llama.cpp` Bonsai 分支（三元内核 + KVMem） | ninfer（交付包引擎树） |
> | 架构 | **sm_86（RTX 30 系）**；同分支官方也编过 89/120a（本机未验证，可用 `-a` 试） | sm_89 / sm_120a |
> | 上下文 | **128K~256K**（历史 KV 在主机内存，GPU 只驻留窗口） | 24K~112K（12GB 卡） |
> | 出字 | ~34 t/s（MTP1 39~41） | ~109 t/s |
> | 内置 | 视觉塔开关、MTP 开关、**上下文自动压缩代理**、客户端兼容模板 | — |

## 1. 拿到镜像：两条路（**先选这条**）

> **① 直接用 CI 构建好的镜像 —— 不构建，推荐给其他机器**（跳到下面的
> 「不自己构建：直接用 CI 构建好的镜像（GHCR）」：`docker pull ghcr.io/<owner>/llama-kvmem-bonsai:cuda13.3-arch86`）
>
> **② 自己构建 —— 只在两种情况下需要**：
> - 你要**改引擎源码**（vendor 树里的东西）
> - 你要在**本机验证这张镜像**（CI 只能证明"编得出来"，**证明不了"跑得起来"** ——
>   CI runner 没有显卡；今天 3 个运行期 bug 全靠本机构建 + 冒烟才抓到）

### 路线 ②：自己构建

```bash
./docker/build-kvmem-image.sh                  # 默认 arch 86 + CUDA 13.3.0 → llama-kvmem-bonsai:cuda13.3-arch86
./docker/build-kvmem-image.sh -m 1             # 国内构建（apt 走清华源）
./docker/build-kvmem-image.sh -j 3             # nvcc 并行度（默认 6）
./docker/build-kvmem-image.sh -b nvidia/cuda:<13.2.x>-devel-ubuntu24.04   # 贴合作者工具链基线（tag 以 Docker Hub 为准）
```

- **国内构建建议**：`./docker/build-kvmem-image.sh -m 1 -n host`
  —— `-m 1` 让容器内 apt 走清华源；`-n host` 让**构建容器与宿主同网络**，这样宿主机代理
  （如 `127.0.0.1:10808`）才可达、克隆 GitHub 源码才稳（脚本会自动透传 `HTTPS_PROXY`，也可 `-p` 显式指定）。
  docker daemon 侧建议配 registry mirror 加速拉基础镜像。
- **本机已实测**：`-a 86 -n host -m 1` 构建通过 → `llama-kvmem-bonsai:cuda13.3-arch86`，**2.25 GB**；
  镜像在 `--gpus all` 下引擎能加载运行，入口脚本缺模型时给出明确报错。
- 构建期会**克隆** `kvmem-llama.cpp`（分支 `kvmem-bonsai-llama.cpp`）与其 `llama.cpp` 子模块，
  revision 已 pin 在 Dockerfile 里（可复现）；需能访问 GitHub。
- 首次约 **10~25 分钟**（含拉取 CUDA devel 基础镜像 ~7 GB）；产物镜像约 **2~4 GB**。
- CI 也会自动构建并推送：`.github/workflows/docker-build-kvmem.yml` → `ghcr.io/<owner>/llama-kvmem-bonsai`。
  首次拉包前要在 GitHub 网页把该包设为 Public（或在本地 `docker login ghcr.io`）。

### 不自己构建：直接用 CI 构建好的镜像（GHCR）

推送到 `main` 且改动命中 `docker/kvmem/**` 等路径时，CI 会自动构建并推送：

```
ghcr.io/<owner>/llama-kvmem-bonsai:cuda13.3-arch86   ← RTX 30 系（sm_86）
ghcr.io/<owner>/llama-kvmem-bonsai:latest
```

> `<owner>` = **构建那次 CI 的仓库所有者**（workflow 里是 `${{ github.repository_owner }}`）。
> 所以：本仓库（`ygxiuming/ninfer-bonsai-4070ti`）发布在 `ghcr.io/ygxiuming/llama-kvmem-bonsai`；
> 如果你是从自己的 fork 上构建的，就把 `<owner>` 换成你自己的用户名。

```bash
# ① 让 daemon 能拉私有包（任选其一）
#    a) 建一个 classic PAT 勾上 read:packages，然后登录：
echo "<你的PAT>" | docker login ghcr.io -u "<owner>" --password-stdin
#    b) 或在 GitHub 网页把该包设为 Public：Package settings → Change visibility

# ② 拉取
docker pull ghcr.io/<owner>/llama-kvmem-bonsai:cuda13.3-arch86

# ③ 起服务（模型只读挂载；端口与宿主脚本一致）
docker run -d --name kvmem-3060 --gpus all --restart unless-stopped \
  -p 127.0.0.1:18210:18210 -p 127.0.0.1:18202:18202 \
  -v "$PWD/models:/models:ro" \
  ghcr.io/<owner>/llama-kvmem-bonsai:cuda13.3-arch86
# 或把 docker-compose.kvmem.yml 里的 image: 换成上面的 GHCR 地址，再 compose up -d
```

> **用 compose（含 VS Code 的 Docker 面板点 Compose Up）时，不用改文件**：
> 在 `docker/kvmem/` 下放一个 `.env`，写一行即可（compose 会自动读同目录 `.env`，已验证）：
>
> ```
> KVMEM_IMAGE=ghcr.io/<owner>/llama-kvmem-bonsai:cuda13.3-arch86
> ```
>
> 不写就默认用本机构建出来的 `llama-kvmem-bonsai:cuda13.3-arch86` ✓。
>
> ⚠️ 注意：**不要只对镜像点 "Run"** —— 那样不会带模型挂载、端口、GPU 直通，容器会
> 启动即退出（"缺模型"）或起来了也连不上。**起服务请走 compose 或上文的 `docker run` 完整命令。**

**"拉下来"不等于"直接能用"** —— 三件前置：

| 前置 | 说明 |
|---|---|
| **模型**（约 6 GB） | 镜像**不含权重**：把 `.gguf` 放进 `models/` 并挂载到 `/models`（与 ninfer 线同设计） |
| **NVIDIA 驱动 + nvidia-container-toolkit** | 镜像带 CUDA **用户态库**（cudart/cublas），但显卡与驱动来自宿主 —— 容器变不出显卡 |
| **架构必须匹配** | `arch86` 只能跑 **RTX 30 系**；40 系要 `arch89`、50 系要 `arch120a`。不匹配会在运行期报 `cudaErrorSymbolNotFound` / `no kernel image is available for execution on the device` |
| 端口 18202/18210 空闲 | 与宿主脚本互切时要等约 60 秒（TIME-WAIT，见 §6 排错） |

> 国内拉 `ghcr.io` 常慢或超时：给 docker daemon 配代理
> （`/etc/systemd/system/docker.service.d/http-proxy.conf` 里写 `HTTPS_PROXY=http://127.0.0.1:<端口>`，
> 再 `systemctl daemon-reload && systemctl restart docker`）；或直接在本机自建（§1，约 15 分钟）。

## 2. 放模型（镜像不含权重）

把下面文件放进**仓库的 `models/`**（compose 会只读挂载到容器 `/models`）：

| 文件 | 必需 | 说明 |
|---|---|---|
| `Ternary-Bonsai-2-27B-PTQ1_0.gguf`（5.95 GB） | ✅ | 主模型（魔搭/HF：`prism-ml/Ternary-Bonsai-2-27B-gguf`） |
| `Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf`（6.19 GB） | 可选 | 合并了 MTP 头 → 出字 +15~22%（`patches/` 里有 graft 脚本说明） |
| `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`（0.63 GB） | 可选 | 视觉塔 → 能贴图 |

**文件在同目录就自动启用**（`KVMEM_MTP=0` / `KVMEM_VISION=0` 可分别关掉）。

**一条命令下载主模型**（走 hf-mirror 国内直连，实测 ~35 MB/s；SHA256 已与 HF 核对）：

```bash
curl -L --noproxy '*' -o models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  https://hf-mirror.com/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0.gguf
sha256sum models/Ternary-Bonsai-2-27B-PTQ1_0.gguf   # 期望 53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3
```

可选两个（**放进同一个目录即自动启用**）：

```bash
# MTP 合并版（+15~22% 出字；合并方法见仓库 patches/）
curl -L --noproxy '*' -o models/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf \
  https://hf-mirror.com/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-PTQ1_0-MTP-r3-Q4_0.gguf 2>/dev/null || true
# 视觉塔（能贴图）
curl -L --noproxy '*' -o models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  https://hf-mirror.com/prism-ml/Ternary-Bonsai-2-27B-gguf/resolve/main/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf
```

> MTP 合并版是用 `scripts`+`patches` 里的 graft 流程现场生成的（HF 上**没有**这个合并文件）⇒
> 从零复现时先下**主模型**，再按仓库 `patches/` 的说明合并；只想直接跑就只下主模型即可。

## 3. 运行

```bash
docker compose -f docker/kvmem/docker-compose.kvmem.yml up -d
docker compose -f docker/kvmem/docker-compose.kvmem.yml logs -f      # 首次加载约 10~30 s
```

客户端（ZCode / Chatbox / Cherry Studio 等）填：

| 项 | 值 |
|---|---|
| Base URL | **`http://127.0.0.1:18210/v1`**（走压缩代理；`18202` 是引擎直连，调试用） |
| API Key | 任意 |
| 模型名 | 任意 |

裸跑（不用 compose）：

```bash
docker run --rm -it --gpus all \
  -p 127.0.0.1:18210:18210 -p 127.0.0.1:18202:18202 \
  -v "$PWD/models:/models:ro" \
  llama-kvmem-bonsai:cuda13.3-arch86
```

## 4. 常用环境变量（与宿主机 `scripts/start-kvmem.sh` 同名同义）

| 变量 | 默认 | 说明 |
|---|---|---|
| `KVMEM_CTX` | 131072 | 逻辑上下文；长会话用 `262144`（KV 在主机内存，显存不变） |
| `KVMEM_RESERVE` | 10240 | 单轮输出预留 = **单轮输出上限**（含思考 token） |
| `KVMEM_BUDGET` | 24576 | GPU 检索窗口 |
| `KVMEM_KV` | q8_0 | KV 量化档（**30 系没有 FP8 张量核，不要 fp8**） |
| `KVMEM_MTP` / `KVMEM_DRAFT_N` | 1 / 1 | MTP 投机；`DRAFT_N=2` 在 3060 上实测更慢 |
| `KVMEM_VISION` / `KVMEM_VISION_CPU` | 1 / 0 | 视觉塔；`CPU=1` 编码器放 CPU 省显存 |
| `KVMEM_PROXY` / `KVMEM_PROXY_THRESHOLD` | 1 / 45000 | 上下文自动压缩代理与其触发阈值 |

> ⚠️ **提示长度硬上限** = `KVMEM_CTX − KVMEM_RESERVE`（131072−10240 = 120832）。
> 超了引擎会**秒拒** `prompt + max_tokens exceeds n_ctx`；压缩代理会在这之前先把早期对话摘要掉。

## 5. 验证镜像（一键自检）

```bash
./docker/kvmem/verify.sh                      # 默认检查 llama-kvmem-bonsai:cuda13.3-arch86
```

**不占显存**（容器里用 `KVMEM_NGL=0` 在 CPU 上跑整条链路），约 2 分钟。前提是本机装了 NVIDIA 驱动
（容器要 `--gpus all` 注入驱动，只是不需要为模型留出显存）。八项检查：

| # | 检查 | 为什么 |
|---|---|---|
| ① | 镜像 ENV 不含 `KVMEM_MODEL` | 含了会让入口脚本误判"用户指定了模型"，**跳过自动切 MTP 模型**（实测踩过） |
| ② | 引擎二进制能加载（`--help`） | CUDA 运行库 + 驱动注入是否配齐 |
| ③ | 缺模型时报错清晰 | 挂载路径配错时能立刻看出来 |
| ④⑤ | 引擎端口 / 压缩代理端口 `/v1/models` 均 200 | 端口与代理绑定（容器内需 0.0.0.0） |
| ⑥⑦ | 横幅显示 `视觉=开` / `投机=开(MTP1)` | 可选件是否真的挂上（且 MTP 换用了合并模型） |
| ⑧ | 经代理真实生成一句话 | 端到端链路（分词→生成→API→代理） |

> CPU 生成一句约 1~2 分钟（GPU 上几秒）。**唯一不在自检范围**的是 CUDA 内核执行路径 ——
> 那需要显存，用下面的 GPU 冒烟测。

**GPU 冒烟**（需要腾出约 7 GB 显存：先 `./scripts/start-kvmem.sh stop`）：

```bash
docker run --rm --gpus all -p 127.0.0.1:18202:18202 \
  -v "$PWD/models:/models:ro" llama-kvmem-bonsai:cuda13.3-arch86
# 另开一个终端：
curl -s --noproxy '*' http://127.0.0.1:18202/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":64}'
# 看返回里的 timings.predicted_per_second —— 3060 上应在 30 t/s 量级
```

## 6. 排错
| 症状 | 处置 |
|---|---|
| 起容器就退，日志 `缺模型: /models/…` | 模型没挂对；确认 `models/*.gguf` 在仓库 `models/` 下，或自己 `-v <目录>:/models:ro` |
| `docker: Error response … could not select device driver ""` | 宿主没装 nvidia-container-toolkit |
| 客户端整轮 400 `Unexpected reasoning effort high` | 镜像自带兼容模板（`KVMEM_TEMPLATE`），若被覆盖请检查该文件是否存在 |
| 客户端整轮 400 `System message must be at the beginning` | 同上（兼容模板的补丁 2） |
| 看着"没输出"、日志在跑 `prompt processing … progress=` | 在预填充，等它跑完；**中途不要重发**（会从头开始） |
| 出字一块一块蹦 | 压缩代理的 SSE 转发必须用 `read1`（镜像内已修；改代码时别改回落 `read`） |
| 容器与宿主**互相切换**后，后起的一方报 `cannot bind 127.0.0.1:18202` | 上一位占用者的连接还在 **TIME-WAIT**（引擎用 httplib，未开 SO_REUSEADDR）⇒ **等约 60 秒**再起，或让两者用不同端口（容器可映射到 18302/18310） |

引擎与代理日志：`docker compose … logs -f`（代理的摘要状态在 `kvmem-proxy-state` 卷里，
删卷即重置记忆）。
