# NOTICE — 出处与第三方声明

本仓库是**部署实录与教程**，不分发任何模型权重。`scripts/` 与 `docker/` 下的脚本为本仓库原创（Apache-2.0）。
引擎源码与指南工具链以 Apache-2.0 **随仓库分发**于 `vendor/`（保留原 LICENSE/NOTICE 文件于其目录内）：

- `vendor/ninfer-4090w-ternary/` —— 引擎源码树（源自官方交付包 src-tree，剔除了 Windows 专用 ffmpeg/ 二进制目录）
- `vendor/shensanshu-guide/` —— 魔搭指南的工具链副本（pack.py / verify 套件 / changed-files 补丁）

所有被引用的组件归属原作：

## 模型

- **Bonsai-2-27B**（三元量化，PQ2_0 / PTQ1_0）
  © 2026-present Prism ML, Inc. — Apache-2.0
  来源：https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
  官方请求署名：**"Created using Bonsai by Prism ML."**
- **Qwen3.8-27B**（基座，亦为 MTP 头来源）
  © 2026 Alibaba Cloud — Apache-2.0
  来源：https://huggingface.co/Qwen/Qwen3.8-27B

## 引擎

- **NInfer**（上游规范实现，Linux + RTX 5090）
  © Neroued — Apache-2.0 — https://github.com/Neroued/ninfer
- **三元移植、打包器与验证工具链**
  shensanshu/ninfer-ada-ternary（魔搭）— Apache-2.0
- **Ada Linux 移植与性能调优**（本文引擎源码的来源之一）
  CraneBW/ninfer-ternary-bonsai-ada — Apache-2.0
- **KVMem 线**：naamfung/zatfung — Apache-2.0
- **官方交付包**（PQ2/PTQ1 制品与 src-tree）：作者离线交付，
  其引擎为 v1.0.8 线 dev fork（含未合并的上游改进）

## 第三方依赖（引擎编译时拉取/链接）

CUDA Toolkit（NVIDIA EULA）、ffmpeg 开发库（LGPL/GPL，系统包）、
spdlog / cpp-httplib / utf8proc（各随上游许可）——均以系统包或上游仓库原样使用。

## 本仓库脚本的使用说明

`scripts/` 内脚本针对 **RTX 4070 Ti 12GB + Ubuntu 26.04 + CUDA 13.3** 调优，
其他硬件请按 README §9 的速查表调整上下文与 KV 参数。
脚本不包含任何权重、密钥或个人信息；路径均为可在文件头修改的变量。
