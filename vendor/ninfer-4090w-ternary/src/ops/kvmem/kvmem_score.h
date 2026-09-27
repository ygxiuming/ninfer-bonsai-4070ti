#pragma once

// KVMem 选择探针（**最小切片**）· 只做三件事：按 query 打分 → 选块 → 打一行证据日志
//
// 【它是什么】把已经在库里的三件套接上主路径的**第一刀**：
//     kvmem_retrieve（官方 KVMem 论文 Eq.10 的 query-conditioned 打分器）
//   → kvmem_select  （host：sink/recent 恒留 + 中段按分降序）
//   → 一行 fprintf（对齐官方实现的三口径：selected 非连续 / skip>0 / window≈budget）
//
// 【它不是什么】**不碰页表、不碰 cache position、不碰 attention envelope、不 arm 窗口**。
//   ⇒ 可见集不变、OFF 臂逐位不变、**不需要 re-RoPE**（那属于第二刀：见 kvmem_window.h 的 arming 前置）。
//   ⇒ 因此本探针**不会让任何答案变对或变错**，它只回答"选择器选得对不对"。
//
// 【开关】NINFER_TERNARY_KVMEM_SCORE=1（默认关）。OFF 臂：不分配、不启动、不读索引、不写日志。
//   ⚠️ 开关与 env 只在这个头文件里读（与 kvmem_shadow.h 同一条纪律：开关的读取点必须唯一）。
//
// 【为什么只有打分器需要新代码】索引（mean_k_index，prefill 时按 chunk 喂）、raw-K 收割
//   （raw_k_harvest）、窗口载体（kvmem_window）**都已经接在引擎里**；缺的正是中间那句"按分选块"。
//
// 【两处必须记住的接缝】
//   1. 索引存的是每块 key 的 **和**（不是 mean）⇒ 打分器要用 block_tokens 自己除；而索引里的计数是
//      **F32**，打分器要 **int32** ⇒ 这里在 host 侧按 blocks_written/tail_fill 造 int32 数组
//      （尾部不满的块必须带**真实填充**，不能带名义块大小 —— mean_k_index.h:127-131）。
//   2. query 必须是**内容帧**（de-RoPE 之后的位置无关帧），与索引同帧 ⇒ 引擎里那个可用的点只有
//      `qn`（RMS 归一化后、`ops::rope` 覆盖之前）。这就是为什么 accumulate() 的调用点在 rope 之前。
//
// 【已知边界（写死在代码里，别当没看见）】
//   * **多 lane 不安全**：mean-K 索引是**全进程一份**（mean_k_index.h:248-251 自陈），本探针的
//     score 缓冲同样是全进程一份 ⇒ 只在单路（--max-concurrency 1）下有意义。多路时必须 per-lane。
//   * 只对"短 query span"打分（≤ NINFER_TERNARY_KVMEM_SCORE_MAXQ，默认 256 个 token）：文档
//     ingest 那种几万 token 的 chunk 打分代价是 O(块数×query长)，会拖死 prefill；而且那种 chunk
//     的"query"本来也不是检索 query。
//   * ★ 2026-09-26（语义档）：NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL=N ⇒ 只把 chunk 的最后 N 个
//     token 当检索 query 打分（官方 kvmem_set_query_span 的等价物：query 是"问"，不是整段
//     ingest），MAXQ 的门限看 **span 长度**、不看 chunk 长度 —— 解掉"探针只在短 query chunk
//     上打分（skip long chunk tokens=1024 > MAXQ=256）"那块拦路石。默认 0 = 旧行为（逐位不变）。
//   * 本探针**不改可见集** ⇒ 它与"答案对不对"无关；要验证"被选中 ⇒ 真的进了 softmax"必须等第二刀。

#include "ops/kvmem/kvmem_retrieve_launch.h"
#include "ops/kvmem/kvmem_select.h"
#include "ops/kvmem/mean_k_index.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// ---- 开关（唯一读取点；env 只读一次，与 kvmem_shadow_enabled() 同形）---------------------------
inline bool kvmem_score_enabled() noexcept {
    static const bool on = [] {
        const char* v = std::getenv("NINFER_TERNARY_KVMEM_SCORE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

inline std::int32_t kvmem_score_env_i32(const char* name, std::int32_t fallback) noexcept {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') { return fallback; }
    const long parsed = std::strtol(v, nullptr, 10);
    return parsed > 0 ? static_cast<std::int32_t>(parsed) : fallback;
}

// ---- 探针状态（进程级单例，与 mean-K 索引同寿命）---------------------------------------------
struct KvMemScoreProbe {
    // device 缓冲
    float*        score            = nullptr;  // [capacity_blocks]，每 chunk 前清零（打分器只 ADD）
    std::int32_t* block_tokens_dev = nullptr;  // [capacity_blocks]，尾块带真实填充（device 侧）
    // host staging
    std::vector<float>        score_host;
    std::vector<std::int32_t> tokens_host;

    std::int32_t capacity_blocks = 0;
    std::int32_t block_tokens    = 0;      // == kPagedKVPageSize，由调用方传入（house rule：不许写死）
    std::int32_t chunk_tokens    = 0;      // live 平面的 token 行数（= q_layer_stride 口径）
    std::int32_t n_blocks        = 0;      // 本 chunk 打分的"历史"块数（= 打分开始时的 blocks_written）
    // ---- ★ 分数的**出处**（2026-09-26 语义档要用：静默用错查询/错时刻 = 静默错答案）------------
    std::int32_t query_span_begin  = 0;    // 本次 query span 在 live 平面里的起点（QUERY_TAIL 档）
    std::int32_t query_span_tokens = 0;    // 本次真正打分的 query token 数（= span 长度）
    std::int32_t scored_n_blocks   = 0;    // 这份分数描述块 [0, scored_n_blocks)（finish 时登记）
    bool scored_last_chunk = false;        // 最近一个 prefill chunk 真打过分（语义档的准入闸）
    std::int32_t layers_total    = 0;
    std::int32_t n_heads         = 0;
    std::int32_t n_kv_heads      = 0;
    std::int32_t head_dim        = 0;
    bool armed    = false;                 // 本 chunk 正在打分
    bool scored_this_chunk = false;        // 本 chunk 真的累加过（否则 finish 打出来的是全零假日志）
    bool failed   = false;                 // 出过致命几何/分配错 ⇒ 之后一律不打分（但引擎继续跑）
    bool logged_once = false;
};

inline KvMemScoreProbe& kvmem_score_probe() {
    static KvMemScoreProbe probe;
    return probe;
}

inline void kvmem_score_fail(const char* why) noexcept {
    KvMemScoreProbe& p = kvmem_score_probe();
    p.failed = true;
    p.armed  = false;
    std::fprintf(stderr, "kvmem_score: DISABLED (%s)\n", why);
}

// ---- ① chunk 开始：确保容量 + 清零 + 记下"历史块数"--------------------------------------------
//
// history_blocks 取**打分开始那一刻**的 blocks_written()：本 chunk 自己还没进索引（索引在 chunk
// 之后才 append_round），所以它天然是"历史"——这正是检索该对的东西。
inline void kvmem_score_begin(std::int32_t requested_capacity, std::int32_t history_blocks,
                              std::int32_t tail_fill, std::int32_t block_tokens,
                              std::int32_t layers_total, std::int32_t n_heads,
                              std::int32_t n_kv_heads, std::int32_t head_dim,
                              cudaStream_t stream) noexcept {
    if (!kvmem_score_enabled()) { return; }
    KvMemScoreProbe& p = kvmem_score_probe();
    if (p.failed) { return; }

    if (requested_capacity <= 0 || history_blocks <= 0 || block_tokens <= 0) { p.armed = false; return; }

    if (p.score == nullptr || p.capacity_blocks < requested_capacity) {
        // 只在**首个 chunk** 分配；prefill 不在 CUDA graph 捕获期（捕获期禁 cudaMalloc，坑表 §3.7）
        if (p.score != nullptr) { (void)cudaFree(p.score); p.score = nullptr; }
        if (p.block_tokens_dev != nullptr) { (void)cudaFree(p.block_tokens_dev); p.block_tokens_dev = nullptr; }
        const std::size_t bytes = static_cast<std::size_t>(requested_capacity) * sizeof(float);
        if (cudaMalloc(reinterpret_cast<void**>(&p.score), bytes) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&p.block_tokens_dev),
                       static_cast<std::size_t>(requested_capacity) * sizeof(std::int32_t)) !=
                cudaSuccess) {
            kvmem_score_fail("cudaMalloc");
            return;
        }
        p.capacity_blocks = requested_capacity;
        p.score_host.assign(static_cast<std::size_t>(requested_capacity), 0.0F);
        p.tokens_host.assign(static_cast<std::size_t>(requested_capacity), 0);
        std::fprintf(stderr, "kvmem_score: ARMED capacity_blocks=%d layers_total=%d heads=%d kv_heads=%d "
                             "head_dim=%d\n", requested_capacity, layers_total, n_heads, n_kv_heads, head_dim);
    }

    p.n_blocks     = history_blocks;
    p.block_tokens = block_tokens;
    p.layers_total = layers_total;
    p.n_heads      = n_heads;
    p.n_kv_heads   = n_kv_heads;
    p.head_dim     = head_dim;
    p.armed        = true;
    p.scored_this_chunk = false;
    p.scored_last_chunk = false;

    // block_tokens：尾块（blocks_written-1）用真实填充，其余用名义块大小
    for (std::int32_t i = 0; i < history_blocks; ++i) {
        p.tokens_host[static_cast<std::size_t>(i)] =
            (i == history_blocks - 1) ? (tail_fill > 0 ? tail_fill : block_tokens) : block_tokens;
    }
    if (cudaMemcpyAsync(p.block_tokens_dev, p.tokens_host.data(),
                        static_cast<std::size_t>(history_blocks) * sizeof(std::int32_t),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        kvmem_score_fail("block_tokens H2D");
        return;
    }
    if (cudaMemsetAsync(p.score, 0, static_cast<std::size_t>(history_blocks) * sizeof(float), stream) !=
        cudaSuccess) {
        kvmem_score_fail("score memset");
        return;
    }
}

// ---- ② 每层一次：把该层的 query 行累加到 score 上 ----------------------------------------------
//
// 调用点必须在 `ops::rope(qn, ...)` **之前**：rope 会原地覆盖 qn，之后就没有内容帧的 query 了。
inline void kvmem_score_accumulate(std::int32_t fidx, const void* q, std::int32_t n_query_tokens,
                                   std::int32_t chunk_tokens, cudaStream_t stream) noexcept {
    if (!kvmem_score_enabled()) { return; }
    KvMemScoreProbe& p = kvmem_score_probe();
    if (!p.armed || p.failed) { return; }
    if (n_query_tokens <= 0 || p.n_blocks <= 0) { return; }

    // ★ query span（2026-09-26 语义档）：NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL=N ⇒ 只把本 chunk 的
    //   **最后 N 个 token** 当检索 query（官方 kvmem_set_query_span 的等价物）。默认 0 = 整 chunk
    //   （旧行为，逐位不变）。MAXQ 的门限看 **span 长度**、不看 chunk 长度。
    const std::int32_t query_tail = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL", 0);
    std::int32_t span_begin  = 0;
    std::int32_t span_tokens = (chunk_tokens > 0 && chunk_tokens < n_query_tokens) ? chunk_tokens
                                                                                  : n_query_tokens;
    if (query_tail > 0 && chunk_tokens > 0) {
        const std::int32_t tail = query_tail < chunk_tokens ? query_tail : chunk_tokens;
        span_begin  = chunk_tokens - tail;
        span_tokens = tail;
    }
    const std::int32_t maxq = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_MAXQ", 256);
    if (span_tokens > maxq) {   // query 太长：不打分（它也不是检索 query）
        if (!p.logged_once) {
            p.logged_once = true;
            std::fprintf(stderr, "kvmem_score: skip long query span tokens=%d (> MAXQ=%d); chunk=%d begin=%d\n",
                         span_tokens, maxq, chunk_tokens, span_begin);
        }
        return;
    }
    p.chunk_tokens      = chunk_tokens;   // live 平面的行数（= q_layer_stride）
    p.query_span_begin  = span_begin;
    p.query_span_tokens = span_tokens;    // 供 finish 的 oracle 自检用（正确口径 = span 的 token 数）

    ops::MeanKIndex* index = ops::mean_k_index_for(p.layers_total, p.n_kv_heads, p.head_dim,
                                                  p.capacity_blocks);
    if (index == nullptr || index->blocks_written() <= 0) { return; }
    if (fidx < 0 || fidx >= index->layers()) { return; }

    const Tensor& sums = index->layer_sums(fidx);
    if (sums.data == nullptr) { return; }

    KvMemRetrieveConfig cfg;
    cfg.n_layers          = 1;                      // 逐层调用
    cfg.n_layers_total    = p.layers_total;         // head_w = 1/(总层数×query 头数)
    cfg.n_query_tokens    = span_tokens;
    cfg.n_heads           = p.n_heads;
    cfg.n_kv_heads        = p.n_kv_heads;
    cfg.head_dim          = p.head_dim;
    cfg.n_blocks          = p.n_blocks;
    cfg.q_layer_stride    = chunk_tokens;           // live plane 的 token 数（>= n_query_tokens）
    cfg.kbar_layer_stride = p.capacity_blocks;      // >= n_blocks
    cfg.q_token_begin     = span_begin;
    cfg.budget_blocks     = 0;                      // 掩码只用在这里；探针不设带（Official 规则见下）
    cfg.sink_blocks       = 0;
    cfg.recent_blocks     = 0;
    cfg.mask_mode         = KvMemRetrieveMaskMode::Never;   // 探针要"全历史"的原始分数，不套带掩码
    cfg.dtype             = KvMemRetrieveDtype::BF16;

    const cudaError_t status =
        ops::kvmem_retrieve_scores(cfg, p.score, q, static_cast<const float*>(sums.data),
                                   p.block_tokens_dev, stream);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "kvmem_score: launch failed layer=%d: %s\n", fidx, cudaGetErrorString(status));
        kvmem_score_fail("launch");
        return;
    }
    p.scored_this_chunk = true;
}

// ---- ③ chunk 收尾：D2H + 选块 + 一行证据日志 ---------------------------------------------------
//
// 放在与本文件同类的位置：prefill chunk 的尾巴、**任何捕获之外**（现有 kvmem_index 探针就在那儿
// 做 cudaMemcpyAsync + cudaStreamSynchronize，理由照抄它：一次同步换一个外部可核的数）。
//
// 免 oracle 自检：契约保证 `sum_b score[b] == n_query_tokens`（与带掩码与否无关）⇒ sum_score 就是
// 内置 oracle。对不上说明打分器没跑、跑了半截、或几何配错 —— 三种都得当场喊出来。
inline void kvmem_score_finish(const char* label, std::int32_t query_tokens,
                               cudaStream_t stream) noexcept {
    if (!kvmem_score_enabled()) { return; }
    KvMemScoreProbe& p = kvmem_score_probe();
    if (!p.armed || p.failed || p.n_blocks <= 0) { return; }
    p.armed = false;
    if (!p.scored_this_chunk) { return; }   // 本 chunk 没打过（如长 ingest）⇒ 不打印全零假日志
    // ★ 出处登记（2026-09-26 语义档的准入闸）：这份分数描述块 [0, n_blocks)，且来自**最近一个
    //   chunk 的 query**。语义档只在 scored_last_chunk 为真时才用它装窗（宁可退压力档，也不装
    //   一个来路不明的窗）。
    p.scored_n_blocks   = p.n_blocks;
    p.scored_last_chunk = true;
    const std::int32_t q_used = p.query_span_tokens > 0 ? p.query_span_tokens : p.chunk_tokens;

    // 选块的带（只影响这一行日志；本探针不改可见集 ⇒ 不改变引擎行为）。
    // 默认取"小预算"以便**逼出选择**：budget 32768 / sink 4096 / recent 0（官方那台仪器用的就是这组）。
    const std::int32_t budget_tokens = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_BUDGET", 32768);
    const std::int32_t sink_tokens   = kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_SINK", 4096);
    const std::int32_t recent_tokens = std::getenv("NINFER_TERNARY_KVMEM_SCORE_RECENT") != nullptr
                                           ? kvmem_score_env_i32("NINFER_TERNARY_KVMEM_SCORE_RECENT", 0)
                                           : 0;

    const std::size_t bytes = static_cast<std::size_t>(p.n_blocks) * sizeof(float);
    if (cudaMemcpyAsync(p.score_host.data(), p.score, bytes, cudaMemcpyDeviceToHost, stream) !=
        cudaSuccess) {
        kvmem_score_fail("score D2H");
        return;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        kvmem_score_fail("sync");
        return;
    }

    double sum_score = 0.0;
    for (std::int32_t i = 0; i < p.n_blocks; ++i) { sum_score += static_cast<double>(p.score_host[i]); }

    KvMemSelectConfig scfg;
    scfg.total_blocks  = p.n_blocks;
    scfg.budget_blocks = budget_tokens / p.block_tokens;
    scfg.sink_blocks   = sink_tokens / p.block_tokens;
    scfg.recent_blocks = recent_tokens / p.block_tokens;
    const KvMemSelectResult sel = ops::kvmem_select_blocks(scfg, p.score_host.data(), 0);

    // 选择结构的形状（这是判据的核心：非连续 ⇒ 真在选择）
    std::int32_t runs = 0;
    for (std::size_t i = 0; i < sel.kept.size(); ++i) {
        if (i == 0 || sel.kept[i] != sel.kept[i - 1] + 1) { ++runs; }
    }
    const std::int32_t skip = p.n_blocks - static_cast<std::int32_t>(sel.kept.size());
    const std::int32_t kept_first = sel.kept.empty() ? -1 : sel.kept.front();
    const std::int32_t kept_last  = sel.kept.empty() ? -1 : sel.kept.back();

    // ★ 把**选中的块号本身**打出来：没有它就无法核对"含针的块有没有被选中"（聚合量 runs/skip 做不到）。
    //   与官方实现同形（它也是逐行 `KVMEM_TRACE selected <ids...>`）。只在短 query chunk 上出现。
    {
        std::fprintf(stderr, "kvmem_score: KEPT");
        for (std::size_t i = 0; i < sel.kept.size(); ++i) {
            std::fprintf(stderr, " %d", static_cast<int>(sel.kept[i]));
        }
        std::fprintf(stderr, "\n");
    }

    std::fprintf(stderr,
                 "kvmem_score: SELECT label=%s n_blocks=%d budget_blocks=%d sink_blocks=%d "
                 "recent_blocks=%d kept=%zu runs=%d skip=%d window_tokens=%zu "
                 "sink_kept=%d recent_kept=%d scored_kept=%d candidates=%d kept_range=[%d,%d] "
                 "sum_score=%.3f query_tokens=%d oracle_ok=%d scale_floor=%.6f\n",
                 label, p.n_blocks, scfg.budget_blocks, scfg.sink_blocks, scfg.recent_blocks,
                 sel.kept.size(), runs, skip,
                 sel.kept.size() * static_cast<std::size_t>(p.block_tokens), sel.sink_kept,
                 sel.recent_kept, sel.scored_kept, sel.candidates, kept_first, kept_last, sum_score,
                 q_used, (q_used > 0 && sum_score > 0.5 * q_used &&
                                  sum_score < 2.0 * q_used)
                                     ? 1
                                     : 0,
                 static_cast<double>(ops::kvmem_retrieve_scale(p.head_dim)));
}

// 逐块分数导出（给离线判据用：核对"含针的块"有没有被选中）。只在 switch 打开且本 chunk 打过时写。
inline void kvmem_score_dump_if_requested(const char* label) noexcept {
    if (!kvmem_score_enabled()) { return; }
    const char* path = std::getenv("NINFER_TERNARY_KVMEM_SCORE_DUMP");
    if (path == nullptr || path[0] == '\0') { return; }
    KvMemScoreProbe& p = kvmem_score_probe();
    if (p.n_blocks <= 0) { return; }
    FILE* f = std::fopen(path, "ab");
    if (f == nullptr) { return; }
    std::fprintf(f, "# %s n_blocks=%d\n", label, p.n_blocks);
    for (std::int32_t i = 0; i < p.n_blocks; ++i) {
        std::fprintf(f, "%d %.6f\n", i, static_cast<double>(p.score_host[static_cast<std::size_t>(i)]));
    }
    std::fclose(f);
}

} // namespace ninfer::ops::detail
