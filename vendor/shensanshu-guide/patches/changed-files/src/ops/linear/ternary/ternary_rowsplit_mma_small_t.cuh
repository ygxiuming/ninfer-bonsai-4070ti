// PART OF the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// NEW file, not present upstream; see patches/ in the release bundle for the change list,
// rebuild steps and required verification. Roll back with NINFER_TERNARY_MMA=0.
#pragma once

// Ternary PQ2_0 tensor-core path for the small-T regime (T = 2..8) -- which is exactly the
// speculative VERIFY pass, T = draft + 1.
//
// WHY THIS EXISTS (measured, not assumed)
// The SIMT tile GEMV is INSTRUCTION-ISSUE bound at T > 1, not bandwidth bound. Its register
// footprint is 36 (tile<4>) against 40 for the T=1 GEMV, so occupancy is NOT the constraint --
// the handover's "extra accumulators cost registers and dropped the resident warp count" story
// does not hold. What actually scales with T is work per weight: one FMA per weight PER TOKEN,
// plus two activation loads and a bf16->f32 convert per token. Measured on a synthetic PQ2_0
// payload at N=248320, K=5120:
//
//     T=1   gemv      0.652 ms   483 GiB/s     (the plain warp-per-row GEMV)
//     T=3   gemv_tile 1.493 ms   211 GiB/s     (2.29x the T=1 time, identical weight bytes)
//     T=3   reference 5.402 ms    60 GiB/s     (amortises well but 8x slower in absolute terms)
//
// So the tile GEMV does reuse each weight across the tile, but pays a per-token instruction bill
// that eats the entire gain. The tensor core removes that bill: one mma.m16n8k16 covers
// 16 rows x 8 tokens x 16 k in ONE instruction, so every token in the tile rides the same weight
// read at zero extra instruction cost. Weights are decoded once per (row, k-byte) through a
// 256-entry shared LUT, which turns "one packed byte -> four ternary weights" into one LDS.64.
//
// LAYOUT NOTES THAT ARE LOAD-BEARING (violate them and T == 1 still looks perfect):
//   * ninfer/ggml keep ne[0] contiguous, so an activation with ne=(k,T) is TOKEN-major: element
//     (column, token) lives at token*k + column. The mma B operand wants n rows of k with k
//     contiguous, so the activation tile is staged verbatim, with no transpose.
//   * the output is token-major as well: (row, token) at token*out_row_stride + row.
//   * PQ2_0 packs FOUR 2-bit codes per byte, and code c denotes weight value (c - 1). The group
//     scale is applied AFTER the mma, per row, because one group spans 8 mma k-steps.
//
// GEOMETRY: one warp owns 16 output rows and walks the whole K; the CTA is 8 warps, so a CTA
// covers 128 rows. Warps never cooperate, so there is no cross-warp reduction and no barrier
// beyond the shared staging handshake. K is consumed in 256-wide chunks (two PQ2_0 groups),
// double buffered with cp.async so the weight stream is not serialised behind the mma.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <string>

namespace ninfer::ops::detail {

inline constexpr int kTernaryMmaRowsPerWarp     = 16;
inline constexpr int kTernaryMmaTokens          = 8;
// Warps (and therefore rows) per CTA. This is the knob that decides how many CTAs a narrow
// projection launches: a PQ2_0 weight is tiled 16 rows per warp, so with 8 warps a N=5120 projection
// gets only 40 CTAs and leaves half of this card's 80 SMs idle, measuring 322 GiB/s instead of the
// 618 GiB/s the same kernel reaches on the 272-CTA MLP shape. Four warps doubles the CTA count
// (N=5120 -> 80 CTAs, one per SM) at the cost of fewer resident warps per SM, which the cp.async
// pipeline is there to absorb.
inline constexpr int kTernaryMmaWarpsPerCta     = 4;
inline constexpr int kTernaryMmaRowsPerCta      = kTernaryMmaRowsPerWarp * kTernaryMmaWarpsPerCta;
inline constexpr int kTernaryMmaGroupK          = PQ2RowSplitStorage::kGroupK;
// K consumed per staged chunk. Doubling this from 256 to 512 halves both the number of
// __syncthreads pairs (two per chunk) and the number of times each CTA re-stages its activation
// tile -- the two overheads that keep this kernel at ~400 GiB/s against the ~540 GiB/s the T=1
// GEMV reaches on the same weights. The cost is more shared memory per CTA, so it is an empirical
// question, not an obvious win.
inline constexpr int kTernaryMmaChunkK          = 512;
inline constexpr int kTernaryMmaGroupsPerChunk  = kTernaryMmaChunkK / kTernaryMmaGroupK;
inline constexpr int kTernaryMmaKStepsPerChunk  = kTernaryMmaChunkK / 16;
inline constexpr int kTernaryMmaCodesPerRow     = kTernaryMmaChunkK / 4; // four codes per byte
inline constexpr int kTernaryMmaRowStride       = 144;                   // 128 used + pad
inline constexpr int kTernaryMmaActStride       = kTernaryMmaChunkK + 8; // 16B-aligned, padded
inline constexpr int kTernaryMmaStages          = 2;
inline constexpr int kTernaryMmaThreads         = kTernaryMmaWarpsPerCta * 32;

static_assert((kTernaryMmaRowStride % 16) == 0, "cp.async needs a 16-byte aligned row start");
static_assert(kTernaryMmaRowStride >= kTernaryMmaCodesPerRow, "the row must hold a whole chunk");
static_assert((kTernaryMmaActStride % 8) == 0, "ldmatrix wants 8-element row starts");
static_assert((kTernaryMmaChunkK % kTernaryMmaGroupK) == 0, "chunks must be whole groups");
static_assert((kTernaryMmaChunkK % 16) == 0, "chunks must be whole mma k-steps");

union TernaryMmaPairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

// NINFER_TERNARY_MMA=0 forces the SIMT paths, which is the A/B switch for validating this kernel
// end to end on the real artifact.
//
// It is needed because the perplexity harness runs with --context 512, i.e. T = 512, which is far
// outside this kernel's 2..8 token window and therefore falls through to the reference kernel. A
// perplexity run with a window inside 2..8 IS covered here, so comparing the two values with the
// switch on and off is a direct numeric equivalence test of the tensor-core path on real weights
// rather than on a synthetic payload.
//
// Read once: the choice decides whether a kernel appears in the captured CUDA graph at all, so it
// must not change between graph construction and replay (same rule as the rotation switch).
[[nodiscard]] inline bool ternary_mma_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_MMA");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// Shared staging. The row strides are chosen so that the shared accesses in the hot loop are
// bank-conflict free, which is why they are not just the natural 64 / 256:
//   * codes: a whole k-step is one 4-byte word, and the four lanes of a quad read the SAME word,
//     so the only conflicts are between rows. With a 64-byte row stride the word offset is
//     16*row + kstep and rows collapse onto two banks; 80 bytes gives 20*row + kstep, whose
//     low five bits are distinct across the eight rows a single load touches.
//   * activations: ldmatrix pulls 16 bytes per row, so rows must start on distinct banks.
//     528 bytes = 132 words => 4 banks apart, which lands rows 0..7 on banks 0,4,...,28 exactly.
struct TernaryMmaStorage {
    uint2 lut[256];
    alignas(16) std::uint8_t codes[kTernaryMmaStages][kTernaryMmaWarpsPerCta]
                                  [kTernaryMmaRowsPerWarp][kTernaryMmaRowStride];
    alignas(16) __nv_bfloat16 act[kTernaryMmaStages][kTernaryMmaTokens][kTernaryMmaActStride];
    std::uint16_t scales[kTernaryMmaStages][kTernaryMmaWarpsPerCta][kTernaryMmaRowsPerWarp]
                        [kTernaryMmaGroupsPerChunk];
};

__global__ __launch_bounds__(kTernaryMmaThreads)
void ternary_pq2_mma_small_t_kernel(const __nv_bfloat16* __restrict__ x,
                                    const std::uint8_t* __restrict__ codes,
                                    const std::uint8_t* __restrict__ scales,
                                    __nv_bfloat16* __restrict__ out, std::int32_t rows,
                                    std::int32_t k, std::int32_t tokens,
                                    std::int32_t out_row_stride) {
    // STATIC shared memory, deliberately not `extern __shared__`: the footprint is a compile-time
    // constant (32 KB, under the 48 KB per-block static cap), and declaring an extern shared
    // VARIABLE OF A NAMED AGGREGATE type emits a device symbol that nvlink then cannot resolve
    // ("nvlink error : Undefined reference to 'ninfer::ops::detail::shared'"). Static also removes
    // the need for a cudaFuncSetAttribute opt-in and for a size at every launch site.
    __shared__ TernaryMmaStorage staging;
    auto& lut      = staging.lut;
    auto& codes_sh = staging.codes;
    auto& act_sh   = staging.act;
    auto& scale_sh = staging.scales;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2; // 0..7  -- mma row within the fragment
    const int lid  = lane & 3;  // 0..3  -- mma k pair / column pair index

    // ---- decode table: packed byte -> four ternary weights in bf16 ----------------------------
    // Every code c means value (c - 1), so the table holds exact signed integers and the multiply
    // is exact (no rounding: +-1 scaling in bf16 is a pure exponent/sign change).
    for (int i = tid; i < 256; i += kTernaryMmaThreads) {
        TernaryMmaPairBits low;
        TernaryMmaPairBits high;
        low.pair  = __floats2bfloat162_rn(static_cast<float>((i & 3) - 1),
                                         static_cast<float>(((i >> 2) & 3) - 1));
        high.pair = __floats2bfloat162_rn(static_cast<float>(((i >> 4) & 3) - 1),
                                          static_cast<float>(((i >> 6) & 3) - 1));
        uint2 entry;
        entry.x = low.bits;
        entry.y = high.bits;
        lut[i]  = entry;
    }

    const int row0 = static_cast<int>(blockIdx.x) * kTernaryMmaRowsPerCta;
    if (row0 >= rows) { return; }

    const std::int32_t groups_per_row = k / kTernaryMmaGroupK;
    const std::int64_t code_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * PQ2RowSplitStorage::kCodeBytesPerGroup;
    const std::int64_t scale_row_bytes =
        static_cast<std::int64_t>(groups_per_row) * PQ2RowSplitStorage::kScaleBytesPerGroup;
    const std::int32_t chunks = k / kTernaryMmaChunkK;

    const std::int64_t warp_row_base =
        static_cast<std::int64_t>(row0) + static_cast<std::int64_t>(warp) * kTernaryMmaRowsPerWarp;

    // Stage one chunk: this warp's 16x64 code window, its scales, and (cooperatively) the
    // CTA-wide activation tile. Out-of-range rows are zero-filled rather than read, so a partial
    // CTA never touches global memory past the end of the weight.
    //
    // `tok_base` selects which tile of kTernaryMmaTokens tokens this staging is for. A CTA walks
    // the whole sequence in such tiles, which is what lets one kernel serve both the 2..8 token
    // speculative verify pass and a full prefill, instead of falling back to the reference kernel
    // above 8 tokens -- the fallback that made prefill measure 55 tok/s against 412-437 for the
    // non-ternary engine.
    const auto stage = [&](int buf, int chunk, int tok_base) {
        const std::int64_t chunk_byte =
            static_cast<std::int64_t>(chunk) * kTernaryMmaCodesPerRow;
        const std::int64_t chunk_k = static_cast<std::int64_t>(chunk) * kTernaryMmaChunkK;

        // Codes: kTernaryMmaCodesPerRow bytes per row, moved as 16-byte copies filled round-robin
        // across the lanes. Written generically on purpose -- sizing this loop by hand for one
        // chunk width silently stages only PART of each row at another width, which presents as a
        // suspiciously fast kernel that produces garbage (all four correctness checks went to NaN).
        constexpr int kCopiesPerRow  = kTernaryMmaCodesPerRow / 16;
        constexpr int kCopiesPerWarp = kTernaryMmaRowsPerWarp * kCopiesPerRow;
#pragma unroll
        for (int copy = lane; copy < kCopiesPerWarp; copy += 32) {
            const int local_row           = copy / kCopiesPerRow;
            const int byte_off            = (copy % kCopiesPerRow) * 16;
            const std::int64_t global_row = warp_row_base + local_row;
            std::uint8_t* dst             = &codes_sh[buf][warp][local_row][byte_off];
            if (global_row < rows) {
                cp_async<16, Cache::cg>(
                    dst, codes + global_row * code_row_bytes + chunk_byte + byte_off);
            } else {
                store_vec<std::uint8_t, uint4>(dst, make_uint4(0u, 0u, 0u, 0u));
            }
        }

        { // scales: one entry per (row, group-in-chunk) -- 16 x 4 at a 512-wide chunk
            constexpr int kScaleEntries = kTernaryMmaRowsPerWarp * kTernaryMmaGroupsPerChunk;
#pragma unroll
            for (int entry = lane; entry < kScaleEntries; entry += 32) {
                const int local_row           = entry / kTernaryMmaGroupsPerChunk;
                const int group_in_chunk      = entry % kTernaryMmaGroupsPerChunk;
                const std::int64_t global_row = warp_row_base + local_row;
                std::uint16_t value           = 0u;
                if (global_row < rows) {
                    value = load_vec<std::uint16_t>(
                        scales + global_row * scale_row_bytes +
                        (static_cast<std::int64_t>(chunk) * kTernaryMmaGroupsPerChunk +
                         group_in_chunk) *
                            PQ2RowSplitStorage::kScaleBytesPerGroup);
                }
                scale_sh[buf][warp][local_row][group_in_chunk] = value;
            }
        }

        { // activation tile: 8 tokens x 256 k, staged cooperatively by the whole CTA.
            // Written as a stride loop rather than `tok = tid >> 5` so it stays correct for any
            // warp count: with 4 warps that expression would only ever cover tokens 0..3.
            // These offsets are in bf16 ELEMENTS, not bytes: x is a __nv_bfloat16* while the code
            // staging above is byte-based because `codes` is a std::uint8_t*. A byte offset here
            // over-advances the source window by 2x and walks past the end of each token's row,
            // which presents as "values plausible but wrong, NaN at the tail" rather than as a
            // clean bounds error.
            // 16 bytes (8 bf16) per cp.async, and a chunk is kTernaryMmaChunkK elements per token,
            // so the number of copies a lane issues scales with the chunk width -- at 512 each lane
            // covers an element offset and that offset + 256.
            for (int tok = warp; tok < kTernaryMmaTokens; tok += kTernaryMmaWarpsPerCta) {
                const int global_tok = tok_base + tok;
                // Tokens past the end of the sequence are zero-filled (src_bytes 0 reads nothing),
                // so the mma sees a defined activation and the guarded store drops the result.
                const int safe_tok = (global_tok < tokens) ? global_tok : tok_base;
                for (int elem = lane * 8; elem < kTernaryMmaChunkK; elem += 32 * 8) {
                    cp_async_zfill<16>(
                        &act_sh[buf][tok][elem],
                        x + static_cast<std::int64_t>(safe_tok) * k + chunk_k + elem,
                        (global_tok < tokens) ? 16 : 0);
                }
            }
        }
    };

    // Walk the sequence in tiles of kTernaryMmaTokens tokens.
    //
    // The weight window is re-staged for every tile, because accumulators for the whole sequence
    // would not fit in registers. That amortises each weight read across kTernaryMmaTokens tokens
    // -- the same factor the reference kernel gets from its 8-token tile, but through a kernel that
    // sustains ~400 GiB/s instead of ~60 GiB/s. Above 8 tokens this replaces a fallback that cost
    // prefill a factor of ~8 against the non-ternary engine.
    const int row_lo = row0 + warp * kTernaryMmaRowsPerWarp + gid;
    const int row_hi = row_lo + 8;

    for (int tok_base = 0; tok_base < tokens; tok_base += kTernaryMmaTokens) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;

        stage(0, 0, tok_base);
        cp_commit();

        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int buf = chunk & 1;

            if (chunk + 1 < chunks) {
                stage(buf ^ 1, chunk + 1, tok_base);
                cp_commit();
                cp_wait<1>(); // chunk `buf` has landed; chunk+1 may still be in flight
            } else {
                cp_wait<0>();
            }
            __syncthreads();

#pragma unroll 1
        for (int group_in_chunk = 0; group_in_chunk < kTernaryMmaGroupsPerChunk;
             ++group_in_chunk) {
            float g0 = 0.0f;
            float g1 = 0.0f;
            float g2 = 0.0f;
            float g3 = 0.0f;

#pragma unroll
            for (int step = 0; step < 8; ++step) {
                const int kstep    = group_in_chunk * 8 + step;
                const int word_off = kstep * 4; // one k-step of 16 codes == one 4-byte word

                // A operand: rows gid and gid+8, each contributing k and k+8 pairs.
                const unsigned word_lo =
                    load_vec<unsigned>(&codes_sh[buf][warp][gid][word_off]);
                const unsigned word_hi =
                    load_vec<unsigned>(&codes_sh[buf][warp][gid + 8][word_off]);
                const unsigned shift_near = 8u * static_cast<unsigned>(lid >> 1);
                const unsigned shift_far  = 8u * static_cast<unsigned>((lid >> 1) + 2);

                const uint2 near_lo = lut[(word_lo >> shift_near) & 0xFFu];
                const uint2 far_lo  = lut[(word_lo >> shift_far) & 0xFFu];
                const uint2 near_hi = lut[(word_hi >> shift_near) & 0xFFu];
                const uint2 far_hi  = lut[(word_hi >> shift_far) & 0xFFu];

                // A quad of lanes shares a byte; lanes with an odd lid need its high code pair.
                const bool odd = (lid & 1) != 0;
                const unsigned a0 = odd ? near_lo.y : near_lo.x;
                const unsigned a1 = odd ? near_hi.y : near_hi.x;
                const unsigned a2 = odd ? far_lo.y : far_lo.x;
                const unsigned a3 = odd ? far_hi.y : far_hi.x;

                // B operand: the activation tile is already n-major (token rows, k contiguous),
                // which is exactly the .col layout the mma wants -- no transpose needed.
                unsigned bf0 = 0u;
                unsigned bf1 = 0u;
                ldmatrix_x2(bf0, bf1,
                            smem_addr(&act_sh[buf][lane & 7]
                                            [kstep * 16 + ((lane >> 3) & 1) * 8]));

                mma_bf16(g0, g1, g2, g3, a0, a1, a2, a3, bf0, bf1);
            }

            // One PQ2_0 group spans eight k-steps, so its scale is applied once, per row.
            const float top = __half2float(
                __ushort_as_half(scale_sh[buf][warp][gid][group_in_chunk]));
            const float bottom = __half2float(
                __ushort_as_half(scale_sh[buf][warp][gid + 8][group_in_chunk]));
            acc0 = fmaf(g0, top, acc0);
            acc1 = fmaf(g1, top, acc1);
            acc2 = fmaf(g2, bottom, acc2);
            acc3 = fmaf(g3, bottom, acc3);
        }

            __syncthreads(); // the next stage reuses this buffer
        }

        // C fragment: c0/c1 are rows gid, columns 2*lid / 2*lid+1; c2/c3 are rows gid+8.
        const int token_a = tok_base + 2 * lid;
        const int token_b = token_a + 1;

        if (row_lo < rows) {
            if (token_a < tokens) {
                out[static_cast<std::int64_t>(token_a) * out_row_stride + row_lo] =
                    __float2bfloat16_rn(acc0);
            }
            if (token_b < tokens) {
                out[static_cast<std::int64_t>(token_b) * out_row_stride + row_lo] =
                    __float2bfloat16_rn(acc1);
            }
        }
        if (row_hi < rows) {
            if (token_a < tokens) {
                out[static_cast<std::int64_t>(token_a) * out_row_stride + row_hi] =
                    __float2bfloat16_rn(acc2);
            }
            if (token_b < tokens) {
                out[static_cast<std::int64_t>(token_b) * out_row_stride + row_hi] =
                    __float2bfloat16_rn(acc3);
            }
        }
    }
}

} // namespace ninfer::ops::detail
