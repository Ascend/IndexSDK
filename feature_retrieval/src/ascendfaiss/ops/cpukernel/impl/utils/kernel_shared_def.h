/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * -------------------------------------------------------------------------
 */

#ifndef AICPU_KERNEL_SHARED_DEF_H
#define AICPU_KERNEL_SHARED_DEF_H

#include <cstddef>
#include <cstdint>

namespace aicpu
{
enum TopkFlatAttrIdx : int32_t
{
    TOPK_FLAT_ATTR_ASC_IDX = 0,
    TOPK_FLAT_ATTR_K_IDX,
    TOPK_FLAT_ATTR_BURST_LEN_IDX,
    TOPK_FLAT_ATTR_BLOCK_NUM_IDX,
    TOPK_FLAT_ATTR_PAGE_IDX,
    TOPK_FLAT_ATTR_PAGE_NUM_IDX,
    TOPK_FLAT_ATTR_PAGE_SIZE_IDX,
    TOPK_FLAT_ATTR_QUICK_HEAP,
    TOPK_FLAT_ATTR_BLOCK_SIZE,
    TOPK_FLAT_ATTR_IDX_COUNT,
};

enum TopkMultisearchAttrIdx : int32_t
{
    TOPK_MULTISEARCH_ATTR_ASC_IDX = 0,
    TOPK_MULTISEARCH_ATTR_K_IDX,
    TOPK_MULTISEARCH_ATTR_BURST_LEN_IDX,
    TOPK_MULTISEARCH_ATTR_PAGE_BLOCK_NUM_IDX,
    TOPK_MULTISEARCH_ATTR_INDEX_NUM_IDX,
    TOPK_MULTISEARCH_ATTR_QUICK_HEAP,
    TOPK_MULTISEARCH_ATTR_BLOCK_SIZE,
    TOPK_MULTISEARCH_ATTR_IDX_COUNT,
};

enum TopkIvfAttrIdx : int32_t
{
    TOPK_IVF_ATTR_ASC_IDX = 0,
    TOPK_IVF_ATTR_K_IDX,
    TOPK_IVF_ATTR_BURST_LEN_IDX,
    TOPK_IVF_ATTR_BLOCK_NUM_IDX,
    TOPK_IVF_ATTR_FLAG_NUM_IDX,
    TOPK_IVF_ATTR_QUICK_HEAP,
    TOPK_IVF_ATTR_IDX_COUNT,
};

// AttrIdx is an int32 index into the AICPU int64 attr vector (same as other Topk*AttrIdx).
enum TopkIvfRabitqAttrIdx : int32_t
{
    TOPK_IVF_RABITQ_ATTR_ASC_IDX = 0,
    TOPK_IVF_RABITQ_ATTR_K_IDX,
    TOPK_IVF_RABITQ_ATTR_BURST_LEN_IDX,
    TOPK_IVF_RABITQ_ATTR_BLOCK_NUM_IDX,
    TOPK_IVF_RABITQ_ATTR_QUERY_NUM_IDX,
    TOPK_IVF_RABITQ_ATTR_CORE_NUM_IDX,
    // IDSelector late-filter: mode / device payload ptr / aux0 / aux1 / negate
    TOPK_IVF_RABITQ_ATTR_SEL_MODE_IDX,
    TOPK_IVF_RABITQ_ATTR_SEL_PTR_IDX,
    TOPK_IVF_RABITQ_ATTR_SEL_AUX0_IDX,
    TOPK_IVF_RABITQ_ATTR_SEL_AUX1_IDX,
    TOPK_IVF_RABITQ_ATTR_SEL_NEGATE_IDX,
    TOPK_IVF_RABITQ_ATTR_LAYOUT_MODE_IDX,
    TOPK_IVF_RABITQ_ATTR_IDX_COUNT,
};

// FilterMode is an attr VALUE stored as int64 in the AICPU attr vector.
enum RabitqIdFilterMode : int64_t
{
    RABITQ_ID_FILTER_NONE = 0,
    RABITQ_ID_FILTER_RANGE = 1,          // aux0=imin, aux1=imax (half-open)
    RABITQ_ID_FILTER_SORTED = 2,         // ptr -> int64[aux0], binary search
    RABITQ_ID_FILTER_BITMAP = 3,         // ptr -> uint8[aux0/8], aux0 = bit count (multiple of 8)
    RABITQ_ID_FILTER_SORTED_PREFIX = 4,  // ptr -> RabitqSortedPrefixPayloadHeader + sorted ids + prefix offsets
    RABITQ_ID_FILTER_ROARING = 5,        // ptr -> frozen CRoaring bytes, aux0 = byte length
};

// CRoaring roaring_bitmap_frozen_view requires this alignment on the frozen buffer.
constexpr size_t RABITQ_ROARING_FROZEN_ALIGN = 32;

constexpr int64_t RABITQ_SORTED_PREFIX_MAGIC = 0x5242515052465831;

struct RabitqSortedPrefixPayloadHeader
{
    int64_t magic;
    int64_t sortedCount;
    int64_t sortedOffsetBytes;
    int64_t prefixBits;
    int64_t prefixShift;
    int64_t prefixBucketCount;
    int64_t prefixOffsetBytes;
};

// Order the host fills the L2 distance grid in, and therefore the order the AI Core produces blocks and
// the order TopK can consume them. Only the MODE travels to the kernel: it decides whether a query's
// blocks are contiguous in the grid, which is all the kernel needs to know to address them. The group
// granularity shapes the host-side fill alone and stays host-local.
//
// LEGACY is query-major contiguous: every block of query 0, then query 1, and so on. Later queries start
// late and finish late -- a staircase. GROUPED round-robins within a window of `group` consecutive queries
// and runs the windows back to back: a window's queries finish together and the next window starts on the
// threads that just freed up. group=1 degenerates to the LEGACY order.
enum TopkIvfRabitqLayoutMode : int32_t
{
    RABITQ_LAYOUT_LEGACY = 0,
    RABITQ_LAYOUT_GROUPED,
    RABITQ_LAYOUT_MODE_COUNT,
};

// Fallback AI CPU count for the GROUPED window, used only when the device cannot be asked: a window is
// only ever drained `AICPU_NUM` queries at a time, so a wider one just delays every query's finish.
//
// The host normally resolves this at runtime from ACL (ACL_DEV_ATTR_AICPU_CORE_NUM; see
// ResolveTopkGroupSize in IndexIVFRaBitQ.cpp) so a deployment never has to declare its own ctrl:AI CPU
// split. This constant is what that query falls back to when no device is set yet, the ACL version
// predates the attribute, or the code runs against a simulator. The value is the same one the other AI CPU
// kernels cap their thread count at (THREAD_CNT), so the fallback matches long-standing behaviour.
//
// Note the two counts are not identical: ACL reports the device's AI CPU count, whereas the kernel sees
// the per-context count via CpuKernelUtils::GetCPUNum (which it already mins with nq). On a shared card
// the host therefore reads high, which costs parallelism but never correctness -- blockSlots is an
// explicit bijection the kernel reads back, so the window width never affects which block a query scans.
constexpr int64_t TOPK_IVF_RABITQ_AICPU_NUM = 6;

// Bits OR-ed into the QUERY_NUM attr slot by the host to toggle optional kernel paths. They sit in high
// bits so the slot's low bits still read as the query count; bit 32 and up are far above any realistic
// batch. The env vars behind them are read on the HOST — an AICPU kernel runs in a separate device-side
// execution environment and does not inherit the host process's environ, so getenv there returns null.
//
// Why spare bits of an existing slot and not a new attr: the attr tensor's length is baked into the op desc
// (resetL2TopkOp passes TOPK_IVF_RABITQ_ATTR_IDX_COUNT as the input shape) and the kernel hard-checks
// attrCount against the same constant. Adding a slot therefore changes the op signature, and since the host
// and the kernel .om are deployed as separate, independently-versioned artifacts, a version-skewed pair
// would fail the attr-count check outright. Riding spare bits degrades safely in both skew directions: an
// old kernel never reads this slot at all (nq comes from the tensor shapes, so the slot has always been
// written by the host and ignored by the kernel), and a new kernel reads 0 from an old host — in both cases
// the historical k-heap path runs.
constexpr int64_t TOPK_IVF_RABITQ_RESERVOIR_FLAG = 1LL << 32;    // reservoir/nth_element select path
constexpr int64_t TOPK_IVF_RABITQ_PACKED_SORT_FLAG = 1LL << 33;  // Reorder's packed introsort

// Bits 36-39 carry the reservoir's REFILL_CHUNK (B) as a LOG2, so the whole knob costs 4 bits instead of
// the 16 a literal B would need. 0 means "the host said nothing", and the kernel then uses its own default.
// A log2 rather than the value because B is only ever interesting as a power of two (it is a buffer size
// traded against a refill count, and the interesting experiment is doubling it).
constexpr int64_t TOPK_IVF_RABITQ_REFILL_LOG2_SHIFT = 36;
constexpr int64_t TOPK_IVF_RABITQ_REFILL_LOG2_BITS = 4;
constexpr int64_t TOPK_IVF_RABITQ_REFILL_LOG2_MASK = (1LL << TOPK_IVF_RABITQ_REFILL_LOG2_BITS) - 1;
// Legal log2 range when the field is non-zero. The floor is not arbitrary: below ~256 the per-refill O(k+B)
// selection stops being amortised over enough accepts and the refill count explodes, so a smaller B is
// never a question worth asking. The ceiling bounds the buffer at k=5000 to (5000+32768)*8B = ~295KB per
// worker thread, which is already far past any plausible win and is where an experiment should be stopped
// by the validator rather than by a device OOM.
constexpr int64_t TOPK_IVF_RABITQ_REFILL_LOG2_MIN = 8;   // B = 256
constexpr int64_t TOPK_IVF_RABITQ_REFILL_LOG2_MAX = 15;  // B = 32768
// The B used when the field is 0 (host sent nothing / older host). Lives here rather than in the kernel
// .cpp because both the kernel's fallback and the kernel member's initialiser need the same number. The
// reasoning for the VALUE 2048 is in the REFILL_CHUNK_DEFAULT comment in the kernel .cpp.
constexpr int64_t TOPK_IVF_RABITQ_REFILL_DEFAULT = 2048;

enum TopkIvfpqL3AttrIdx : int32_t
{
    TOPK_IVFPQ_L3_ATTR_ASC_IDX = 0,
    TOPK_IVFPQ_L3_ATTR_K_IDX,
    TOPK_IVFPQ_L3_ATTR_BLOCK_NUM_IDX,
    TOPK_IVFPQ_L3_ATTR_BATCH_NUM_IDX,
    TOPK_IVFPQ_L3_ATTR_IDX_COUNT,
};

enum TopkIvfFuzzyAttrIdx : int32_t
{
    TOPK_IVF_FUZZY_ATTR_ASC_IDX = 0,
    TOPK_IVF_FUZZY_ATTR_K_IDX,
    TOPK_IVF_FUZZY_ATTR_BURST_LEN_IDX,
    TOPK_IVF_FUZZY_ATTR_L3_SEG_NUM_IDX,
    TOPK_IVF_FUZZY_ATTR_L3_SEG_SIZE_IDX,
    TOPK_IVF_FUZZY_ATTR_K_HEAP_RATIO_IDX,
    TOPK_IVF_FUZZY_ATTR_K_BUF_RATIO_IDX,
    TOPK_IVF_FUZZY_ATTR_Q_BATCH_SIZE_IDX,
    TOPK_IVF_FUZZY_ATTR_SORT_MODE,
    TOPK_IVF_FUZZY_ATTR_IDX_COUNT,
};

enum TopkIvfsqtL1AttrIdx : int32_t
{
    TOPK_IVFSQT_L1_ATTR_ASC_IDX = 0,
    TOPK_IVFSQT_L1_ATTR_K_IDX,
    TOPK_IVFSQT_L1_ATTR_BURST_LEN_IDX,
    TOPK_IVFSQT_L1_ATTR_OP_SIZE_IDX,
    TOPK_IVFSQT_L1_ATTR_Q_BATCH_SIZE_IDX,
    TOPK_IVFSQT_L1_ATTR_QUICK_HEAP,
    TOPK_IVFSQT_L1_ATTR_IDX_COUNT,
};

enum TopkIvfsqtL2AttrIdx : int32_t
{
    TOPK_IVFSQT_L2_ATTR_K_IDX,
    TOPK_IVFSQT_L2_ATTR_SUBCENTER_NUM_IDX,
    TOPK_IVFSQT_L2_ATTR_L3_SEG_NUM_IDX,
    TOPK_IVFSQT_L2_ATTR_L3_SEG_SIZE_IDX,
    TOPK_IVFSQT_L2_ATTR_PAGE_SHAPED_DATA_OFFSET_STEP_IDX,
    TOPK_IVFSQT_L2_ATTR_L1_NPROBE_IDX,
    TOPK_IVFSQT_L2_ATTR_Q_BATCH_SIZE_IDX,
    TOPK_IVFSQT_L2_ATTR_IDX_COUNT,
};

enum TransdataShapedAttrIdx : int32_t
{
    TRANSDATA_SHAPED_ATTR_NTOTAL_IDX = 0,
    TRANSDATA_SHAPED_ATTR_IDX_COUNT,
};

enum TransdataRawAttrIdx : int32_t
{
    TRANSDATA_RAW_ATTR_OFFSET_IDX = 0,
    TRANSDATA_RAW_ATTR_IDX_COUNT,
};

enum CodesQuantifyAttrIdx : int32_t
{
    CODES_QUANTIFY_ATTR_QMAX_IDX = 0,
    CODES_QUANTIFY_ATTR_QMIN_IDX = 1,
    CODES_QUANTIFY_ATTR_IDX_COUNT,
};

enum TakeCareOfVoidClusterAttrIdx : int32_t
{
    TAKE_CARE_OF_VOID_CLUSTER_K_IDX = 0,
    TAKE_CARE_OF_VOID_CLUSTER_N_IDX = 1,
    TAKE_CARE_OF_VOID_CLUSTER_IDX_COUNT,
};

enum RomovedataShapedAttrIdx : int32_t
{
    REMOVEDATA_SHAPED_ATTR_DATA_TYPE = 0,
    REMOVEDATA_SHAPED_ATTR_ZREGION_HEIGHT = 1,
    REMOVEDATA_SHAPED_ATTR_CUBE_ALIGN = 2,
    REMOVEDATA_SHAPED_ATTR_DIM_ALIGN_NUM = 3,
    REMOVEDATA_SHAPED_ATTR_IDX_COUNT,
};

enum RomovedataAttrIdx : int32_t
{
    REMOVEDATA_ATTR_DATA_TYPE = 0,
    REMOVEDATA_ATTR_COPY_NUM = 1,
    REMOVEDATA_ATTR_IDX_COUNT,
};

enum TransdataCustomAttrIdx : int32_t
{
    TRANSDATA_CUSTOM_ATTR_NTOTAL_IDX = 0,
    TRANSDATA_CUSTOM_ATTR_IDX_COUNT,
};

enum RomovedataCustomAttrIdx : int32_t
{
    REMOVEDATA_CUSTOM_ATTR_DATA_TYPE = 0,
    REMOVEDATA_CUSTOM_ATTR_LEN = 1,
    REMOVEDATA_CUSTOM_ATTR_BLOCKSIZE = 2,
    REMOVEDATA_CUSTOM_ATTR_IDX_COUNT,
};

}  // namespace aicpu

#endif
