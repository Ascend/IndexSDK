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
};

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
