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


#ifndef IVFSP_KERNELSHAREDDEF_H
#define IVFSP_KERNELSHAREDDEF_H

#include <cstdint>

namespace ascendSearchacc {
/**
 * Attribute index of Op[TransDataShaped], last one is Attribute count.
 */
enum class TransDataShapedAttrIdx {
    TRANSDATA_SHAPED_ATTR_NTOTAL_IDX = 0,
    TRANSDATA_SHAPED_ATTR_COUNT,
};

/**
 * Attribute index of Op[TopK], last one is Attribute count.
 */
enum class TopKAttrIdx {
    TOPK_ATTR_ASC_IDX = 0,
    TOPK_ATTR_K_IDX,
    TOPK_ATTR_QUICK_HEAP_IDX,
    TOPK_ATTR_COUNT,
};

enum class TopkIvfSpL1AttrIdx {
    TOPK_L1_ATTR_ASC_IDX = 0,
    TOPK_L1_ATTR_K_IDX,
    TOPK_L1_ATTR_QUICK_HEAP,
    TOPK_L1_ATTR_IDX_COUNT,
};

enum class TopkIvfSpL2AttrIdx {
    TOPK_L2_ATTR_ASC_IDX = 0,
    TOPK_L2_ATTR_K_IDX,
    TOPK_L2_ATTR_NLIST2_IDX,
    TOPK_L2_ATTR_SEG_SIZE_IDX,
    TOPK_L2_ATTR_BUCKET_NUM_IDX,
    TOPK_L2_ATTR_DIM_STORE_IDX,
    TOPK_L2_ATTR_SEG_NUM_IDX,
    TOPK_L2_ATTR_IDX_COUNT,
};

enum class IvfMultiSpTopkL2AttrIdx {
    TOPK_L2_ATTR_ASC_IDX = 0,
    TOPK_L2_ATTR_K_IDX,
    TOPK_L2_ATTR_NLIST2_IDX,
    TOPK_L2_ATTR_IDX_COUNT,
};

enum class TopkIvfSpL3AttrIdx {
    TOPK_L3_ATTR_ASC_IDX = 0,
    TOPK_L3_ATTR_K_IDX,
    TOPK_L3_ATTR_NLIST2_IDX,
    TOPK_L3_ATTR_SEG_SIZE_IDX,
    TOPK_L3_ATTR_BUCKET_NUM_IDX,
    TOPK_L3_ATTR_DIM_STORE_IDX,
    TOPK_L3_ATTR_SEG_NUM_IDX,
    TOPK_L3_ATTR_IDX_COUNT,
};
}  // namespace ascendSearchacc

#endif  // IVFSP_KERNELSHAREDDEF_H
