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


#ifndef OCK_HCPS_HCOP_ASCEND_OPERATOR_H
#define OCK_HCPS_HCOP_ASCEND_OPERATOR_H
#include <memory>
namespace ock {
namespace hcps {
namespace nop {
const int64_t CUBE_ALIGN = 16;
const int64_t CUBE_ALIGN_INT8 = 32;
const int64_t FP16_ALIGN = 16;
const int64_t CORE_NUM = 8;
const int64_t SIZE_ALIGN = 8;
const int64_t FLAG_NUM = 16;
const int64_t FLAG_SIZE = 16;
const int64_t BURST_LEN = 64;
const int64_t TRANSFER_SIZE = 256;
const int64_t L2NORM_COMPUTE_BATCH = 16384;
const int64_t FLAT_DEFAULT_DIST_COMPUTE_BATCH = 16384 * 16;
const int64_t BINARY_BYTE_SIZE = 8;
const int64_t TOPK_FLAT_ATTR_IDX_COUNT = 9;
const int64_t UNIT_BLOCK_SIZE = 16384;
const int64_t DEFAULT_PAGE_BLOCK_NUM = 16;
const int64_t DEFAULT_GROUP_BLOCK_NUM = 64;
const int64_t TRANSDATA_SHAPED_ATTR_IDX_COUNT = 1;
const int64_t TRANSDATA_CUSTOM_ATTR_IDX_COUNT = 1;
const int64_t REMOVEDATA_CUSTOM_ATTR_IDX_COUNT = 3;
const int64_t REMOVEDATA_ATTR_IDX_COUNT = 2;
const int64_t REMOVEDATA_SHAPED_IDX_COUNT = 4;
const int64_t DEFAULT_CODE_BLOCK_SIZE = 262144;
const int64_t OPS_DATA_TYPE_ALIGN = 8;
const int64_t OPS_DATA_TYPE_TIMES = 2;

enum Type {
    UNDEFINED = -1,
    FLOAT,
    FLOAT16,
    INT8,
    INT32,
    UINT8,
    UINT16,
    UINT32,
    INT64,
    UINT64,
    DOUBLE,
    BOOL,
    STRING
};

} // namespace nop
} // namespace hcps
} // namespace ock
#endif