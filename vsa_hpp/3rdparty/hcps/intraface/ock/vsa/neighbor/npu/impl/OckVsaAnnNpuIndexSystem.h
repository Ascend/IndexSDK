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


#ifndef OCK_VSA_NPU_ANN_INDEX_SYSTEM_H
#define OCK_VSA_NPU_ANN_INDEX_SYSTEM_H
#include <deque>
#include <memory>
#include "ock/hcps/hfo/OckLightIdxMap.h"
#include "ock/hcps/hfo/OckTokenIdxMap.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
#include "ock/vsa/OckVsaErrorCode.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuBlockGroup.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/vsa/neighbor/base/OckVsaAnnIndexBase.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/utils/OckSafeUtils.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
const uint64_t MAX_GET_NUMBER = 1.0E6;
const uint32_t MAX_SEARCH_TOPK = 1e5;
const uint32_t MAX_SEARCH_BATCH_SIZE = 10240;             //  限定count最大值为10240
const uint64_t MIN_MAX_FEATURE_ROW_COUNT = 16777216ULL;   //  限定 MaxFeatureRowCount 最小值为 16777216
const uint64_t MAX_MAX_FEATURE_ROW_COUNT = 262144UL * 3UL * 550UL;  //  限定 MaxFeatureRowCount 最大值为 432537600
const uint32_t MIN_TOKEN_NUM = 1U;                        //  限定 TokenNum 最小值为 1
const uint32_t MAX_TOKEN_NUM = 300000U;                   //  限定 TokenNum 最大值为 300000
const long MIN_CPU_SET_NUM = 4L;                          //  限定有效 cpu 核数最小值为 4
const uint32_t MAX_EXTKEYATTR_BYTESIZE = 22U;             //  限定 ExtKeyAttrByteSize 最大值为 22
const uint32_t MIN_EXTKEYATTR_BLOCKSIZE = 262144U;        //  限定 ExtKeyAttrBlockSize 最小值为 262144
const uint32_t MAX_GROUP_ROW_COUNT = 419430400U;          //  限定 GroupRowCount 最大值为 419430400
const uint32_t MAX_MAX_GROUP_COUNT = 100U;                //  限定 MaxGroupCount 最大值为 100

const uint64_t MULTIPLE_MAX_FEATURE_ROW_COUNT = 256UL; //  限定 MaxFeatureRowCount 能被 256 整除
const uint32_t MULTIPLE_GROUP_BLOCK_COUNT = 16U;       //  限定 GroupBlockCount 能被 16 整除
const uint32_t MULTIPLE_SLICE_ROW_COUNT = 64U;         //  限定 SliceRowCount 能被 64 整除
const uint32_t MULTIPLE_GROUP_SLICE_COUNT = 64U;       //  限定 GroupSliceCount 能被 64 整除
}
}
}
}

#endif