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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_SYSTEM_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_EXT_SYSTEM_H
#include <cstdint>
#include <utility>
#include <deque>
#include <memory>
#include "ock/hcps/algo/OckBitSet.h"
#include "ock/hcps/algo/OckElasticBitSet.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/hfo/OckLightIdxMap.h"
#include "ock/hcps/hfo/OckTokenIdxMap.h"
#include "ock/hcps/hfo/feature/OckHashFeatureGen.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/vsa/attr/OckMaxMinTrait.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCollectResultProcessor.h"
#include "ock/vsa/neighbor/base/OckVsaAnnQueryCondition.h"
#include "ock/vsa/neighbor/base/OckVsaAnnQueryResult.h"
#include "ock/vsa/neighbor/base/OckVsaAnnIndexBase.h"
#include "ock/vsa/neighbor/base/OckVsaHPPInnerIdConvertor.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuIndex.h"
#include "ock/vsa/neighbor/hpp/kernel/OckVsaHPPDataType.h"
#include "ock/log/OckVsaHppLogger.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaVectorHash.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaSampleFeatureMgr.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaNeighborRelation.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
const bool FORCE_PRINT = false;
const uint32_t EXT_FEATURE_SCALE_OUT_BITS = 16UL; // 每条数据采用16bit来计算特征
const uint32_t HASH_SAMPLE_INTERVAL = 1UL;
const uint32_t MIN_TOPN_SELECT_SLICE_COUNT = 10UL;
const uint32_t MAX_TOPN_SELECT_SLICE_COUNT = 64UL;
const uint32_t TOPK_SELECT_SLICE_TIMES = 3UL;
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif