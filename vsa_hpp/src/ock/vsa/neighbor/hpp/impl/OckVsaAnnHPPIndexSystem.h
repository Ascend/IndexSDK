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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_SYSTEM_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_SYSTEM_H
#include <deque>
#include "ock/hcps/hfo/OckLightIdxMap.h"
#include "ock/hcps/hfo/OckTokenIdxMap.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/vsa/neighbor/base/OckVsaAnnIndexBase.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuBlockGroup.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuIndex.h"
#include "ock/vsa/neighbor/hpp/kernel/OckVsaHPPKernelExt.h"
#include "ock/log/OckVsaHppLogger.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
const uint64_t IDX_MAP_BUCKET_NUMBER = 4000000ULL;
const uint32_t MAX_FEATURE_GROUP_IN_DEVICE = 2UL;
const uint32_t MIN_FEATURE_GROUP_IN_DEVICE = 1UL;
const uint32_t MAX_GET_NUMBER = 1e6;
const uint32_t MAX_SEARCH_TOPK = 1e5;
const uint32_t MAX_SEARCH_BATCH_SIZE = 10240UL; //  限定count最大值为10240
const uint64_t FRAGMENT_SIZE_THRESHOLD = 64ULL * 1024ULL * 1024ULL;
const uint64_t MAX_COVERAGE_TIMES_OF_SAMPLE_NEIGHBOR = 10ULL;
const uint64_t ONE_GROUP_SPACE = 4ULL * 1024ULL * 1024ULL * 1024ULL;
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif