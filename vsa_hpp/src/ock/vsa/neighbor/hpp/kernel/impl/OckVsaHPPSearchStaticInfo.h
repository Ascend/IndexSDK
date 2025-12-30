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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_SEARCH_USED_TIME_STATIC_INFO_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_KERNEL_SEARCH_USED_TIME_STATIC_INFO_H
#include <cstdint>
#include <utility>
#include <chrono>
#include <memory>
#include <atomic>
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPFilterType.h"
#include "ock/vsa/OckVsaErrorCode.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
inline uint64_t ElapsedMicroSeconds(const std::chrono::steady_clock::time_point &startTime)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startTime).count();
}
struct OckVsaHPPSearchStaticInfo {
    uint64_t processorInitTime{ 0 };
    uint64_t searchTime{ 0 };
    uint64_t filterTime{ 0 };
    uint64_t notifyEndTime{ 0 };
    uint64_t composeNotifytime{ 0 };
    uint64_t sampleTopNTime{ 0 };
    uint64_t maskGenerateTime{ 0 };
    uint64_t npuSearchTime{ 0 };
    std::atomic<uint64_t> transferRows{ 0 };

    OckVsaHPPFilterType filterType{ OckVsaHPPFilterType::FULL_FILTER };
};

inline std::ostream &operator << (std::ostream &os, const OckVsaHPPSearchStaticInfo &info)
{
    return os << "{'processorInitTime': " << info.processorInitTime << ",'searchTime': " << info.searchTime <<
        ",'filterTime': " << info.filterTime << ",'filterTime+notifyTime': " << info.composeNotifytime <<
        ",'notifyEndTime': " << info.notifyEndTime << ",'sampleTopNTime': " << info.sampleTopNTime <<
        ",'maskGenerateTime': " << info.maskGenerateTime << ",'npuSearchTime': " << info.npuSearchTime <<
        ",'transferRows': " << info.transferRows.load() << "(about:" << (info.transferRows.load() / 4096UL) << "MB)"
              << ",'filterType': " << info.filterType << "}";
}
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif