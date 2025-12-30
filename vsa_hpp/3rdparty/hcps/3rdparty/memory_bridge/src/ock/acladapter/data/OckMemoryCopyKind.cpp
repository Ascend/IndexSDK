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

#include <map>
#include "ock/utils/StrUtils.h"
#include "ock/acladapter/data/OckMemoryCopyKind.h"
namespace ock {
namespace acladapter {
std::ostream &operator<<(std::ostream &os, OckMemoryCopyKind kind)
{
    const std::map<OckMemoryCopyKind, const char *> enumNameMap{
        {OckMemoryCopyKind::HOST_TO_HOST, "HOST_TO_HOST"},
        {OckMemoryCopyKind::HOST_TO_DEVICE, "HOST_TO_DEVICE"},
        {OckMemoryCopyKind::DEVICE_TO_HOST, "DEVICE_TO_HOST"},
        {OckMemoryCopyKind::DEVICE_TO_DEVICE, "DEVICE_TO_DEVICE"},
    };
    auto iter = enumNameMap.find(kind);
    if (iter == enumNameMap.end()) {
        os << "UnknownCopyKind(" << static_cast<uint32_t>(kind) << ")";
    } else {
        os << iter->second;
    }
    return os;
}

OckMemoryCopyKind CalcMemoryCopyKind(hmm::OckHmmHeteroMemoryLocation src, hmm::OckHmmHeteroMemoryLocation dst)
{
    if (src == hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY) {
        if (dst == hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR || dst == hmm::OckHmmHeteroMemoryLocation::DEVICE_HBM) {
            return OckMemoryCopyKind::HOST_TO_DEVICE;
        } else {
            return OckMemoryCopyKind::HOST_TO_HOST;
        }
    } else {
        if (dst == hmm::OckHmmHeteroMemoryLocation::DEVICE_DDR || dst == hmm::OckHmmHeteroMemoryLocation::DEVICE_HBM) {
            return OckMemoryCopyKind::DEVICE_TO_DEVICE;
        } else {
            return OckMemoryCopyKind::DEVICE_TO_HOST;
        }
    }
}
std::pair<bool, uint32_t> ParseMemoryCopyKind(const std::string &inData)
{
    auto kind = utils::ToUpper(inData);
    if (kind == "HOST_TO_HOST") {
        return std::make_pair(true, (uint32_t)OckMemoryCopyKind::HOST_TO_HOST);
    } else if (kind == "HOST_TO_DEVICE") {
        return std::make_pair(true, (uint32_t)OckMemoryCopyKind::HOST_TO_DEVICE);
    } else if (kind == "DEVICE_TO_HOST") {
        return std::make_pair(true, (uint32_t)OckMemoryCopyKind::DEVICE_TO_HOST);
    } else if (kind == "DEVICE_TO_DEVICE") {
        return std::make_pair(true, (uint32_t)OckMemoryCopyKind::DEVICE_TO_DEVICE);
    } else if (kind == "ALL") {
        return std::make_pair(true,
            (uint32_t)OckMemoryCopyKind::DEVICE_TO_DEVICE + (uint32_t)OckMemoryCopyKind::HOST_TO_HOST +
                (uint32_t)OckMemoryCopyKind::HOST_TO_DEVICE + (uint32_t)OckMemoryCopyKind::DEVICE_TO_HOST);
    } else {
        return std::make_pair(false, 0U);
    }
}
}  // namespace acladapter
}  // namespace ock