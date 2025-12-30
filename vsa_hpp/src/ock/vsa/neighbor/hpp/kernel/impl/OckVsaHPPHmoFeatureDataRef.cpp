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


#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPHmoFeatureDataRef.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace impl {
OckVsaHPPHmoFeatureDataRef::OckVsaHPPHmoFeatureDataRef(hmm::OckHmmHMObject &srcHmo,
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &unusedHmo,
    std::deque<std::shared_ptr<hmm::OckHmmHMObject>> &usedHmo)
    : srcData(srcHmo), unusedContainer(unusedHmo), usedContainer(usedHmo)
{}

void OckVsaHPPHmoFeatureDataRef::PopUnusedToUsed(void)
{
    auto dstData = unusedContainer.front();
    unusedContainer.pop_front();
    usedContainer.push_back(dstData);
}
std::shared_ptr<hmm::OckHmmHMObject> OckVsaHPPHmoFeatureDataRef::PickUnused(void)
{
    return unusedContainer.front();
}
} // namespace impl
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock