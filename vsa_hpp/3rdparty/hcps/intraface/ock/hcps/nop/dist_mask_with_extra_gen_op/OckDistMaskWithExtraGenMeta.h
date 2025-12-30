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

#ifndef HCPS_PIER_OCKDISTMASKWITHEXTRAGENMETA_H
#define HCPS_PIER_OCKDISTMASKWITHEXTRAGENMETA_H
#include <vector>
#include "ock/utils/OckSafeUtils.h"
#include "ock/hcps/nop/OckOpConst.h"
#include "ock/hmm/mgr/OckHmmHMObject.h"
namespace ock {
namespace hcps {
namespace nop {
struct OckDistMaskWithExtraGenOpMeta {
    uint32_t tokenNum{ 2500 };
    uint32_t featureAttrBlockSize{ DEFAULT_CODE_BLOCK_SIZE };
    uint32_t blockCount{ DEFAULT_GROUP_BLOCK_NUM };
};

struct OckDistMaskWithExtraGenOpHmoGroups {
    // 全局底库特征time,int32_t
    std::vector<std::shared_ptr<std::vector<std::shared_ptr<hmm::OckHmmSubHMObject>>>> attrTimes{};
    // 全局底库特征qs，int32_t
    std::vector<std::shared_ptr<std::vector<std::shared_ptr<hmm::OckHmmSubHMObject>>>> attrTokenQuotients{};
    // 全局底库特征rs，uint8_t
    std::vector<std::shared_ptr<std::vector<std::shared_ptr<hmm::OckHmmSubHMObject>>>> attrTokenRemainders{};
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> queryTimes{}; // 长度为batch，数据类型为int32_t
    std::vector<std::shared_ptr<hmm::OckHmmHMObject>> queryTokenIds{}; // 长度为batch，数据类型为uint8_t
    std::shared_ptr<hmm::OckHmmSubHMObject> mask{ nullptr }; // 长度为batch * maskLen，数据类型为uint8_t
    std::shared_ptr<hmm::OckHmmSubHMObject> extraMask{ nullptr }; // 长度为batch * maskLen，数据类型为uint8_t
    uint32_t tokenNum{ 2500 };
    uint32_t featureAttrBlockSize{ DEFAULT_CODE_BLOCK_SIZE };
    uint32_t blockCount{ DEFAULT_GROUP_BLOCK_NUM };
    uint32_t maskLen{ DEFAULT_CODE_BLOCK_SIZE * DEFAULT_GROUP_BLOCK_NUM / OPS_DATA_TYPE_ALIGN }; // 单batch的全底库mask长度
};

struct OckDistMaskWithExtraGenOpHmoGroup {
    std::shared_ptr<hmm::OckHmmSubHMObject> queryTimes{ nullptr };
    std::shared_ptr<hmm::OckHmmSubHMObject> queryTokenIds{ nullptr };
    std::shared_ptr<hmm::OckHmmSubHMObject> attrTimes{ nullptr };
    std::shared_ptr<hmm::OckHmmSubHMObject> attrTokenQuotients{ nullptr };
    std::shared_ptr<hmm::OckHmmSubHMObject> attrTokenRemainders{ nullptr };
    std::shared_ptr<hmm::OckHmmSubHMObject> mask{ nullptr };
    std::shared_ptr<hmm::OckHmmSubHMObject> extraMask{ nullptr };
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // HCPS_PIER_OCKDISTMASKWITHEXTRAGENMETA_H
