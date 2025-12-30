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

#ifndef HCPS_OCK_TRANS_DATA_CUSTOM_ATTR_META_H
#define HCPS_OCK_TRANS_DATA_CUSTOM_ATTR_META_H
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hcps/nop/OckOpConst.h"
namespace ock {
namespace hcps {
namespace nop {
struct OckTransDataCustomAttrOpMeta {
    int64_t customAttrLen{ 20 };
    int64_t customAttrBlockSize{ DEFAULT_CODE_BLOCK_SIZE };
    int64_t copyCount{ DEFAULT_CODE_BLOCK_SIZE };
};

struct OckTransDataCustomAttrBufferMeta {};

struct OckTransDataCustomAttrOpHmoBlock {
    std::shared_ptr<hmm::OckHmmHMObject> srcHmo{ nullptr }; // 待分形原数据，数据类型为uint8_t
    std::shared_ptr<hmm::OckHmmHMObject> dstHmo{ nullptr }; // 分形后数据, 数据类型为uint8_t
    uint32_t customAttrLen{ 20 };
    uint32_t customAttrBlockSize{ DEFAULT_CODE_BLOCK_SIZE };
    uint32_t copyCount{ DEFAULT_CODE_BLOCK_SIZE };
    uint32_t offsetInBlock{ 0 };
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // HCPS_OCK_TRANS_DATA_CUSTOM_ATTR_META_H
