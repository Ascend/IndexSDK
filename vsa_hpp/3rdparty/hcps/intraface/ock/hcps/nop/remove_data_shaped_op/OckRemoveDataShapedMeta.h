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

#ifndef HCPS_PIER_OCKREMOVEDATASHAPEDMETA_H
#define HCPS_PIER_OCKREMOVEDATASHAPEDMETA_H
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hcps/nop/OckOpConst.h"
namespace ock {
namespace hcps {
namespace nop {
struct OckRemoveDataShapedOpMeta {
    uint32_t removeCount{ 1U };
};

struct OckRemoveDataShapedOpHmoBlock {
    std::shared_ptr<hmm::OckHmmHMObject> srcPosHmo{ nullptr }; // 最后removeCount条数据的首地址，数据类型为uint64_t
    std::shared_ptr<hmm::OckHmmHMObject> dstPosHmo{ nullptr }; // 待删除的数据的首地址, 数据类型为uint64_t
    uint32_t dims{ 256U };
    uint32_t removeCount{ 1U };
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // HCPS_PIER_OCKREMOVEDATASHAPEDMETA_H
