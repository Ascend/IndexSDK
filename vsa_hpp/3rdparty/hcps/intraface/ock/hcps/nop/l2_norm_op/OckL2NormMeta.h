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


#ifndef HCPS_OCKL2NORMOPMETA_H
#define HCPS_OCKL2NORMOPMETA_H
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hcps/nop/OckOpConst.h"
namespace ock {
namespace hcps {
namespace nop {
struct OckL2NormOpMeta {
    uint32_t dims{ 256 };
};

struct OckL2NormBufferMeta {
    const uint32_t ntotal{ L2NORM_COMPUTE_BATCH };
};

struct OckL2NormOpHmoBlock {
    // 存储底库数据的hmo(device侧)，s按L2NORM_COMPUTE_BATCH * dims对齐，数据类型为int8_t
    std::shared_ptr<hmm::OckHmmHMObject> dataBase{ nullptr };
    // 存储norm结果的hmo(device侧)，按L2NORM_COMPUTE_BATCH对齐，数据类型为float16_t
    std::shared_ptr<hmm::OckHmmHMObject> normResult{ nullptr };
    uint32_t dims{ 256 };
    uint32_t addNum{ DEFAULT_CODE_BLOCK_SIZE };
};
} // namespace nop
} // namespace hcps
} // namespace ock
#endif // HCPS_OCKL2NORMOPMETA_H
