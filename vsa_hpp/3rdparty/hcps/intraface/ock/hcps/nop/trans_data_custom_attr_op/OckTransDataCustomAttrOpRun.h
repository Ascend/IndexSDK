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

#ifndef HCPS_OCK_TRANS_DATA_CUSTOM_ATTR_OP_RUN_H
#define HCPS_OCK_TRANS_DATA_CUSTOM_ATTR_OP_RUN_H
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/nop/trans_data_custom_attr_op/OckTransDataCustomAttrMeta.h"
namespace ock {
namespace hcps {
namespace nop {
class OckTransDataCustomAttrOpRun {
public:
    static void AddTransCustomAttrOp(std::shared_ptr<OckTransDataCustomAttrOpHmoBlock> hmoBlock,
        handler::OckHeteroHandler &handler, std::shared_ptr<OckHeteroStreamBase> streamBase);
};
}
}
}

#endif // HCPS_OCK_TRANS_DATA_CUSTOM_ATTR_OP_RUN_H
