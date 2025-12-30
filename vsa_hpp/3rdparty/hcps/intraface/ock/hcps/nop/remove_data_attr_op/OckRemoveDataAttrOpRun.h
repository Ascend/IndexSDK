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

#ifndef HCPS_OCK_REMOVE_DATA_ATTR_OP_RUN_H
#define HCPS_OCK_REMOVE_DATA_ATTR_OP_RUN_H
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/nop/remove_data_attr_op/OckRemoveDataAttrMeta.h"
namespace ock {
namespace hcps {
namespace nop {
class OckRemoveDataAttrOpRun {
public:
    static std::shared_ptr<OckHeteroOperatorBase> CreateOp(std::shared_ptr<OckRemoveDataAttrOpHmoBlock> hmoBlock,
        handler::OckHeteroHandler &handler);
};
}
}
}

#endif // HCPS_OCK_REMOVE_DATA_ATTR_OP_RUN_H
