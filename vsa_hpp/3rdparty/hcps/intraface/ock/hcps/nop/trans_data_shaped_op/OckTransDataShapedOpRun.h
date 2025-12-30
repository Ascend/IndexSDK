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

#ifndef HCPS_OCKTRANSDATASHAPEDOPRUN_H
#define HCPS_OCKTRANSDATASHAPEDOPRUN_H
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/nop/trans_data_shaped_op/OckTransDataShapedMeta.h"
namespace ock {
namespace hcps {
namespace nop {
class OckTransDataShapedOpRun {
public:
    static void AddTransShapedOp(std::shared_ptr<OckTransDataShapedOpHmoBlock> hmoBlock,
        handler::OckHeteroHandler &handler, std::shared_ptr<OckHeteroStreamBase> streamBase);
};
}
}
}

#endif // HCPS_OCKTRANSDATASHAPEDOPRUN_H
