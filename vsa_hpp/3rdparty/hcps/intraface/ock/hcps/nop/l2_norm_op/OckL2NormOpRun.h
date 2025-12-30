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


#ifndef HCPS_OCKL2NORMOPRUN_H
#define HCPS_OCKL2NORMOPRUN_H
#include "ock/hmm/mgr/OckHmmHeteroMemoryMgr.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormMeta.h"
namespace ock {
namespace hcps {
namespace nop {
class OckL2NormOpRun {
public:
    static OckHcpsErrorCode ComputeNormSync(std::shared_ptr<OckL2NormOpHmoBlock> hmoBlock,
        handler::OckHeteroHandler &handler,
        std::shared_ptr<OckHeteroStreamBase> streamBase);
    static std::shared_ptr<OckL2NormOpHmoBlock> BuildNormHmoBlock(std::shared_ptr<hmm::OckHmmHMObject> data,
        handler::OckHeteroHandler &handler, uint64_t dims, uint64_t addNum, OckHcpsErrorCode &errorCode);
};
}
}
}

#endif // HCPS_OCKL2NORMOPRUN_H
