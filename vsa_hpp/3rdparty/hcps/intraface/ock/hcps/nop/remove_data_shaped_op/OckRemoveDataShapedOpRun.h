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

#ifndef HCPS_PIER_OCKREMOVEDATASHAPEDOPRUN_H
#define HCPS_PIER_OCKREMOVEDATASHAPEDOPRUN_H
#include "ock/hcps/handler/OckHeteroHandler.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
#include "ock/hcps/nop/remove_data_shaped_op/OckRemoveDataShapedMeta.h"
namespace ock {
namespace hcps {
namespace nop {
class OckRemoveDataShapedOpRun {
public:
    static std::shared_ptr<OckHeteroOperatorBase> GenRemoveShapedOp(
        std::shared_ptr<OckRemoveDataShapedOpHmoBlock> hmoBlock, handler::OckHeteroHandler &handler);
};
}
}
}
#endif // HCPS_PIER_OCKREMOVEDATASHAPEDOPRUN_H
