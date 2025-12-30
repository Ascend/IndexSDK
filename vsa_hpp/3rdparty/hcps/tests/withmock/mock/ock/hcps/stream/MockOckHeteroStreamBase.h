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

#ifndef HCPS_PIER_MOCKOCKHETEROSTREAMBASE_H
#define HCPS_PIER_MOCKOCKHETEROSTREAMBASE_H
#include <gmock/gmock.h>
#include "ock/hcps/stream/OckHeteroStreamBase.h"
namespace ock {
namespace hcps {
class MockOckHeteroStreamBase : public OckHeteroStreamBase {
public:
    MOCK_CONST_METHOD0(DevRtStream, acladapter::OckDevRtStream());
    MOCK_METHOD1(AddOp, void(std::shared_ptr<OckHeteroOperatorBase> op));
    MOCK_METHOD1(AddOps, void(OckHeteroOperatorGroup &ops));
    MOCK_METHOD1(AddOps, void(OckHeteroOperatorTroupe &troupes));
    MOCK_METHOD3(RunOps, hmm::OckHmmErrorCode(
            OckHeteroOperatorGroupQueue &ops, OckStreamExecPolicy policy, uint32_t timeout));
    MOCK_METHOD1(WaitExecComplete, hmm::OckHmmErrorCode(uint32_t timeout));
};
}
}
#endif // HCPS_PIER_MOCKOCKHETEROSTREAMBASE_H
