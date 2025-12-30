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


#ifndef OCK_HCPS_OCK_HETERO_STREAM_BASE_H
#define OCK_HCPS_OCK_HETERO_STREAM_BASE_H
#include <memory>
#include <vector>
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/hcps/stream/OckHeteroOperatorBase.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
namespace ock {
namespace hcps {
enum class OckStreamExecPolicy : uint32_t { STOP_IF_ERROR, TRY_BEST };
enum class OckDevStreamType : uint32_t { AI_DEFAULT, AI_CPU, AI_CORE, AI_NULL };
class OckHeteroStreamBase {
public:
    virtual ~OckHeteroStreamBase() noexcept = default;
    virtual acladapter::OckDevRtStream DevRtStream(void) const = 0;
    virtual void AddOp(std::shared_ptr<OckHeteroOperatorBase> op) = 0;
    virtual void AddOps(OckHeteroOperatorGroup &ops) = 0;
    virtual void AddOps(OckHeteroOperatorTroupe &troupes) = 0;
    virtual OckHcpsErrorCode RunOps(
        OckHeteroOperatorGroupQueue &ops, OckStreamExecPolicy policy, uint32_t timeout = 0) = 0;
    virtual OckHcpsErrorCode WaitExecComplete(uint32_t timeout = 0) = 0;

    /*
    @brief 1)对于HOST算子，使用AI_NULL，AI_NULL时，WaitExecComplete只会等待HOST侧的算子执行
    2) 同一个时间，只能有1个对象拥有 AI_CPU和AI_CORE stream。
    3) AI_NULL之外的其他类型， WaitExecComplete执行时，会同步等待当前OckDevStreamType中的所有任务。
    */
    static std::pair<hmm::OckHmmErrorCode, std::shared_ptr<OckHeteroStreamBase>> Create(
        std::shared_ptr<acladapter::OckAsyncTaskExecuteService> service,
        OckDevStreamType streamType = OckDevStreamType::AI_NULL);
};
std::ostream &operator<<(std::ostream &os, const OckHeteroStreamBase &data);
std::ostream &operator<<(std::ostream &os, OckDevStreamType data);
}  // namespace hcps
}  // namespace ock
#endif