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


#ifndef OCK_HCPS_OCK_DEVICE_STREAM_MGR_H
#define OCK_HCPS_OCK_DEVICE_STREAM_MGR_H
#include <memory>
#include <vector>
#include "ock/hcps/error/OckHcpsErrorCode.h"
#include "ock/acladapter/executor/OckAsyncTaskExecuteService.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
namespace ock {
namespace hcps {
struct OckDevStreamInfo {
    acladapter::OckDevRtStream stream;
    OckDevStreamType streamType;
};

OckDevStreamInfo MakeOckDevStreamInfo(acladapter::OckDevRtStream stream, OckDevStreamType streamType);
class OckDevStreamMgr {
public:
    virtual ~OckDevStreamMgr() noexcept = default;
    /*
    @brief AI_CPU和AI_CORE类型的stream，同一时间只能1个对象拥有
    */
    virtual OckDevStreamInfo CreateStream(
        acladapter::OckAsyncTaskExecuteService &service, OckDevStreamType streamType, OckHcpsErrorCode &errorCode) = 0;
    virtual void DestroyStreams(acladapter::OckAsyncTaskExecuteService &service) = 0;

    static OckDevStreamMgr &Instance(void);
};
std::ostream &operator<<(std::ostream &os, const OckDevStreamInfo &data);
}  // namespace hcps
}  // namespace ock
#endif