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


#ifndef LIBASCENDCL_ASCENDSIMUDEVICE_H
#define LIBASCENDCL_ASCENDSIMUDEVICE_H

#include <cstdint>
#include <vector>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <map>
#include "../acl.h"
#include "AscendSimuExecFlow.h"

constexpr uint32_t MAX_DEVICE = 64;

class AscendSimuDevice {
public:
    explicit AscendSimuDevice(int32_t deviceId, uint8_t aicore = 0) : m_deviceId(deviceId), m_aicore(aicore) {};
    virtual ~AscendSimuDevice();
    void Init();
    void DeInit();

    void CreateContext(aclrtContext *ctxt);
    void DestoryContext(aclrtContext ctxt);
    aclrtContext GetActiveContext();
    void SetActiveContext(aclrtContext);

    void CreateStream(aclrtContext ctxt, aclrtStream *stream);
    void DestoryStream(aclrtStream stream);
    void WaitStreamExecFlowIdle(aclrtStream stream);
    bool StreamExecFlowPushOpHandle(aclrtStream stream, aclopHandle *opHandle);
    bool StreamExecFlowPushOpHandle(aclrtStream stream, aclrtCallback fn, void *userData);
    bool StreamIsRunning(aclrtStream stream);
    size_t GetRefCnt() const;

private:
    void CreateStreamExecFlow(aclrtStream stream);
    void DestoryStreamExecFlow(aclrtStream stream);
    void ClearAll();

    int32_t m_deviceId;
    size_t m_refCnt {0}; // device引用次数
    aclrtContext m_defaultContext {nullptr};
    uint8_t m_aicore {0};

    static thread_local std::vector<aclrtContext> m_activeContexts;
    std::vector<aclrtContext> m_ctxts;
    std::map<aclrtStream, AscendSimuExecFlow *> m_flowMap;
};

#endif // LIBASCENDCL_ASCENDSIMUDEVICE_H
