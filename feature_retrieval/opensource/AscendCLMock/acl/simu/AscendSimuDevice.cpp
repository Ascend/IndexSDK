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


#include <algorithm>
#include <iostream>
#include "AscendSimuDevice.h"

thread_local std::vector<aclrtContext> AscendSimuDevice::m_activeContexts {MAX_DEVICE, nullptr};

void AscendSimuDevice::ClearAll()
{
    DestoryContext(m_defaultContext);
    m_defaultContext = nullptr;
    m_activeContexts[m_deviceId] = nullptr;
    m_ctxts.clear();
}

AscendSimuDevice::~AscendSimuDevice()
{
    ClearAll();
}

void AscendSimuDevice::Init()
{
    if (m_refCnt != 0) {
        ++m_refCnt;
        return;
    }
    // 用户调用aclrtSetDevice 默认创建context和stream
    CreateContext(&m_defaultContext);
    ++m_refCnt;
}

void AscendSimuDevice::DeInit()
{
    if (m_refCnt == 0) {
        return;
    }

    if (--m_refCnt == 0) {
        ClearAll();
    }
}

void AscendSimuDevice::CreateStreamExecFlow(aclrtStream stream)
{
    m_flowMap[stream] = new AscendSimuExecFlow;
    m_flowMap[stream]->start();
}

void AscendSimuDevice::DestoryStreamExecFlow(aclrtStream stream)
{
    auto it = m_flowMap.find(stream);
    if (it == m_flowMap.end()) {
        return;
    }

    auto &execFlow = m_flowMap[stream];
    if (execFlow->isRuning()) {
        execFlow->stop();
    }
    delete execFlow;
    m_flowMap.erase(it);
}

size_t AscendSimuDevice::GetRefCnt() const
{
    return m_refCnt;
}

bool AscendSimuDevice::StreamIsRunning(aclrtStream stream)
{
    auto it = m_flowMap.find(stream);
    if (it == m_flowMap.end()) {
        return false;
    }

    auto &execFlow = m_flowMap[stream];
    return execFlow->isRuning();
}

bool AscendSimuDevice::StreamExecFlowPushOpHandle(aclrtStream stream, aclopHandle *opHandle)
{
    if (m_flowMap.find(stream) == m_flowMap.end()) {
        return false;
    }
    return m_flowMap[stream]->put(opHandle);
}

bool AscendSimuDevice::StreamExecFlowPushOpHandle(aclrtStream stream, aclrtCallback fn, void *userData)
{
    if (m_flowMap.find(stream) == m_flowMap.end()) {
        return false;
    }
    return m_flowMap[stream]->put(fn, userData);
}

void AscendSimuDevice::CreateContext(aclrtContext *ctxt)
{
    (*ctxt) = new (tagAclrtContext);
    (*ctxt)->deviceId = m_deviceId;
    CreateStream((*ctxt), &((*ctxt)->m_defaultStream));
    m_ctxts.push_back(*ctxt);
    m_activeContexts[m_deviceId] = (*ctxt);
}

void AscendSimuDevice::DestoryContext(aclrtContext ctxt)
{
    if (ctxt != nullptr) {
        if (ctxt->m_defaultStream != nullptr) {
            DestoryStream(ctxt->m_defaultStream);
            delete ctxt->m_defaultStream;
            ctxt->m_defaultStream = nullptr;
        }
        delete ctxt;
        ctxt = nullptr;
    }
}

aclrtContext AscendSimuDevice::GetActiveContext()
{
    return m_activeContexts[m_deviceId];
}

void AscendSimuDevice::SetActiveContext(aclrtContext ctxt)
{
    m_activeContexts[m_deviceId] = ctxt;
}

void AscendSimuDevice::CreateStream(aclrtContext ctxt, aclrtStream *stream)
{
    if (ctxt == nullptr) {
        ctxt = m_defaultContext;
    }

    (*stream) = new tagAclrtStream;
    (*stream)->deviceId = m_deviceId;
    (*stream)->ctxt = ctxt;
    (*stream)->ctxt->streams.push_back(*stream);
    CreateStreamExecFlow(*stream);
    // m_threadIdMap[*stream] = {};
}

void AscendSimuDevice::DestoryStream(aclrtStream stream)
{
    DestoryStreamExecFlow(stream);
    stream->ctxt->streams.erase(
        std::find_if(stream->ctxt->streams.begin(), stream->ctxt->streams.end(), [stream](aclrtStream _stream) {
            return stream == _stream;
        }));
}

void AscendSimuDevice::WaitStreamExecFlowIdle(aclrtStream stream)
{
    m_flowMap[stream]->waitForEmpty();
}
