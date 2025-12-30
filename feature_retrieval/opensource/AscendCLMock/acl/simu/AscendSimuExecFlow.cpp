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

#include <cstring>
#include <utility>
#include <iostream>
#include "../acl_base.h"
#include "AscendSimuExecFlow.h"

std::map<const char *, AscendSimuExecFlow::SimuExecOpFunc> AscendSimuExecFlow::m_opExecMap {};
std::mutex AscendSimuExecFlow::m_OpMut;

struct aclrtCallbackInfo {
    aclrtCallback callback;
    void *userData;
};

AscendSimuExecFlow::~AscendSimuExecFlow()
{
    stop();
}

bool AscendSimuExecFlow::start()
{
    if (isRuning()) {
        return false;
    }

    m_running.store(true, std::memory_order_release);
    m_execFlow = new std::thread(&AscendSimuExecFlow::run, this);
    return true;
}

bool AscendSimuExecFlow::stop()
{
    if (!isRuning()) {
        return false;
    }

    if (!m_execFlow->joinable()) {
        return false;
    }

    m_running.store(false, std::memory_order_release);
    m_cond.notify_one();
    m_execFlow->join();
    delete m_execFlow;
    m_execFlow = nullptr;
    return true;
}

bool AscendSimuExecFlow::isRuning()
{
    return m_running.load(std::memory_order_acquire);
}

bool AscendSimuExecFlow::put(aclopHandle *opHandle)
{
    std::lock_guard<std::mutex> lk(m_mut);
    callbackHandle *handle = new callbackHandle;
    handle->execType = 0;
    handle->opHandle = (void *)opHandle;
    m_queue.push(handle);
    m_cond.notify_one();
    return true;
}

bool AscendSimuExecFlow::put(aclrtCallback fn, void *userData)
{
    std::lock_guard<std::mutex> lk(m_mut);
    aclrtCallbackInfo *cbInfo = new aclrtCallbackInfo;
    cbInfo->callback = fn;
    cbInfo->userData = userData;
    callbackHandle *handle = new callbackHandle;
    handle->execType = 1;
    handle->opHandle = (void *)cbInfo;
    m_queue.push(handle);
    m_cond.notify_one();
    return true;
}

void AscendSimuExecFlow::run()
{
    while (isRuning()) {
        callbackHandle *opHandle = nullptr;
        {
            std::unique_lock<std::mutex> lk{m_mut};
            m_cond.wait_for(lk, std::chrono::milliseconds(COND_WAIT_TIME_OUT),
                            [this] { return !m_queue.empty() && isRuning(); });

            if (!isRuning()) {
                return;
            }

            if (!m_queue.empty()) {
                opHandle = m_queue.front();
                m_queue.pop();
                exec(opHandle);
            }

            if (m_queue.empty()) {
                m_emptyCond.notify_one();
            }
        }
    }
}

void AscendSimuExecFlow::exec(callbackHandle *cbHandle)
{
    if (cbHandle->execType == 0) {
        aclopHandle *opHandle = (aclopHandle *)cbHandle->opHandle;
        SimuExecOpFunc opFunc;
        {
            // std::lock_guard<std::mutex> lk(m_OpMut);
            opFunc = TryGetOpFunc(opHandle->opName);
        }

        if (opFunc != nullptr) {
            opFunc(*opHandle);
            delete []opHandle->inputData;
            delete []opHandle->outputData;
            delete []opHandle->opName;
            delete opHandle;
            delete cbHandle;
        }
    } else {
        aclrtCallbackInfo *opHandle = (aclrtCallbackInfo *)cbHandle->opHandle;
        opHandle->callback(opHandle->userData);
        delete opHandle;
        delete cbHandle;
    }
}

void AscendSimuExecFlow::waitForEmpty()
{
    if (!isRuning()) {
        return;
    }
    std::unique_lock<std::mutex> lk(m_mut);
    if (!m_queue.empty()) {
        m_emptyCond.wait(lk, [this] { return m_queue.empty(); });
    }
}

void AscendSimuExecFlow::reg(const char *opName, AscendSimuExecFlow::SimuExecOpFunc func)
{
    std::lock_guard<std::mutex> lk(m_OpMut);
    m_opExecMap[opName] = std::move(func);
}

void AscendSimuExecFlow::unReg(const char *opName)
{
    std::lock_guard<std::mutex> lk(m_OpMut);
    if (!TryIsOpReg(opName)) {
        return;
    }
    m_opExecMap.erase(opName);
}

bool AscendSimuExecFlow::IsOpReg(const char *opName)
{
    std::lock_guard<std::mutex> lk(m_OpMut);
    return TryIsOpReg(opName);
}

bool AscendSimuExecFlow::TryIsOpReg(const char *opName)
{
    for (auto &op : m_opExecMap) {
        const char *opStr = op.first;
        if (!strcmp(opStr, opName)) {
            return true;
        }
    }
    return false;
}

AscendSimuExecFlow::SimuExecOpFunc AscendSimuExecFlow::TryGetOpFunc(const char *opName)
{
    for (auto &op : m_opExecMap) {
        const char *opStr = op.first;
        if (!strcmp(opStr, opName)) {
            return op.second;
        }
    }
    return nullptr;
}

void AscendSimuExecFlow::ShowOp()
{
    printf("------RegOpCnt:%d------\n", m_opExecMap.size());
    for (auto &op : m_opExecMap) {
        const char *opStr = op.first;
        ACL_APP_LOG(ACL_INFO, "RegOp:%s\n", opStr);
    }
}
