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


#include "AscendSimuEnv.h"

thread_local int32_t AscendSimuEnv::m_activeDeviceId = -1;

const char *AscendSimuEnv::getSocName() const
{
    return m_socName == nullptr ? "Ascend310P" : m_socName;
}

uint32_t AscendSimuEnv::getDeviceCount() const
{
    return m_devices.size();
}

aclrtRunMode AscendSimuEnv::getRunMode() const
{
    return m_runMode;
}

bool AscendSimuEnv::construct(const char *socName,
                              aclrtRunMode runMode,
                              std::vector<AscendSimuDevice *> deviceList)
{
    m_socName = socName;
    m_devices = deviceList;
    m_runMode = runMode;

    if (m_devices.size() > MAX_DEVICE) {
        return false;
    }
    return true;
}

void AscendSimuEnv::destruct()
{
    for (auto device : m_devices) {
        delete device;
    }
    m_devices.clear();

    m_runMode = ACL_HOST;
    m_socName = nullptr;
}

void AscendSimuEnv::SetDevice(int32_t deviceId)
{
    m_devices[deviceId]->Init();
    m_activeDeviceId = deviceId;
}

void AscendSimuEnv::ReSetDevice(int32_t deviceId)
{
    m_devices[deviceId]->DeInit();
    if (deviceId == m_activeDeviceId && m_devices[deviceId]->GetRefCnt() == 0) {
        m_activeDeviceId = -1;
    }
}

AscendSimuDevice *AscendSimuEnv::getDevice(int32_t deviceId)
{
    if (deviceId < 0 || deviceId > 7) {
        deviceId = 0;
    }
    return m_devices[deviceId];
}

int32_t AscendSimuEnv::getActiveDeviceId()
{
    return m_activeDeviceId;
}

void AscendSimuEnv::setActiveDeviceId(int32_t deviceId)
{
    m_activeDeviceId = deviceId;
}
