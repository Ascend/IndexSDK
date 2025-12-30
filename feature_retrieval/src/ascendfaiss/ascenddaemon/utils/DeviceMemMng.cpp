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


#include "DeviceMemMng.h"
#include "common/utils/LogUtils.h"

using namespace ascend;

DeviceMemMng::DeviceMemMng() : strategy(DevMemStrategy::PURE_DEVICE_MEM)
{
    APP_LOG_INFO("using PURE_DEVICE_MEM strategy");
}

DeviceMemMng::~DeviceMemMng() {}

APP_ERROR DeviceMemMng::InitHmm(uint32_t deviceId, size_t hostCapacity)
{
    HmmMemoryInfo memoryInfo(deviceId, deviceCapacity, deviceBuffer, hostCapacity, 0);
    hmm = HmmIntf::CreateHmm();
    if (hmm == nullptr) {
        APP_LOG_ERROR("CreateHmm failed");
        return APP_ERR_INNER_ERROR;
    }
    auto ret = hmm->Init(memoryInfo);
    if (ret != APP_ERR_OK) {
        APP_LOG_ERROR("hmm Init failed, ret[%d]", ret);
        return ret;
    }
    return APP_ERR_OK;
}

APP_ERROR DeviceMemMng::SetHeteroParam(uint32_t deviceId, size_t deviceCapacity, size_t deviceBuffer,
    size_t hostCapacity, size_t devVecSize)
{
    this->deviceCapacity = deviceCapacity;
    this->deviceBuffer = deviceBuffer;

    if (deviceBuffer <= devVecSize) {
        APP_LOG_ERROR("deviceBuffer[%zu] is too small than devVecSize[%zu]", deviceBuffer, devVecSize);
        return APP_ERR_INVALID_PARAM;
    }
    // buff空间一半的大小做为每次push/pull的节点
    halfBlockNumOfBuff = utils::divDown(deviceBuffer, devVecSize) >> 1;

    APP_LOG_INFO("using HETERO_MEM strategy, deviceCapacity[%zu], deviceBuffer[%zu], hostCapacity[%zu]",
        deviceCapacity, deviceBuffer, hostCapacity);

    auto ret = InitHmm(deviceId, hostCapacity);
    if (ret != APP_ERR_OK) {
        APP_LOG_ERROR("load ock hmm err[%d]", ret);
        return APP_ERR_INNER_ERROR;
    }

    return APP_ERR_OK;
}

bool DeviceMemMng::UsingGroupSearch() const
{
    return this->strategy == DevMemStrategy::HETERO_MEM;
}

void DeviceMemMng::SetHeteroStrategy()
{
    APP_LOG_INFO("Set HETERO_MEM strategy");
    strategy = DevMemStrategy::HETERO_MEM;
}

DevMemStrategy DeviceMemMng::GetStrategy() const
{
    return strategy;
}

size_t DeviceMemMng::GetDeviceCapacity() const
{
    return this->deviceCapacity;
}

size_t DeviceMemMng::GetDeviceBuffer() const
{
    return this->deviceBuffer;
}
