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

#include "npu/NpuIndex.h"
#include "npu/common/utils/AscendAssert.h"
#include "npu/common/utils/LogUtils.h"
#include "npu/common/utils/CommonUtils.h"

namespace ascendSearchacc {
NpuIndex::NpuIndex(int dim, MetricType metricType, NpuIndexConfig config)
    : ascendConfig(config), trained(false), dim(dim), metricType(metricType)
{
    InitResources();
    resources = std::make_unique<AscendResourcesProxy>();
    if (ascendConfig.resourceSize == 0) {
        resources->noTempMemory();
    } else if (ascendConfig.resourceSize > 0) {
        resources->setTempMemory(ascendConfig.resourceSize);
    } else {
        // resourceSize < 0 means use default mem configure
        resources->setTempMemory(DEFAULT_RESOURCE_MEM);
    }

    resources->initialize();
}

NpuIndex::NpuIndex(NpuIndexConfig config) : ascendConfig(config)
{
    InitResources();
    resources = std::make_unique<AscendResourcesProxy>();
    if (ascendConfig.resourceSize == 0) {
        resources->noTempMemory();
    } else if (ascendConfig.resourceSize > 0) {
        resources->setTempMemory(ascendConfig.resourceSize);
    } else {
        // resourceSize < 0 means use default mem configure
        resources->setTempMemory(DEFAULT_RESOURCE_MEM);
    }

    resources->initialize();
}

NpuIndex::~NpuIndex()
{
}

APP_ERROR NpuIndex::InitResources()
{
    if (aclrtSetDevice(ascendConfig.deviceList[0]) != ACL_SUCCESS) {
        APP_LOG_ERROR("Set device failed. deviceId is %d", ascendConfig.deviceList[0]);
        (void)aclFinalize();
        return ACL_ERROR_FAILURE;
    }
    APP_LOG_INFO("Set device[%d] success", ascendConfig.deviceList[0]);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        APP_LOG_ERROR("Get run mode failed");
        return ACL_ERROR_FAILURE;
    }
    APP_LOG_INFO("Get RunMode[%d] success", runMode);

    return APP_ERR_OK;
}

}  // namespace ascendSearchacc
