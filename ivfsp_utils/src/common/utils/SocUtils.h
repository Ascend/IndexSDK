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


#ifndef FEATURERETRIEVAL_SOCUTILS_H
#define FEATURERETRIEVAL_SOCUTILS_H

#include <string>
#include <cstring>
#include <cerrno>
#include <iostream>
#include <cstdio>
#include <unistd.h>
#include "common/utils/LogUtils.h"
#include "acl/acl.h"

namespace faiss {
namespace ascendSearch {

constexpr int MAX_DEV_LENGTH = 1024;
constexpr int MAX_DAVINCI_LENGTH = 17;

class SocUtils {
enum SocType {
    SOC_310 = 0,
    SOC_310P,
    SOC_910,
    SOC_INVALID,
};

const std::string socName310 = "Ascend310";
const std::string socName710 = "Ascend710";
const std::string socName310P = "Ascend310P";
const std::string socName910 = "Ascend910";

public:
    SocUtils()
    {
        APP_LOG_DEBUG("construct SocUtils ------------- \n");
        if (access("/dev/davinci_manager", F_OK) == 0) {
            hasDev = true;
        }
        if (!hasDev) {
            coreNum = 0;
            socType = SOC_INVALID;
            deviceCount = 0;
            APP_LOG_WARNING("these is no devices on this machine,set parameter to default value\n");
            return;
        }
        auto ret = aclInit(nullptr);
        APP_LOG_INFO("call aclInit return %d\n", ret);
        if (ret == ACL_ERROR_REPEAT_INITIALIZE) {
            APP_LOG_WARNING("warning acl has allready init by other component");
            repeatInitFlag = true;
        } else if (ret != ACL_SUCCESS) {
            APP_LOG_ERROR("acl init failed, ret:%d\n", ret);
            return;
        }
        Init();
    }

    ~SocUtils()
    {
        APP_LOG_DEBUG("call ~SocUtils ------------- \n");
        if (!hasDev) {
            return;
        }
        Finalize();
    }

public:
    static SocUtils &GetInstance()
    {
        static SocUtils socUtils;
        return socUtils;
    }

    int GetCoreNum() const
    {
        return coreNum;
    }

    bool IsAscend310P() const
    {
        return socType == SOC_310P;
    }

    bool IsAscend310() const
    {
        return socType == SOC_310;
    }

    int GetExtremeListSize() const
    {
        // Extreme list size is 1024 for Ascend310 cases and extreme list size is 2048 for Ascend310P cases
        return IsAscend310() ? 1024 : 2048;
    }

    int GetHandleBatch() const
    {
        // Handle batch is 4 for Ascend310 cases and handle batch is 8 for Ascend310P cases
        return IsAscend310() ? 4 : 8;
    }

    int GetSearchListSize() const
    {
        // Search list size is 8192 for Ascend310 cases and search list size is 32768 for Ascend310P cases
        return IsAscend310() ? 8192 : 32768;
    }

    int GetThreadsCnt() const
    {
        // Ascend310 can use 6 ctrlcpu at most and Ascend310P can use 7 ctrlcpu at most
        return IsAscend310() ? 6 : 7;
    }

    int GetSqopInputNum() const
    {
        // DistanceIVFSQ8IP4 used in Ascend310 has 7 inputs and DistanceIVFSQ8IP8 used in Ascend310P has 5 inputs
        return IsAscend310() ? 7 : 5;
    }

    int GetQueryBatch() const
    {
        // Query batch is 8 for Ascend310 cases and query batch is 16 for Ascend310P cases
        return IsAscend310() ? 8 : 16;
    }

    uint32_t GetDeviceCount() const
    {
        return deviceCount;
    }

private:
    void Init()
    {
        // init soc type
        coreNum = 0;
        socType = SOC_INVALID;
        auto socName = aclrtGetSocName();
        if (socName == nullptr) {
            APP_LOG_ERROR("aclrtGetSocName() return nullptr. please check your environment "
                "variables and compilation options to make sure you use the correct ACL library.");
            return;
        }
        const std::string name(socName);
        if (name.find(socName710) == 0 || name.find(socName310P) == 0) {
            socType = SOC_310P;
            coreNum = 8; // Ascend310P has 8 aicore
        } else if (name.find(socName310) == 0) {
            socType = SOC_310;
            coreNum = 2; // Ascend310 has 2 aicore
        } else if (name.find(socName910) == 0) {
            socType = SOC_910;
            coreNum = 32; // Ascend910 has 32 aicore
        } else {
            APP_LOG_ERROR("soc error. please check.");
        }

        // init device count
        auto ret = aclrtGetDeviceCount(&deviceCount);
        if (ret != ACL_SUCCESS) {
            APP_LOG_ERROR("Fail to get device count ret: %d", ret);
        }

        APP_LOG_INFO("Finish initing SocUtils, soc type: %u, core num: %d, device count %u",
            socType, coreNum, deviceCount);
    }

    void Finalize() const
    {
        const char setOff = '0';
        const char setOn = '1';
        const size_t envLen = 1;
        char *finalizeEnv = std::getenv("MX_INDEX_FINALIZE");
        bool validEnv = ((finalizeEnv != nullptr) && (std::strlen(finalizeEnv) == envLen) &&
            ((finalizeEnv[0] == setOff) || (finalizeEnv[0] == setOn)));
        if (!validEnv) {
            if (repeatInitFlag) {
                APP_LOG_DEBUG("do not aclFinalize ------------- \n");
            } else {
                APP_LOG_DEBUG("exec aclFinalize ------------- \n");
                (void)aclFinalize();
            }
            return;
        }
        APP_LOG_DEBUG("set env -------------%s \n", finalizeEnv);
        return;
    }

private:
    int coreNum = 0;
    SocType socType = SOC_INVALID;
    uint32_t deviceCount = 0;
    bool hasDev = false;
    bool repeatInitFlag = false;
};
}
}

#endif // FEATURERETRIEVAL_SOCUTILS_H