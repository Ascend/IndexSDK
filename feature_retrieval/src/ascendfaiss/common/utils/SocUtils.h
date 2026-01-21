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

#include <cstring>
#include <errno.h>
#include <vector>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include "acl/acl.h"
#include "common/utils/LogUtils.h"

namespace faiss {
namespace ascend {

constexpr int MAX_DEV_LENGTH = 1024;
constexpr int MAX_DAVINCI_LENGTH = 17;
class SocUtils {
    enum SocType {
        SOC_310 = 0,
        SOC_310P,
        SOC_910B1,
        SOC_910B2,
        SOC_910B3,
        SOC_910B4,
        SOC_910_9392,
        SOC_910_9382,
        SOC_910_9372,
        SOC_910_9362,
        SOC_910_9579,
        SOC_INVALID,
    };

    const std::string SOCNAME310 = "Ascend310";
    const std::string SOCNAME710 = "Ascend710";
    const std::string SOCNAME310P = "Ascend310P";
    const std::string SOCNAME910B1 = "Ascend910B1";
    const std::string SOCNAME910B2 = "Ascend910B2";
    const std::string SOCNAME910B3 = "Ascend910B3";
    const std::string SOCNAME910B4 = "Ascend910B4";
    const std::string SOCNAME910_9392 = "Ascend910_9392";
    const std::string SOCNAME910_9382 = "Ascend910_9382";
    const std::string SOCNAME910_9372 = "Ascend910_9372";
    const std::string SOCNAME910_9362 = "Ascend910_9362";
    const std::string SOCNAME910_9579 = "Ascend910_95";

    enum CodeFormatType {
        FORMAT_TYPE_ZZ = 0,
        FORMAT_TYPE_ND,
        FORMAT_TYPE_INVALID,
    };

    struct SocAttribute {
        SocType socType;
        int coreNum;
        CodeFormatType codeFormType;
    };

public:
    SocUtils()
    {
        APP_LOG_DEBUG("construct SocUtils ------------- \n");
        if (access("/dev/davinci_manager", F_OK) == 0) {
            hasDev = true;
        }
        if (!hasDev) {
            deviceCount = 0;
            APP_LOG_WARNING("these is no devices on this machine,set parameter to default value\n");
            return;
        }
        auto ret = aclInit(nullptr);
        APP_LOG_INFO("aclInit return %d\n", ret);
        if (ret == ACL_ERROR_REPEAT_INITIALIZE) {
            APP_LOG_WARNING("acl has allready init by other component");
            repeatInitFlag = true;
        } else if (ret != ACL_SUCCESS) {
            APP_LOG_ERROR("Failed to init acl, ret:%d\n", ret);
            return;
        }
        Init();
    }

    ~SocUtils()
    {
        APP_LOG_DEBUG("destroy ~SocUtils ------------- \n");
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
        return socAttr.coreNum;
    }

    bool IsAscend310P() const
    {
        return socAttr.socType == SOC_310P;
    }

    bool IsAscend310() const
    {
        return socAttr.socType == SOC_310;
    }

    bool IsAscend910B() const
    {
        return (socAttr.socType == SOC_910B1) || (socAttr.socType == SOC_910B2) ||
            (socAttr.socType == SOC_910B3) || (socAttr.socType == SOC_910B4) ||
            IsAscendA3(); // 这里是表示A3和A2的device结构一样，底库保存格式一样
    }

    bool IsAscendA3() const
    {
        return (socAttr.socType == SOC_910_9392) || (socAttr.socType == SOC_910_9382) ||
            (socAttr.socType == SOC_910_9372) || (socAttr.socType == SOC_910_9362);
    }

    bool IsAscendA5() const
    {
        return (socAttr.socType == SOC_910_9579);
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

    bool IsRunningInHost() const
    {
        return (runMode == ACL_HOST);
    }

    CodeFormatType GetCodeFormatType() const
    {
        return socAttr.codeFormType;
    }

    bool IsZZCodeFormat() const
    {
        return socAttr.codeFormType == CodeFormatType::FORMAT_TYPE_ZZ;
    }

    bool ParseA2Data(const std::string &name)
    {
        if (name.find(SOCNAME910B4) == 0) {
            socAttr.socType = SOC_910B4;
            socAttr.coreNum = 40; // Ascend910B4 has 40 aiv
            socAttr.codeFormType = FORMAT_TYPE_ND;
            return true;
        } else if (name.find(SOCNAME910B3) == 0) {
            socAttr.socType = SOC_910B3;
            socAttr.coreNum = 40; // Ascend910B3 has 40 aiv
            socAttr.codeFormType = FORMAT_TYPE_ND;
            return true;
        } else if (name.find(SOCNAME910B2) == 0) {
            socAttr.socType = SOC_910B2;
            socAttr.coreNum = 48; // Ascend910B2 has 48 aiv
            socAttr.codeFormType = FORMAT_TYPE_ND;
            return true;
        } else if (name.find(SOCNAME910B1) == 0) {
            socAttr.socType = SOC_910B1;
            socAttr.coreNum = 48; // Ascend910B1 has 48 aiv
            socAttr.codeFormType = FORMAT_TYPE_ND;
            return true;
        }

        return false;
    }

    bool ParseA3Data(const std::string &name)
    {
        if (name.find(SOCNAME910_9392) == 0) {
            socAttr.socType = SOC_910_9392;
            socAttr.coreNum = 48; // Ascend910_9392 has 48 aiv
            socAttr.codeFormType = FORMAT_TYPE_ND;
            return true;
        } else if (name.find(SOCNAME910_9382) == 0) {
            socAttr.socType = SOC_910_9382;
            socAttr.coreNum = 48; // Ascend910_9382 has 48 aiv
            socAttr.codeFormType = FORMAT_TYPE_ND;
            return true;
        } else if (name.find(SOCNAME910_9372) == 0) {
            socAttr.socType = SOC_910_9372;
            socAttr.coreNum = 40; // Ascend910_9372 has 40 aiv
            socAttr.codeFormType = FORMAT_TYPE_ND;
            return true;
        } else if (name.find(SOCNAME910_9362) == 0) {
            socAttr.socType = SOC_910_9362;
            socAttr.coreNum = 40; // Ascend910_9362 has 40 aiv
            socAttr.codeFormType = FORMAT_TYPE_ND;
            return true;
        }

        return false;
    }

    bool ParseA5Data(const std::string &name)
    {
        if (name.find(SOCNAME910_9579) == 0) {
            socAttr.socType = SOC_910_9579;
            socAttr.coreNum = 56; // Ascend910_95 has 56 aiv
            socAttr.codeFormType = FORMAT_TYPE_ND;
            return true;
        }

        return false;
    }

    SocUtils(const SocUtils&) = delete;
    SocUtils& operator=(const SocUtils&) = delete;

private:
    void Init()
    {
        // init soc type
        auto socName = aclrtGetSocName();
        if (socName == nullptr) {
            APP_LOG_ERROR("aclrtGetSocName() return nullptr. please check your environment "
                "variables and compilation options to make sure you use the correct ACL library.");
            return;
        }
        const std::string name(socName);
        APP_LOG_DEBUG("get socname:%s", socName);
        if (name.find(SOCNAME710) == 0 || name.find(SOCNAME310P) == 0) {
            socAttr.socType = SOC_310P;
            socAttr.coreNum = 8; // Ascend310P has 8 aicore
            socAttr.codeFormType = FORMAT_TYPE_ZZ;
        } else if (name.find(SOCNAME310) == 0) {
            socAttr.socType = SOC_310;
            socAttr.coreNum = 2; // Ascend310 has 2 aicore
            socAttr.codeFormType = FORMAT_TYPE_ZZ;
        } else if (ParseA2Data(name)) {
            APP_LOG_INFO("find socname A2");
        } else if (ParseA3Data(name)) {
            APP_LOG_INFO("find socname A3");
        } else if (ParseA5Data(name)) {
            APP_LOG_INFO("find socname A5");
        } else {
            APP_LOG_ERROR("soc error. please check.");
        }

        // init device count
        auto ret = aclrtGetDeviceCount(&deviceCount);
        if (ret != ACL_SUCCESS) {
            APP_LOG_ERROR("Fail to get device count ret: %d", ret);
        }

        ret = aclrtGetRunMode(&runMode);
        if (ret != ACL_SUCCESS) {
            APP_LOG_ERROR("Fail to get run mode ret: %d", ret);
        }

        APP_LOG_INFO("Finish initing SocUtils, soc type: %u, core num: %d, device count %u",
            socAttr.socType, socAttr.coreNum, deviceCount);
    }

    void Finalize()
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
        if (finalizeEnv[0] == setOff) {
            APP_LOG_DEBUG("set env, not exec aclFinalize ------------- \n");
        } else {
            APP_LOG_DEBUG("set env, exec aclFinalize ------------- \n");
            (void)aclFinalize();
        }
    }

private:
    SocAttribute socAttr {SOC_INVALID, 0, FORMAT_TYPE_INVALID};
    uint32_t deviceCount;
    bool hasDev = false;
    aclrtRunMode runMode { ACL_HOST };
    bool repeatInitFlag = false;
};
}
}

#endif // FEATURERETRIEVAL_SOCUTILS_H