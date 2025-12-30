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
#include <mutex>
#include <vector>

#include "ascenddaemon/utils/AscendUtils.h"

#include "common/utils/SocUtils.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"

__attribute__ ((weak)) aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout);
namespace {
constexpr int32_t SYNCHRONIZE_STREAM_TIME_OUT = 300000;   // 5 minute
constexpr size_t MAX_SYNCHRONIZE_STREAM_TIME_SIZE = 7;
constexpr size_t MIN_SYNCHRONIZE_STREAM_TIME_SIZE = 5;
constexpr int32_t MAX_SYNCHRONIZE_STREAM_TIME = 1800000;  // 30 minute
constexpr int32_t MIN_SYNCHRONIZE_STREAM_TIME = 60000;  // 1 minute
int32_t g_synchronizeStreamTimeOut = SYNCHRONIZE_STREAM_TIME_OUT;
std::once_flag g_synchronizeStreamSetFlag;
} // namespace
namespace ascend {
static void GetSynchronizeStreamTime()
{
    char *time = std::getenv("MX_INDEX_SYNCHRONIZE_STREAM_TIME");
    if (time == nullptr) {
        APP_LOG_INFO("not set synchronizeStream time, use default");
        g_synchronizeStreamTimeOut = SYNCHRONIZE_STREAM_TIME_OUT;
        return;
    }
    std::string timeStr = std::string(time);
    if (timeStr.size() < MIN_SYNCHRONIZE_STREAM_TIME_SIZE || timeStr.size() > MAX_SYNCHRONIZE_STREAM_TIME_SIZE) {
        APP_LOG_ERROR("set invalid synchronizeStream time, it's size is invalid, use default");
        g_synchronizeStreamTimeOut = SYNCHRONIZE_STREAM_TIME_OUT;
        return;
    }
    if (!CommonUtils::IsNumber(timeStr)) {
        APP_LOG_ERROR("set invalid synchronizeStream time, not a numnber, use default");
        g_synchronizeStreamTimeOut = SYNCHRONIZE_STREAM_TIME_OUT;
        return;
    }
    try {
        int32_t timeNum = std::stoi(timeStr);
        if (timeNum >= MIN_SYNCHRONIZE_STREAM_TIME && timeNum <= MAX_SYNCHRONIZE_STREAM_TIME) {
            g_synchronizeStreamTimeOut = timeNum;
            APP_LOG_INFO("set valid synchronizeStream time: %d", g_synchronizeStreamTimeOut);
        } else {
            APP_LOG_ERROR("set invalid range synchronizeStream time, use default");
            g_synchronizeStreamTimeOut = SYNCHRONIZE_STREAM_TIME_OUT;
        }
    } catch (const std::exception &e) {
        APP_LOG_ERROR("transform synchronizeStream failed, use default");
        g_synchronizeStreamTimeOut = SYNCHRONIZE_STREAM_TIME_OUT;
    }
}

aclError synchronizeStream(aclrtStream stream)
{
    std::call_once(g_synchronizeStreamSetFlag, GetSynchronizeStreamTime);
    if (aclrtSynchronizeStreamWithTimeout != nullptr) {
        return aclrtSynchronizeStreamWithTimeout(stream, g_synchronizeStreamTimeOut);
    } else {
        return aclrtSynchronizeStream(stream);
    }
}
void AscendUtils::setCurrentDevice(int device)
{
    ACL_REQUIRE_OK(aclrtSetDevice(device));
}

void AscendUtils::resetCurrentDevice(int device)
{
    (void) aclrtResetDevice(device);
}

aclrtContext AscendUtils::getCurrentContext()
{
    aclrtContext ctx = 0;
    auto ret = aclrtGetCurrentContext(&ctx);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_ERROR_NONE, "aclrtGetCurrentContext fail %d", ret);
    return ctx;
}

void AscendUtils::setCurrentContext(aclrtContext ctx)
{
    ACL_REQUIRE_OK(aclrtSetCurrentContext(ctx));
}

bool AscendMultiThreadManager::IsMultiThreadMode()
{
    static std::once_flag onceFlag;
    static bool multiThreadFlag = false;
    std::call_once(onceFlag, GetMultiThreadMode, multiThreadFlag);
    return multiThreadFlag;
}

void AscendMultiThreadManager::GetMultiThreadMode(bool &multiThreadFlag)
{
    // 当前仅支持设置环境变量值为1
    const size_t envLen = 1;
    const char envVal = '1';
    char *multiThreadEnv = std::getenv("MX_INDEX_MULTITHREAD");
    multiThreadFlag = (multiThreadEnv != nullptr) && (std::strlen(multiThreadEnv) == envLen) &&
        (multiThreadEnv[0] == envVal);
    APP_LOG_WARNING("Get multithread switch[%d]\r\n", multiThreadFlag);
}

void AscendMultiThreadManager::InitGetBaseMtx(const std::vector<int> &deviceList,
    std::unordered_map<int, std::mutex> &getBaseMtx)
{
    if (!IsMultiThreadMode()) {
        return;
    }

    for (auto deviceId : deviceList) {
        (void)getBaseMtx[deviceId];
    }
}

std::optional<std::lock_guard<std::mutex>> AscendMultiThreadManager::LockGetBaseMtx(int deviceId,
    std::unordered_map<int, std::mutex> &getBaseMtx)
{
    if (!IsMultiThreadMode()) {
        return std::nullopt;
    }

    ASCEND_THROW_IF_NOT_FMT(getBaseMtx.find(deviceId) != getBaseMtx.end(), "Invalid deviceId[%d]", deviceId);

    auto &deviceMtx = getBaseMtx[deviceId];
    return std::optional<std::lock_guard<std::mutex>>(deviceMtx);
}

bool AscendOperatorManager::Init(std::string path)
{
    static bool isInited = false;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);
    if (std::getenv("MX_INDEX_USE_ONLINEOP")) {
        std::string useonline(std::string(std::getenv("MX_INDEX_USE_ONLINEOP")));
        ASCEND_THROW_IF_NOT_MSG(useonline.size() == 1, "len of MX_INDEX_USE_ONLINEOP more than 1");
        if (useonline == "1") {
            return false;
        }
    }
    if (isInited) {
        return true;
    }

#ifdef HOSTCPU
    char *modelpath = std::getenv("MX_INDEX_MODELPATH");
    struct stat fileStat;
    if (modelpath != nullptr) {
        auto res = CommonUtils::CheckSymLink(&fileStat, std::string(modelpath));
        ASCEND_THROW_IF_NOT_MSG(res == APP_ERR_OK, "Modelpath from env is symlink");
        path = CommonUtils::RealPath(std::string(modelpath));
        ASCEND_THROW_IF_NOT_MSG(!path.empty(), "Modelpath from env is invalid");
        ASCEND_THROW_IF_NOT_FMT(CommonUtils::CheckPathValid(path),
            "Modelpath from env: %s must be in home directory and readable", path.c_str());
        APP_LOG_INFO("Use env %s as modelpath", path.c_str());
    } else {
        auto res = CommonUtils::CheckSymLink(&fileStat, path);
        ASCEND_THROW_IF_NOT_MSG(res == APP_ERR_OK, "default Modelpath is symlink");
    }

#endif

    int retCode = aclopSetModelDir(path.c_str());
    ASCEND_THROW_IF_NOT_FMT(retCode == ACL_ERROR_NONE || retCode == ACL_ERROR_REPEAT_INITIALIZE,
        "ACL error %d", retCode);
    if (retCode == ACL_ERROR_REPEAT_INITIALIZE) {
        APP_LOG_WARNING("ACL error %d, repeate to execute aclopSetModelDir().", retCode);
    }
    isInited = true;
    return true;
}

AscendOperatorManager::~AscendOperatorManager()
{
}

DeviceScope::DeviceScope()
{
    AscendUtils::setCurrentDevice(0);
}

DeviceScope::~DeviceScope()
{
    AscendUtils::resetCurrentDevice(0);
}

std::mutex &AscendGlobalLock::GetInstance(uint32_t device)
{
    ASCEND_THROW_IF_NOT_FMT(device <= MAX_DEVICEID,
        "deviceId[%zu] not in range[0, %zu]", device, MAX_DEVICEID);
    static std::vector<std::mutex> gmtx(MAX_DEVICEID);
    return gmtx[device];
}
} // namespace ascend
