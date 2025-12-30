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


#ifndef UT_ASCENDSIMUENV_H
#define UT_ASCENDSIMUENV_H

#include <string>
#include <cstdint>
#include <thread>
#include "../acl.h"
#include "AscendSimuDevice.h"

// 昇腾模拟环境类 使用昇腾ACL接口的之前需要再测试用力的testUP中先初始化。
class AscendSimuEnv {
public:
    static AscendSimuEnv &getIns()
    {
        static AscendSimuEnv simuEnv;
        return simuEnv;
    }

    bool construct(const char *socName,
                   aclrtRunMode runMode,
                   std::vector<AscendSimuDevice *> deviceList = {});
    void destruct();
    void SetDevice(int32_t deviceId);
    void ReSetDevice(int32_t deviceId);
    AscendSimuDevice *getDevice(int32_t deviceId);
    const char *getSocName() const;
    uint32_t getDeviceCount() const;
    aclrtRunMode getRunMode() const;

    static int32_t getActiveDeviceId();
    static void setActiveDeviceId(int32_t deviceId);

    AscendSimuEnv(const AscendSimuEnv&) = delete;
    AscendSimuEnv &operator=(const AscendSimuEnv&) = delete;
private:
    AscendSimuEnv() = default;
    virtual ~AscendSimuEnv() = default;

    const char *m_socName {nullptr}; // 芯片名称
    std::vector<AscendSimuDevice *> m_devices; // 模拟设备
    aclrtRunMode m_runMode {ACL_HOST};
    static thread_local int32_t m_activeDeviceId; // 每个线程激活的deivceId不一样
};

#define ENV() AscendSimuEnv::getIns()
#define DEVICE(deviceId)  ENV().getDevice(deviceId)

#endif // UT_ASCENDSIMUENV_H
