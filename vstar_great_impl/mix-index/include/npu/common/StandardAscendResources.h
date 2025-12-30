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


#ifndef ASCEND_STANDARDASCENDRESOURCES_INCLUDED
#define ASCEND_STANDARDASCENDRESOURCES_INCLUDED

#include <map>
#include <vector>
#include <memory>
#include <mutex>

#include "npu/common/utils/AscendMemory.h"
#include "npu/common/utils/DistComputeOpsManager.h"
#include "npu/common/utils/AscendStackMemory.h"
#include "npu/common/utils/LogUtils.h"
#include "npu/common/threadpool/AscendThreadPool.h"

namespace ascendSearchacc {
class StandardAscendResources {
public:
    // as th rpc interface is called serially, it does not need to consider multi threading
    static std::shared_ptr<StandardAscendResources> getInstance()
    {
        static std::mutex mtx;
        static std::map<int, std::shared_ptr<StandardAscendResources> > resources;
        int deviceId = 0;
        auto ret = aclrtGetDevice(&deviceId);
        if (ret != ACL_SUCCESS) {
            APP_LOG_ERROR("failed to rt get device, acl ret: %d", ret);
            deviceId = 0;
        }
        // the obj should be created after set device
        std::lock_guard<std::mutex> lk(mtx);
        if (resources.find(deviceId) == resources.end()) {
            auto res = std::make_shared<StandardAscendResources>();
            res->setDevice(deviceId);
            resources[deviceId] = res;
        }
        return resources[deviceId];
    }

    ~StandardAscendResources();

    explicit StandardAscendResources(StandardAscendResources &manager) = delete;

    explicit StandardAscendResources(StandardAscendResources &&manager) = delete;

    explicit StandardAscendResources(std::string modelPath = "modelpath");

    StandardAscendResources &operator=(StandardAscendResources &manager) = delete;

    StandardAscendResources &operator=(StandardAscendResources &&manager) = delete;

    void setDevice(int deviceId);

    // / Disable allocation of temporary memory; all temporary memory
    // / requests will call aclrtMalloc / aclrtFree at the point of use
    void noTempMemory();

    // / Specify that we wish to use a certain fixed size of memory on as
    // / temporary memory. This is the upper bound for the Ascend Device
    // / memory that we will reserve.
    // / To avoid any temporary memory allocation, pass 0.
    void setTempMemory(size_t size);

    void setDefaultTempMemory();

    // / Enable or disable the warning about not having enough temporary memory
    // / when aclrtMalloc gets called
    void setAscendMallocWarning(bool flag);

    void resetStack();

    size_t getResourceSize() const
    {
        return tempMemSize;
    }

public:
    // / Initialize resources
    void initialize();

    void uninitialize();

    aclrtStream getDefaultStream();

    void syncDefaultStream();

    std::vector<aclrtStream> getAlternateStreams();

    AscendMemory &getMemoryManager();

public:
    std::unique_ptr<AscendThreadPool> threadPool;
    std::unique_ptr<AscendThreadPool> threadPoolMulSearch;
    // / Our default stream that work is ordered on
    aclrtStream defaultStream;

private:

    // / Have Ascend resources been initialized yet?
    bool isInitialized() const;

    size_t getDefaultTempMemSize(size_t requested);

private:
    int deviceId = 0;

    // / Other streams we can use
    std::vector<aclrtStream> alternateStreams;

    // / Another option is to use a specified amount of memory
    size_t tempMemSize;

    AscendStackMemory ascendStackMemory;
};
}  // namespace ascendSearchacc

#endif  // ASCEND_STANDARDASCENDRESOURCES_INCLUDED
