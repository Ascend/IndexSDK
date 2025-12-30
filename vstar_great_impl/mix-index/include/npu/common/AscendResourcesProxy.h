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


#ifndef ASCEND_ASCEND_RESOURCES_PROXY_INCLUDED
#define ASCEND_ASCEND_RESOURCES_PROXY_INCLUDED

#include "npu/common/StandardAscendResources.h"
#include "npu/common/utils/AscendStackMemory.h"
#include "npu/common/utils/LogUtils.h"
#include "npu/common/threadpool/AscendThreadPool.h"

#include "acl/acl.h"

namespace ascendSearchacc {
class AscendResourcesProxy {
public:
    AscendResourcesProxy() : resources(StandardAscendResources::getInstance())
    {
        APP_LOG_INFO("AscendResourcesProxy Construction");
    }

    ~AscendResourcesProxy()
    {
        APP_LOG_INFO("AscendResourcesProxy Destruction");
        // get the use count after release
        int useCountReleased = resources.use_count() - 1;
        if (useCountReleased == 1) {
            resources->uninitialize();
            resources->resetStack();
        }
    };

    // Disable allocation of temporary memory; all temporary memory
    // requests will call aclrtMalloc / aclrtFree at the point of use
    inline void noTempMemory()
    {
        resources->noTempMemory();
    }

    // Specify that we wish to use a certain fixed size of memory on as
    // temporary memory. This is the upper bound for the Ascend Device
    // memory that we will reserve.
    // To avoid any temporary memory allocation, pass 0.
    inline void setTempMemory(size_t size)
    {
        resources->setTempMemory(size);
    }

    inline void setDefaultTempMemory()
    {
        resources->setDefaultTempMemory();
    }

    // Enable or disable the warning about not having enough temporary memory
    // when aclrtMalloc gets called
    inline void setAscendMallocWarning(bool flag)
    {
        resources->setAscendMallocWarning(flag);
    }

    inline size_t getResourceSize() const
    {
        return resources->getResourceSize();
    }

    // Initialize resources
    inline void initialize()
    {
        resources->initialize();
    }

    inline aclrtStream getDefaultStream() const
    {
        return resources->getDefaultStream();
    }

    inline void syncDefaultStream() const
    {
        return resources->syncDefaultStream();
    }

    inline std::vector<aclrtStream> getAlternateStreams() const
    {
        return resources->getAlternateStreams();
    }

    inline AscendMemory &getMemoryManager() const
    {
        return resources->getMemoryManager();
    }

    std::unique_ptr<AscendThreadPool> &getThreadPool()
    {
        return resources->threadPool;
    }

    std::unique_ptr<AscendThreadPool> &getThreadPoolMulSearch()
    {
        return resources->threadPoolMulSearch;
    }

private:
    std::shared_ptr<StandardAscendResources> resources;
};
}  // namespace ascendSearchacc

#endif  // ASCEND_ASCEND_RESOURCES_PROXY_INCLUDED
