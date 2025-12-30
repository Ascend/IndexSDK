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

#include "ascenddaemon/utils/AscendMemory.h"
#include "ascenddaemon/utils/DistComputeOpsManager.h"
#include "ascenddaemon/utils/AscendStackMemory.h"
#include "common/utils/LogUtils.h"
#include "common/threadpool/AscendThreadPool.h"

namespace ascend {

class AscendStreamIntf {
public:
    explicit AscendStreamIntf(aclrtStream stream) : stream(stream) {}

    // GetStream在多线程场景下会加锁，在同一个调用栈下避免反复调用：一是会造成不必要的死锁等问题风险，二是影响性能
    virtual aclrtStream GetStream() = 0;

    virtual ~AscendStreamIntf() = default;

protected:
    aclrtStream stream;
};

class ThreadSafeAscendStream : public AscendStreamIntf {
public:
    ThreadSafeAscendStream(aclrtStream stream, std::recursive_mutex &streamMtx)
        : AscendStreamIntf(stream), streamMtx(streamMtx) {}
    ~ThreadSafeAscendStream() override = default;

    aclrtStream GetStream() override
    {
        if (!streamLock) {
            streamLock = std::unique_lock<std::recursive_mutex>(streamMtx);
        }
        return stream;
    }

private:
    std::unique_lock<std::recursive_mutex> streamLock;
    std::recursive_mutex &streamMtx;
};

class AclrtStreamAdaptor : public AscendStreamIntf {
public:
    explicit AclrtStreamAdaptor(aclrtStream stream) : AscendStreamIntf(stream) {}
    ~AclrtStreamAdaptor() override = default;

    aclrtStream GetStream() override
    {
        return stream;
    }
};

class StandardAscendResources {
public:
    // as th rpc interface is called serially, it does not need to consider multi threading
    static std::shared_ptr<StandardAscendResources> getInstance();

    explicit StandardAscendResources(std::shared_ptr<DistComputeOpsManager> opsMng,
        std::string modelPath = "modelpath");

    virtual ~StandardAscendResources();

    StandardAscendResources(StandardAscendResources& manager) = delete;
    StandardAscendResources(StandardAscendResources&& manager) = delete;
    StandardAscendResources& operator=(StandardAscendResources& manager) = delete;
    StandardAscendResources& operator=(StandardAscendResources&& manager) = delete;

    virtual void setDevice(int curDeviceId);

    /// Disable allocation of temporary memory; all temporary memory
    /// requests will call aclrtMalloc / aclrtFree at the point of use
    virtual void noTempMemory();

    /// Specify that we wish to use a certain fixed size of memory on as
    /// temporary memory. This is the upper bound for the Ascend Device
    /// memory that we will reserve.
    /// To avoid any temporary memory allocation, pass 0.
    virtual void setTempMemory(size_t size);

    virtual void setDefaultTempMemory();

    /// Enable or disable the warning about not having enough temporary memory
    /// when aclrtMalloc gets called
    virtual void setAscendMallocWarning(bool flag);

    virtual void resetStack();

    virtual size_t getResourceSize() const
    {
        return tempMemSize;
    }

    virtual void addUseCount();

    virtual void reduceUseCount();

public:
    /// Initialize resources
    bool useOnlineOp {false};
    virtual void initialize();

    virtual void uninitialize();

    virtual std::shared_ptr<AscendStreamIntf> getDefaultStream();

    virtual std::vector<std::shared_ptr<AscendStreamIntf>> getAlternateStreams();

    virtual AscendMemory& getMemoryManager();

protected:
    size_t getDefaultTempMemSize(size_t requested);

protected:
    /// Another option is to use a specified amount of memory
    size_t tempMemSize;

    AscendStackMemory ascendStackMemory;

    int deviceId = 0;

private:
    /// Have Ascend resources been initialized yet?
    bool isInitialized() const;

    virtual std::shared_ptr<AscendStreamIntf> createAscendStream(aclrtStream stream);

private:
    std::shared_ptr<DistComputeOpsManager> opsMng;
    /// Our default stream that work is ordered on
    aclrtStream defaultStream;
    /// Other streams we can use
    std::vector<aclrtStream> alternateStreams;
    size_t useCount = 0;
};
}  // namespace ascend

#endif  // ASCEND_STANDARDASCENDRESOURCES_INCLUDED
