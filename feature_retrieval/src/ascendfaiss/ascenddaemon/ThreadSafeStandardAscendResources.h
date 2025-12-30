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


#ifndef THREAD_SAFE_STANDARD_ASCEND_RESOURCES_H
#define THREAD_SAFE_STANDARD_ASCEND_RESOURCES_H

#include <mutex>
#include "StandardAscendResources.h"

namespace ascend {
class ThreadSafeStandardAscendResources : public StandardAscendResources {
public:
    explicit ThreadSafeStandardAscendResources(std::shared_ptr<DistComputeOpsManager> opsMng,
        std::string modelPath = "modelpath");

    ~ThreadSafeStandardAscendResources() override {}

    ThreadSafeStandardAscendResources(const ThreadSafeStandardAscendResources &) = delete;
    ThreadSafeStandardAscendResources& operator=(const ThreadSafeStandardAscendResources &) = delete;

    void setDevice(int deviceId) override;

    void noTempMemory() override;

    void setTempMemory(size_t size) override;

    void setDefaultTempMemory() override;

    void setAscendMallocWarning(bool flag) override;

    size_t getResourceSize() const override;

    void initialize() override;

    void uninitialize() override;

    std::shared_ptr<AscendStreamIntf> getDefaultStream() override;

    std::vector<std::shared_ptr<AscendStreamIntf>> getAlternateStreams() override;

    AscendMemory& getMemoryManager() override;

    void addUseCount() override;

    void reduceUseCount() override;

private:
    std::shared_ptr<AscendStreamIntf> createAscendStream(aclrtStream stream) override;

private:
    mutable std::recursive_mutex mtx;
    std::unordered_map<aclrtStream, std::recursive_mutex> streamMtxMap;
};
} // namespace ascend

#endif // THREAD_SAFE_STANDARD_ASCEND_RESOURCES_H
