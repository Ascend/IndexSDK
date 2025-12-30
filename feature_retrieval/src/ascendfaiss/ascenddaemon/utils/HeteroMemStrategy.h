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


#ifndef HETERO_MEM_STRATEGY_H
#define HETERO_MEM_STRATEGY_H

#include <list>
#include "DevVecMemStrategyIntf.h"
#include "MemorySpace.h"
#include "hmm/HmmIntf.h"
#include "common/utils/LogUtils.h"
#include "common/utils/AscendAssert.h"

namespace ascend {

// P类型预留后期拓展策略
template<typename T, typename P>
class HeteroMemStrategy : public DevVecMemStrategyIntf<T> {
public:
    // 由用户保证入参指针不为空
    explicit HeteroMemStrategy(std::shared_ptr<HmmIntf> hmm) : hmm(hmm)
    {
        if (hmm == nullptr) {
            APP_LOG_ERROR("hmm is nullptr!");
        }
    }

    virtual ~HeteroMemStrategy()
    {
        Clear();
    }

    void Clear() override
    {
        ReleaseHmo();
        size = 0;
        capacity = 0;
    }

    HeteroMemStrategy(const HeteroMemStrategy&) = delete;
    HeteroMemStrategy& operator=(const HeteroMemStrategy&) = delete;

    size_t Size() const override
    {
        return size;
    }

    size_t Capacity() const override
    {
        return capacity;
    }

    T* Data() const override
    {
        if (hmo == nullptr || hmo->Empty()) {
            return nullptr;
        }

        auto ret = hmo->ValidateBuffer();
        if (ret != APP_ERR_OK) {
            APP_LOG_ERROR("hmm ValidateBuffer error, ret[%d]", ret);
            return nullptr;
        }
        return reinterpret_cast<T *>(hmo->GetAddress());
    }

    T& operator[](size_t pos) override
    {
        auto buff = Data();
        ASCEND_THROW_IF_NOT_MSG(buff, "Data is null\n");
        return buff[pos];
    }

    const T& operator[](size_t pos) const override
    {
        auto buff = Data();
        ASCEND_THROW_IF_NOT_MSG(buff, "Data is null\n");
        return buff[pos];
    }

    std::vector<T> CopyToStlVector() const override
    {
        std::vector<T> result;
        if ((size == 0) || (hmo == nullptr)) {
            return result;
        }

        size_t memSize = size * sizeof(T);
        if (memSize > MEMCPY_S_THRESHOLD) {
            APP_LOG_ERROR("memSize[%zu] is over threshold", memSize);
            return result;
        }

        result.resize(size);
        size_t resultMemSize = result.size() * sizeof(T);
        errno_t ret = memcpy_s(result.data(), resultMemSize, Data(), memSize);
        if (ret != EOK) {
            APP_LOG_ERROR("memcpy_s error[%d], dstSize[%zu], srcSize[%zu]", ret, resultMemSize, memSize);
        }

        return result;
    }

    void Append(const T* appendData, size_t appendSize, bool reserveExact = false) override
    {
        // 暂不支持reserveExact的开关
        (void)reserveExact;
        APP_LOG_WARNING("Func Append don't suppot reserveExact param now");

        if ((appendData == nullptr) || (appendSize == 0) || (appendSize > (SIZE_MAX - size))) {
            APP_LOG_ERROR("input check error, curSize[%zu], appendSize[%zu]", size, appendSize);
            return;
        }

        size_t memSize = size * sizeof(T);
        if (memSize > MEMCPY_S_THRESHOLD) {
            APP_LOG_ERROR("memSize[%zu] is over threshold", memSize);
            return;
        }

        size_t newSize = size + appendSize;
        Reserve(newSize);
        auto ret = aclrtMemcpy((Data() + size), appendSize, appendData, appendSize, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            APP_LOG_ERROR("aclrtMemcpy failed[%d], size[%zu], appendSize[%zu]", ret, size, appendSize);
            return;
        }

        size = newSize;
    }

    void Resize(size_t newSize, bool reserveExact = false) override
    {
        // 暂不支持reserveExact的开关
        (void)reserveExact;
        APP_LOG_WARNING("Func Resize don't suppot reserveExact param now");

        if (size < newSize) {
            Reserve(newSize);
        }
        size = newSize;
    }

    size_t Reclaim(bool) override
    {
        // 暂不支持Reclaim
        APP_LOG_ERROR("Func Reclaim don't suppot now");

        return 0;
    }

    void Reserve(size_t newCapacity) override
    {
        if (capacity < newCapacity) {
            Realloc(newCapacity);
        }
    }

    void PushData(bool dataChanged) override
    {
        if (hmo == nullptr || hmo->Empty()) {
            return;
        }

        // 数据没有改变，则不需要进行push行为
        if (!dataChanged) {
            auto ret = hmo->InvalidateBuffer();
            if (ret != APP_ERR_OK) {
                APP_LOG_ERROR("hmm ValidateBuffer error, ret[%d]", ret);
            }
            return;
        }

        auto ret = hmo->FlushData();
        if (ret != APP_ERR_OK) {
            APP_LOG_ERROR("hmm FlushData error, ret[%d], size[%zu]", ret, size);
        }
        ret = hmo->InvalidateBuffer();
        if (ret != APP_ERR_OK) {
            APP_LOG_ERROR("hmm ValidateBuffer error, ret[%d]", ret);
        }
    }

    std::shared_ptr<AscendHMO> GetHmo() override
    {
        return hmo;
    }

private:
    void ReleaseHmo()
    {
        if (hmo != nullptr) {
            hmo->Clear();
        }
    }

    void Realloc(size_t allocSize)
    {
        if (hmm == nullptr) {
            APP_LOG_ERROR("hmm is nullptr");
            return;
        }
        size_t allocMemSize = allocSize * sizeof(T);
        auto createRet = hmm->CreateHmo(allocMemSize);
        if (createRet.first != APP_ERR_OK) {
            APP_LOG_ERROR("hmm CreateHmo failed[%d], allocMemSize[%zu]", createRet.first, allocMemSize);
            return;
        }

        if (hmo != nullptr && !hmo->Empty()) {
            auto ret = hmo->CopyTo(createRet.second, 0, 0, size);
            if (ret != 0) {
                APP_LOG_ERROR("hmm copy failed, ret[%d], size[%zu]", ret, size);
                createRet.second->Clear();
                return;
            }
        }

        ReleaseHmo();

        hmo = createRet.second;
        capacity = allocSize;
    }

private:
    size_t size { 0 };
    size_t capacity { 0 };

    std::shared_ptr<HmmIntf> hmm { nullptr };
    std::shared_ptr<AscendHMO> hmo;
};

}  // namespace ascend

#endif  // HETERO_MEM_STRATEGY_H