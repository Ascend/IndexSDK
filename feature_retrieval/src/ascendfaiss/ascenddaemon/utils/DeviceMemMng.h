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


#ifndef DEVICE_MEM_MNG_H
#define DEVICE_MEM_MNG_H

#include <memory>
#include <algorithm>
#include <list>
#include "DeviceVector.h"
#include "common/utils/LogUtils.h"
#include "ErrorCode.h"

namespace ascend {

enum class DevMemStrategy {
    PURE_DEVICE_MEM = 0,
    HETERO_MEM
};

class DeviceMemMng {
public:
    // 默认构造使用纯设备侧内存策略
    DeviceMemMng();

    virtual ~DeviceMemMng();

    // 根据不同内存策略，构造不同deviceVector
    template<typename T, typename P = ExpandPolicy>
    std::unique_ptr<DeviceVector<T, P>> CreateDeviceVector(MemorySpace space = MemorySpace::DEVICE) const
    {
        if (strategy == DevMemStrategy::PURE_DEVICE_MEM) {
            return std::unique_ptr<DeviceVector<T, P>>(new DeviceVector<T, P>(space));
        }

        if (hmm == nullptr) {
            APP_LOG_ERROR("hmm init failed");
            return nullptr;
        }

        return std::unique_ptr<DeviceVector<T, P>>(new DeviceVector<T, P>(hmm));
    }

    // 异构内存策略下，需要将填充满的内存块由device侧推送至目标存储位置（device/host）
    template<typename T>
    void DevVecFullProc(std::vector<std::unique_ptr<DeviceVector<T>>> &blockBase, size_t idx,
        bool isBlockFull, aclrtStream &stream)
    {
        if (!isBlockFull) {
            lastNotFullBlockIdx = idx;
            return;
        }

        if (strategy == DevMemStrategy::PURE_DEVICE_MEM) {
            return;
        }

        if (fullBlockIdx.size() < halfBlockNumOfBuff) {
            fullBlockIdx.emplace_back(idx);
            return;
        }

        ACL_REQUIRE_OK(synchronizeStream(stream));
        std::for_each(fullBlockIdx.begin(), fullBlockIdx.end(), [&blockBase] (size_t idx) {
            blockBase[idx]->pushData();
        });
        fullBlockIdx.clear();
        fullBlockIdx.emplace_back(idx);
    }

    // DevVecFullProc接口的push行为涉及到性能优化，不是实时push的，这里是将剩余的部分push完
    template<typename T>
    void AddFinshProc(std::vector<std::unique_ptr<DeviceVector<T>>> &blockBase)
    {
        if (strategy == DevMemStrategy::PURE_DEVICE_MEM) {
            return;
        }

        std::for_each(fullBlockIdx.begin(), fullBlockIdx.end(), [&blockBase] (size_t idx) {
            blockBase[idx]->pushData();
        });
        fullBlockIdx.clear();

        if (lastNotFullBlockIdx == (blockBase.size() - 1)) {
            blockBase[lastNotFullBlockIdx]->pushData();
        }
        lastNotFullBlockIdx = 0;
    }

    // 每次search完都会需要将当前block的缓冲空间释放出来，留给其他block使用
    template<typename T>
    void PushDataAfterSearch(std::vector<std::unique_ptr<DeviceVector<T>>> &blockBase,
        size_t blockOffsetIdx, size_t curPageBlockIdx, size_t curPageBlockNum, aclrtStream &stream)
    {
        if (strategy == DevMemStrategy::PURE_DEVICE_MEM) {
            return;
        }

        size_t syncByGroup = utils::roundDown(curPageBlockNum, halfBlockNumOfBuff);
        if ((curPageBlockIdx < syncByGroup) && (curPageBlockIdx % halfBlockNumOfBuff == (halfBlockNumOfBuff - 1))) {
            ACL_REQUIRE_OK(synchronizeStream(stream));
            for (size_t i = halfBlockNumOfBuff; i > 0; i--) {
                blockBase[blockOffsetIdx + curPageBlockIdx + 1 - i]->pushData(false);
            }
        }

        if (curPageBlockIdx == curPageBlockNum - 1) {
            ACL_REQUIRE_OK(synchronizeStream(stream));
            for (size_t i = syncByGroup; i < curPageBlockNum; i++) {
                blockBase[blockOffsetIdx + i]->pushData(false);
            }
        }
    }

    template<typename T>
    void Prefetch(std::vector<std::unique_ptr<DeviceVector<T>>> &blockBase)
    {
        if (strategy == DevMemStrategy::PURE_DEVICE_MEM) {
            return;
        }

        if (hmm == nullptr) {
            return;
        }

        for (auto &dv : blockBase) {
            if ((dv == nullptr) || (dv->getHmo() == nullptr)) {
                return;
            }
            auto ret = dv->getHmo()->ValidateBufferAsync();
            if (ret != APP_ERR_OK) {
                APP_LOG_ERROR("hmm ValidateBufferAsync error, ret[%d]!", ret);
                return;
            }
        }
    }

    template<typename T>
    APP_ERROR MemoryCopy(DeviceVector<T> &dst, size_t dstOffset, DeviceVector<T> &src, size_t srcOffset, size_t copyCnt)
    {
        if (strategy == DevMemStrategy::PURE_DEVICE_MEM) {
            return aclrtMemcpy(dst.data() + dstOffset, (dst.size() - dstOffset) * sizeof(T),
                               src.data() + srcOffset, copyCnt * sizeof(T), ACL_MEMCPY_DEVICE_TO_DEVICE);
        }

        if (hmm == nullptr) {
            APP_LOG_ERROR("hmm is nullptr!");
            return APP_ERR_INNER_ERROR;
        }

        if (src.getHmo() == nullptr) {
            APP_LOG_ERROR("hmo is nullptr!");
            return APP_ERR_INNER_ERROR;
        }

        return src.getHmo()->CopyTo(dst.getHmo(), dstOffset * sizeof(T), srcOffset * sizeof(T), copyCnt * sizeof(T));
    }

    // 设置异构内存参数后，后续将使用异构内存方案
    APP_ERROR SetHeteroParam(uint32_t deviceId, size_t deviceCapacity, size_t deviceBuffer, size_t hostCapacity,
        size_t devVecSize);

    bool UsingGroupSearch() const;

    void SetHeteroStrategy();

    DevMemStrategy GetStrategy() const;
    size_t GetDeviceCapacity() const;
    size_t GetDeviceBuffer() const;

private:
    APP_ERROR InitHmm(uint32_t deviceId, size_t hostCapacity);

private:
    DevMemStrategy strategy { DevMemStrategy::PURE_DEVICE_MEM };
    size_t deviceCapacity { 0 };
    size_t deviceBuffer { 0 };
    size_t halfBlockNumOfBuff { 0 };
    size_t lastNotFullBlockIdx { 0 };

    std::vector<size_t> fullBlockIdx;
    std::shared_ptr<HmmIntf> hmm { nullptr };
};

}  // namespace ascend

#endif  // DEVICE_MEM_MNG_H
