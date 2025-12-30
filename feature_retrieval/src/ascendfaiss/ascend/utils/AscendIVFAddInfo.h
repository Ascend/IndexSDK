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
#include <cmath>
#include <omp.h>
#include <numeric>

#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>
#include "ascenddaemon/utils/AscendUtils.h"
#include "common/threadpool/AscendThreadPool.h"
#include "ascend/utils/fp16.h"


#ifndef ASCEND_IVF_ADD_INFO
#define ASCEND_IVF_ADD_INFO

namespace faiss {
namespace ascend {
struct AscendIVFAddInfo {
    AscendIVFAddInfo(size_t idx, size_t deviceNum, size_t size)
        : addDeviceIdx(idx), deviceAddNum(deviceNum, 0), codeSize(size)
    {
        FAISS_THROW_IF_NOT(idx < deviceNum);
    }

    inline void Add(uint8_t *code, const idx_t *id, float *precompute)
    {
        deviceAddNum[addDeviceIdx] += 1;
        addDeviceIdx = (addDeviceIdx + 1) % deviceAddNum.size();
        codes.insert(codes.end(), code, code + codeSize);
        ids.push_back((ascend_idx_t)(*id));
        precomputes.emplace_back((float)(*precompute));
    }

    inline void Add(float *code, const idx_t *id)
    {
        FAISS_THROW_IF_NOT_MSG(code != nullptr, "code is null");
        FAISS_THROW_IF_NOT_MSG(id != nullptr, "id is null");
        deviceAddNum[addDeviceIdx] += 1;
        addDeviceIdx = (addDeviceIdx + 1) % deviceAddNum.size();
        codesFp32.insert(codesFp32.end(), code, code + codeSize);
        ids.push_back(static_cast<ascend_idx_t>(*id));
    }

    inline int GetOffSet(int idx) const
    {
        int ret = 0;
        for (int i = 0; i < idx; i++) {
            ret += deviceAddNum[i];
        }

        return ret;
    }

    inline void GetCodeAndIdPtr(int idx, uint8_t **codePtr, ascend_idx_t **idPtr, float **precompute)
    {
        FAISS_THROW_IF_NOT_MSG(idx < (int)deviceAddNum.size(), "idx error");
        int off = GetOffSet(idx);
        *codePtr = codes.data() + static_cast<size_t>(off) * codeSize;
        *idPtr = ids.data() + off;
        *precompute = precomputes.data() + off;
        return;
    }

    inline void GetCodeAndIdPtr(int idx, float **codePtr, ascend_idx_t **idPtr)
    {
        FAISS_THROW_IF_NOT_FMT(idx < static_cast<int>(deviceAddNum.size()),
            "device idx (%d) >= total (%zu)", idx, deviceAddNum.size());
        int off = GetOffSet(idx);
        *codePtr = codesFp32.data() + static_cast<size_t>(off) * codeSize;
        *idPtr = ids.data() + off;
        return;
    }

    inline int GetAddNum(int idx) const
    {
        FAISS_THROW_IF_NOT_MSG(idx < (int)deviceAddNum.size(), "idx error");
        return deviceAddNum[idx];
    }

    size_t addDeviceIdx;
    std::vector<int> deviceAddNum;
    size_t codeSize;
    std::vector<uint8_t> codes;
    std::vector<float> codesFp32;
    std::vector<ascend_idx_t> ids;
    std::vector<float> precomputes;
};
} // namespace ascend
} // namespace faiss

#endif // ASCEND_IVF_ADD_INFO