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


#ifndef IVFSP_INDEX_H
#define IVFSP_INDEX_H

#include <vector>
#include <memory>
#include <cstdint>

#include "npu/common/AscendResourcesProxy.h"

#include "npu/common/AscendFp16.h"
#include "npu/common/ErrorCode.h"
#include "acl/acl.h"

namespace ascendSearchacc {
const int CUBE_ALIGN = 16;
const int CUBE_ALIGN_INT8 = 32;

const int FLAG_ALIGN = 32;
const int FLAG_ALIGN_OFFSET = 16;  // core 0 use first 16 flag, and core 1 use the second 16 flag.
const int FLAG_NUM = 16;           // each core have a flag, the max core is 16
const int FLAG_SIZE = 16;

const int SIZE_ALIGN = 8;
const double TIMEOUT_MS = 50000;
const int TIMEOUT_CHECK_TICK = 5120;

const int CORE_NUM = 8;
const int MAX_CORE_NUM = 16;
const int BIT_OF_UINT8 = 8;
// 0x8000000 mean 128M (resource mem pool's size)
const int64_t DEFAULT_RESOURCE_MEM = 0x8000000;

enum class MetricType {
    METRIC_INNER_PRODUCT = 0,
    METRIC_L2 = 1,
    METRIC_COSINE = 2,
};

const size_t KB = 1024;                                                  // / 1024 Bytes
const size_t RETAIN_SIZE = 2048;                                         // / 2048 Bytes
const size_t UNIT_PAGE_MB = 64;                                          // / 64MB
const size_t ADD_PAGE_BYTE_SIZE = UNIT_PAGE_MB * KB * KB - RETAIN_SIZE;  // / ~64M
const int MAX_DIM = 2048;

struct NpuIndexConfig {
    inline NpuIndexConfig(std::vector<int> devices, int64_t resources = DEFAULT_RESOURCE_MEM) : resourceSize(resources)
    {
        deviceList.assign(devices.begin(), devices.end());
    }
    std::vector<int> deviceList;
    int64_t resourceSize;
};

class NpuIndex {
public:
    NpuIndex(int dim, MetricType metricType, NpuIndexConfig config);
    explicit NpuIndex(NpuIndexConfig config);
    virtual ~NpuIndex();

    virtual APP_ERROR Train(const std::vector<float> &data) = 0;
    virtual APP_ERROR Search(std::vector<float> &queryData, int topK, std::vector<float> &dists,
                             std::vector<int64_t> &labels) const = 0;
    virtual APP_ERROR Search(size_t n, float *queryData, int topK, float *dists, int64_t *labels) const = 0;
    virtual APP_ERROR DeleteVectors(const std::vector<int64_t> &ids) = 0;
    virtual APP_ERROR Reset() = 0;
    virtual APP_ERROR Init() = 0;

    size_t GetResourceSize()
    {
        return this->resources->getResourceSize();
    }

protected:
    APP_ERROR InitResources();

protected:
    std::unique_ptr<AscendResourcesProxy> resources = nullptr;
    NpuIndexConfig ascendConfig;
    bool trained = false;
    int dim = 0;
    MetricType metricType = MetricType::METRIC_L2;
    uint64_t ntotal = 0;
};
}  // namespace ascendSearchacc

#endif  // IVFSP_INDEX_H
