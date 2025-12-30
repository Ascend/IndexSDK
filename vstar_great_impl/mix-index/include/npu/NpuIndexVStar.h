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


#ifndef IVFSP_INDEXVSTAR_H
#define IVFSP_INDEXVSTAR_H

#include <vector>
#include <map>

#include "npu/NpuIndexIVFHSP.h"
#include "npu/common/DeviceVector.h"
#include "npu/common/AscendTensor.h"
#include "faiss/impl/ScalarQuantizer.h"
#include "npu/common/threadpool/AscendThreadPool.h"
#include <memory>
#ifdef USE_ACL_NN_INTERFACE
#include <aclnn/acl_meta.h>
#endif

namespace ascendSearchacc {

/**
 * @brief Vstar初始化结构体
 */
struct IndexVstarInitParams {
    IndexVstarInitParams()
    {
    }
    IndexVstarInitParams(int dim, int subSpaceDimL1, int nlistL1, std::vector<int> deviceList, bool verbose = false,
                         int64_t ascendResources = 0x8000000)
        : dim(dim),
          subSpaceDimL1(subSpaceDimL1),
          nlistL1(nlistL1),
          deviceList(deviceList),
          verbose(verbose),
          ascendResources(ascendResources)
    {
    }

    int dim = 1024;
    int subSpaceDimL1 = 128;
    int nlistL1 = 1024;
    std::vector<int> deviceList = {0};
    bool verbose = false;
    int64_t ascendResources = 0x8000000;
};

/**
 * @brief 检索超参结构体 (用于设置检索时算法参数)
 */
struct IndexVstarSearchParams {
    IndexVstarSearchParams(int nProbeL1, int nProbeL2, int l3SegmentNum)
    {
        params.nProbeL1 = nProbeL1;
        params.nProbeL2 = nProbeL2;
        params.l3SegmentNum = l3SegmentNum;
    }
    SearchParams params;
};

/**
 * @brief 检索参数结构体 (用于设置检索时用户输入参数)
 */
struct SearchImplParams {
    SearchImplParams(size_t n, std::vector<float> &queryData, int topK, std::vector<float> &dists,
                     std::vector<int64_t> &labels)
        : n(n), queryData(queryData), topK(topK), dists(dists), labels(labels)
    {
    }

    size_t n = 0;
    std::vector<float> &queryData;
    int topK = 100;
    std::vector<float> &dists;
    std::vector<int64_t> &labels;
};

class NpuIndexVStar {
public:
    std::unique_ptr<NpuIndexIVFHSP> innerHSP = nullptr;

public:
    explicit NpuIndexVStar(const IndexVstarInitParams &params);

    explicit NpuIndexVStar(const std::vector<int> &deviceList = {0}, const bool verbose = false,
                           const int64_t ascendResources = 0x8000000);

    virtual ~NpuIndexVStar(){};

    void LoadIndex(std::string indexPath, const NpuIndexVStar *loadedIndex = nullptr);

    void WriteIndex(const std::string &indexPath);

    APP_ERROR AddCodeBooks(const NpuIndexVStar *loadedIndex);

    // allow users to load both codebooks via a single file path, assuming both codebooks are combined into one binary
    APP_ERROR AddCodeBooks(const std::string &codeBooksPath);

    APP_ERROR AddVectors(const std::vector<float> &baseRawData);
    APP_ERROR AddVectorsWithIds(const std::vector<float> &baseRawData, const std::vector<int64_t> &ids);
    APP_ERROR AddVectorsTmp(const std::vector<float> &baseRawData, const bool tmp);

    APP_ERROR DeleteVectors(const std::vector<int64_t> &ids);
    APP_ERROR DeleteVectors(const int64_t &id);
    APP_ERROR DeleteVectors(const int64_t &startId, const int64_t &endId);

    APP_ERROR Search(const SearchImplParams &params);
    APP_ERROR Search(const SearchImplParams &params, const std::vector<uint8_t> &mask);
    APP_ERROR MultiSearch(const std::vector<NpuIndexVStar *> &indexes, const SearchImplParams &params, bool merge);
    APP_ERROR MultiSearch(const std::vector<NpuIndexVStar *> &indexes, const SearchImplParams &params,
                          const std::vector<uint8_t> &mask, bool merge);

    void SetSearchParams(const IndexVstarSearchParams &params);
    IndexVstarSearchParams GetSearchParams() const;
    void SetTopK(int topK);

    int GetDim() const;
    uint64_t GetNTotal() const;
    APP_ERROR Reset();

private:
    bool IsIndexValid() const;

    int dim = 0;
    int subSpaceDimL1 = 0;
    int subSpaceDimL2 = 0;
    int nlistL1 = 0;
    int nlistL2 = 0;
    int topKVstar = 100;
    std::vector<int> deviceList = {0};
    std::map<int64_t, int64_t> idMap;
    std::map<int64_t, int64_t> idMapReverse;  // key is the virtual base vector id, value is the real base vector id
    bool verbose = false;
    // flags checking the order of operation
    bool paramsChecked = false;  // any operation without paramsChecked is invalid
    bool codebookAdded = false;  // only allow addVectors when codebook added, otherwise throw error
    bool indexCreated = false;   // cannot search with this boolean being true
    bool setSearchParamsSucceed = true;
};
}  // namespace ascendSearchacc

#endif  // IVFSP_INDEXVSTAR_H
