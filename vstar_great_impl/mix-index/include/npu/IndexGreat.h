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


#ifndef IVFSP_INDEXVSTARMIX_H
#define IVFSP_INDEXVSTARMIX_H

#include <vector>
#include <map>
#include <utility>
#include <memory>

#include "npu/NpuIndexVStar.h"
#include "npu/common/DeviceVector.h"
#include "npu/common/AscendTensor.h"
#include "faiss/impl/ScalarQuantizer.h"
#include "npu/common/threadpool/AscendThreadPool.h"
#include "impl/IndexHNSWRobustGraph.h"
#include "npu/common/utils/AscendAssert.h"

#ifdef USE_ACL_NN_INTERFACE
#include <aclnn/acl_meta.h>
#endif

namespace ascendSearchacc {

struct KModeInitParams {
    KModeInitParams()
    {
    }
    KModeInitParams(int dim, int R, int convPQM, int evaluationType, int efConstruction)
        : dim(dim), R(R), convPQM(convPQM), evaluationType(evaluationType), efConstruction(efConstruction)
    {
    }
    int dim = 256;
    int R = 50;
    int convPQM = 128;
    int evaluationType = 0;
    int efConstruction = 300;
};

/**
 * @note: Initilization parameter struct for creating desginated index
 */
struct IndexGreatInitParams {
    IndexGreatInitParams()
    {
    }
    // AKMode
    IndexGreatInitParams(const std::string mode, const IndexVstarInitParams &AInitParams,
                         const KModeInitParams &KInitParams, bool verbose = false)
    {
        ASCEND_THROW_IF_NOT_MSG(AInitParams.dim == KInitParams.dim,
                                "IndexGreatInitParams: dim mismatched between AMode and KMode.\n");
        this->mode = mode;
        this->AInitParams = AInitParams;
        this->KInitParams = KInitParams;
        this->verbose = verbose;
    }
    // KMode
    IndexGreatInitParams(const std::string mode, const KModeInitParams &KInitParams, bool verbose = false)
    {
        this->mode = mode;
        this->KInitParams = KInitParams;
        this->verbose = verbose;
    }
    std::string mode = "AKMode";
    IndexVstarInitParams AInitParams;
    KModeInitParams KInitParams;
    bool verbose = false;
};

/**
 * @note: Adjustable searching parameter struct
 */
struct IndexGreatSearchParams {
    // AKMode
    IndexGreatSearchParams()
    {
    }
    IndexGreatSearchParams(std::string mode, int nProbeL1, int nProbeL2, int l3SegmentNum, int ef)
        : mode(mode), nProbeL1(nProbeL1), nProbeL2(nProbeL2), l3SegmentNum(l3SegmentNum), ef(ef)
    {
    }

    // KMode
    IndexGreatSearchParams(std::string mode, int ef) : mode(mode), ef(ef)
    {
    }

    std::string mode = "AKMode";
    // AMode search params
    int nProbeL1 = 72;
    int nProbeL2 = 64;
    int l3SegmentNum = 512;

    // KMode search params
    int ef = 150;
};

class IndexGreat {
public:
    explicit IndexGreat(const std::string mode, const IndexGreatInitParams &params);

    explicit IndexGreat(const std::string mode, const std::vector<int> &deviceList = {0}, const bool verbose = false);

    void LoadIndex(const std::string &indexPath);

    void LoadIndex(const std::string &AModeindexPath, const std::string &KModeindexPath);

    void WriteIndex(const std::string &indexPath);

    void WriteIndex(const std::string &AModeindexPath, const std::string &KModeindexPath);

    APP_ERROR AddCodeBooks(const std::string &codeBooksPath);

    APP_ERROR AddVectors(const std::vector<float> &baseRawData);

    APP_ERROR AddVectorsWithIds(const std::vector<float> &baseRawData, const std::vector<int64_t>& ids);

    APP_ERROR Search(const SearchImplParams &params);

    APP_ERROR SearchWithMask(const SearchImplParams &params, const std::vector<uint8_t> &mask);

    void SetSearchParams(const IndexGreatSearchParams &params);

    IndexGreatSearchParams GetSearchParams() const;

    int GetDim() const;

    uint64_t GetNTotal() const;

    void Reset();

    ~IndexGreat(){};

private:
    void SearchBatchImpl(size_t n, const float *queryData, int topK, float *dists, int64_t *labels);

    void SearchBatchImplMask(size_t n, const float *queryData, const uint8_t* mask, int topK,
                            float *dists, int64_t *labels);

private:
    bool AKMode = false;
    bool KMode = false;

    int dim = 1024;
    uint64_t nTotal = 0;
    int topKGreat = 100;

    // AMode params
    int subSpaceDimL1 = 0;
    int subSpaceDimL2 = 0;
    int nlistL1 = 0;
    int nlistL2 = 0;
    std::vector<int> deviceList = {0};

    // KMode params
    int R = 0;
    int convPQM = 0;
    int evaluationType = 0;
    int efConstruction = 300;

    // available indices
    std::unique_ptr<NpuIndexVStar> indexVStar = nullptr;
    std::unique_ptr<ascendsearch::IndexHNSWGraphPQHybrid> indexHNSW = nullptr;

    std::map<int64_t, int64_t> idMap;
    std::map<int64_t, int64_t> idMapReverse;
    float mixSearchRatio = 0.5;
    bool verbose = false;
    // flags checking the order of operation
    bool paramsChecked = false;  // any operation without paramsChecked is invalid
    bool codebookAdded = false;  // only allow addVectors when codebook added, otherwise throw error
    bool indexCreated = false;   // cannot search with this boolean being true
    bool setSearchParamsSucceed = true;
};
}  // namespace ascendSearchacc

#endif  // IVFSP_INDEXVSTARMIX_H
