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

#include "npu/IndexGreat.h"

#include <algorithm>
#include <cmath>
#include <atomic>
#include <numeric>
#include <unistd.h>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <iostream>
#include <thread>

#include "omp.h"

#include "npu/common/AscendFp16.h"
#include "utils/fp16.h"
#include "npu/common/utils/AscendAssert.h"
#include "npu/common/threadpool/AscendThreadPool.h"
#include "npu/common/utils/OpLauncher.h"
#include "npu/common/KernelSharedDef.h"
#include "npu/common/utils/StaticUtils.h"
#include "impl/DistanceSimd.h"
#include "npu/common/utils/LogUtils.h"
#include "utils/VstarIoUtil.h"

#include <chrono>
#include <fstream>

#include "npu/common/utils/AscendAssert.h"

namespace ascendSearchacc {
    
namespace {
    constexpr size_t KModeMaxSearch = 260;
    const std::vector<int> validDim = {128, 256, 512, 1024};
    constexpr uint64_t NTOTAL_MAX = 1e8;
    constexpr uint64_t NTOTAL_MIN = 1e4;
}

IndexGreat::IndexGreat(const std::string mode, const IndexGreatInitParams &params)
{
    // if mismatched
    ASCEND_THROW_IF_NOT_MSG(mode == params.mode, "Search Mode mismatched. Choose between 'KMode' and 'AKMode'.\n");
    ASCEND_THROW_IF_NOT_FMT(std::find(validDim.begin(), validDim.end(), params.KInitParams.dim) != validDim.end(),
        "Invalid dim: Should be in [128, 256, 512, 1024]; your dim is %d.\n", params.KInitParams.dim);

    // R is limited to be between 50 and 100 inclusively
    ASCEND_THROW_IF_NOT_FMT(ascendsearch::R_MIN <= params.KInitParams.R &&
                            params.KInitParams.R <= ascendsearch::R_MAX,
        "Invalid degree: Should be between 50 and 100 inclusively; your R is %d.\n", params.KInitParams.R);

    ASCEND_THROW_IF_NOT_FMT(16 <= params.KInitParams.convPQM && // greater than 16
                            params.KInitParams.convPQM % 8 == 0 && // divisible by 8
                            dim % params.KInitParams.convPQM == 0, // divides dim
        "Invalid convPQM: Should be 1) greater than or equal to 16; 2) divisible by 8;"
        "3) divides dim; your conPQM is %d.\n", params.KInitParams.convPQM);

    // evaluationType should be either 0 or 1
    ASCEND_THROW_IF_NOT_FMT(params.KInitParams.evaluationType == faiss::METRIC_L2 ||
                            params.KInitParams.evaluationType == faiss::METRIC_INNER_PRODUCT,
        "Invalid evaluationType: Should be 0 or 1; your evaluationType is %d.\n", params.KInitParams.evaluationType);

    // between 200 and 400 inclusively
    ASCEND_THROW_IF_NOT_FMT(ascendsearch::EF_CONSTRUCTION_MIN <= params.KInitParams.efConstruction &&
                            params.KInitParams.efConstruction <= ascendsearch::EF_CONSTRUCTION_MAX &&
                            params.KInitParams.efConstruction % 10 == 0, // divisible by 10
        "Invalid efConstruction: Should be 1) between 200 and 400 inclusively;"
        "2) divisible by 10; your efConstruction is %d.\n", params.KInitParams.efConstruction);

    this->verbose = params.verbose;
    if (mode == "AKMode") {
        this->AKMode = true;
        if (verbose) {
            printf("Initializing AKMode index with full parameters...\n");
        }
        this->dim = params.AInitParams.dim;
        this->subSpaceDimL1 = params.AInitParams.subSpaceDimL1;
        this->nlistL1 = params.AInitParams.nlistL1;
        this->deviceList = params.AInitParams.deviceList;
        indexVStar = std::make_unique<NpuIndexVStar>(params.AInitParams);
        ASCEND_THROW_IF_NOT_MSG(indexVStar != nullptr, "indexVStar is a nullptr.\n");
        ASCEND_THROW_IF_NOT_MSG(indexVStar->innerHSP != nullptr, "index's innerHSP structure is a nullptr.\n");
        this->subSpaceDimL2 = indexVStar->innerHSP->GetSubDimL2();
        this->nlistL2 = indexVStar->innerHSP->GetNlistL2();
        if (verbose) {
            printf("Finished creating AMode index...\n");
        }
        this->R = params.KInitParams.R;
        this->convPQM = params.KInitParams.convPQM;
        this->evaluationType = params.KInitParams.evaluationType;
        this->efConstruction = params.KInitParams.efConstruction;
        indexHNSW = std::make_unique<ascendsearch::IndexHNSWGraphPQHybrid>(dim, R, convPQM,
            ascendsearch::DEFAULT_INDEX_PQ_BITS, (faiss::MetricType)evaluationType);
        ASCEND_THROW_IF_NOT_MSG(indexHNSW != nullptr, "indexHNSW is a nullptr.\n");
        indexHNSW->verbose = verbose;
        indexHNSW->innerGraph.efConstruction = efConstruction;
        indexHNSW->innerGraph.thresholdEdge = 1.0;
        indexHNSW->innerGraph.isOpenCluster = false;
        indexHNSW->innerGraph.efSearch = 150; // recommended value 150
        if (verbose) {
            printf("Finished creating KMode index.\n");
        }
    } else if (mode == "KMode") {
        this->KMode = true;
        if (verbose) {
            printf("Initializing KMode index with full parameters...\n");
        }
        this->dim = params.KInitParams.dim;
        this->R = params.KInitParams.R;
        this->convPQM = params.KInitParams.convPQM;
        this->evaluationType = params.KInitParams.evaluationType;
        this->efConstruction = params.KInitParams.efConstruction;
        indexHNSW = std::make_unique<ascendsearch::IndexHNSWGraphPQHybrid>(dim, R, convPQM, 8, // 8 mean 8bits
                                                                           (faiss::MetricType)evaluationType);
        ASCEND_THROW_IF_NOT_MSG(indexHNSW != nullptr, "indexHNSW is a nullptr.\n");
        indexHNSW->verbose = verbose;
        indexHNSW->innerGraph.efConstruction = efConstruction;
        indexHNSW->innerGraph.thresholdEdge = 1.0;
        indexHNSW->innerGraph.isOpenCluster = false;
        indexHNSW->innerGraph.efSearch = 150; // recommended value 150
        if (verbose) {
            printf("Finished creating KMode index.\n");
        }
    } else {
        ASCEND_THROW_MSG("Search Mode mismatched. Choose between 'KMode' and 'AKMode'.\n");
    }
    paramsChecked = true;
}

IndexGreat::IndexGreat(const std::string mode, const std::vector<int> &deviceList, const bool verbose)
{
    this->verbose = verbose;

    if (mode == "AKMode") {
        this->AKMode = true;
        ASCEND_THROW_IF_NOT_MSG(deviceList.size() == 1, "Invalid deviceList size, must have a size of 1.\n");
        if (verbose) {
            printf("You are creating an IndexGreat instance in AKMode without specifying explicit parameters."
                " Currently you are using device %d.\n",
                deviceList[0]);
            }
        indexVStar = std::make_unique<NpuIndexVStar>(deviceList);
        ASCEND_THROW_IF_NOT_MSG(indexVStar != nullptr, "indexVStar is a nullptr.\n");
        if (verbose) {
            printf("Finished creating AMode index...\n");
        }
        indexHNSW = std::make_unique<ascendsearch::IndexHNSWGraphPQHybrid>();
        ASCEND_THROW_IF_NOT_MSG(indexHNSW != nullptr, "indexHNSW is a nullptr.\n");
        if (verbose) {
            printf("Finished creating KMode index.\n");
        }
    } else if (mode == "KMode") {
        this->KMode = true;
        if (verbose) {
            printf("You are creating an IndexGreat instance in KMode without specifying explicit parameters."
                " If you inputted any deviceList parameter, it will not be used.\n");
        }
        indexHNSW = std::make_unique<ascendsearch::IndexHNSWGraphPQHybrid>();
        ASCEND_THROW_IF_NOT_MSG(indexHNSW != nullptr, "indexHNSW is a nullptr.\n");
        if (verbose) {
            printf("Finished creating KMode index.\n");
        }
    } else {
        ASCEND_THROW_MSG("Search Mode mismatched. Choose between 'KMode' and 'AKMode'.\n");
    }
}

void IndexGreat::LoadIndex(const std::string &indexPath)
{
    ASCEND_THROW_IF_NOT_MSG(KMode,
        "Currently in AKMode: Please pass in your index paths for both modes"
        " using the 2-parameter LoadIndex instead.\n");
    ascendSearchacc::checkSoftLink(indexPath);
    // 将判断流程是否继续的布尔值全部置成false，如果LoadIndex失败则阻止流程继续进行
    paramsChecked = false;
    codebookAdded = false;
    indexCreated = false;
    indexHNSW = indexHNSW->LoadIndex(indexPath);
    ASCEND_THROW_IF_NOT_MSG(indexHNSW != nullptr, "indexHNSW is a nullptr.\n");
    nTotal = (uint64_t)(indexHNSW->GetNTotal());
    dim = (int)(indexHNSW->GetDim());
    idMap = indexHNSW->GetIdMap();
    for (auto &idPair: idMap) {
        int64_t realId = idPair.first;
        int64_t virtualId = idPair.second;
        idMapReverse.insert(std::make_pair(virtualId, realId)); // 在idMap的基础上，对idMapReverse进行重建
    }
    indexHNSW->innerGraph.efConstruction = 300; // recommended value 300
    indexHNSW->innerGraph.thresholdEdge = 1.0;
    indexHNSW->innerGraph.isOpenCluster = false;
    indexHNSW->innerGraph.efSearch = 150; // recommended value 150
    if (verbose) {
        printf("Finished loading KMode index.\n");
    }
    paramsChecked = true;
    codebookAdded = true;
    indexCreated = true;
}

void IndexGreat::LoadIndex(const std::string &AModeindexPath, const std::string &KModeindexPath)
{
    ASCEND_THROW_IF_NOT_MSG(
        AKMode, "Currently in KMode: Please pass in your index path for KMode using the single WriteIndex instead.\n");
    indexVStar->LoadIndex(AModeindexPath);
    if (verbose) {
        printf("Finished loading AMode index...\n");
    }
    ascendSearchacc::checkSoftLink(KModeindexPath);
    // 将判断流程是否继续的布尔值全部置成false，如果LoadIndex失败则阻止流程继续进行
    paramsChecked = false;
    codebookAdded = false;
    indexCreated = false;
    indexHNSW = indexHNSW->LoadIndex(KModeindexPath);
    ASCEND_THROW_IF_NOT_MSG(indexHNSW != nullptr, "indexHNSW is a nullptr.\n");
    nTotal = (uint64_t)(indexHNSW->GetNTotal());
    dim = (int)(indexHNSW->GetDim());
    ASCEND_THROW_IF_NOT_MSG(indexVStar != nullptr, "indexVStar is a nullptr.\n");
    ASCEND_THROW_IF_NOT_MSG(indexVStar->innerHSP != nullptr, "index's innerHSP structure is a nullptr.\n");
    nlistL1 = indexVStar->innerHSP->GetNlistL1();
    nlistL2 = indexVStar->innerHSP->GetNlistL2();
    subSpaceDimL1 = indexVStar->innerHSP->GetSubDimL1();
    subSpaceDimL2 = indexVStar->innerHSP->GetSubDimL2();
    idMap = indexHNSW->GetIdMap();
    for (auto &idPair: idMap) {
        int64_t realId = idPair.first;
        int64_t virtualId = idPair.second;
        idMapReverse.insert(std::make_pair(virtualId, realId)); // 在idMap的基础上，对idMapReverse进行重建
    }
    indexHNSW->innerGraph.efConstruction = 300; // recommended value 300
    indexHNSW->innerGraph.thresholdEdge = 1.0;
    indexHNSW->innerGraph.isOpenCluster = false;
    indexHNSW->innerGraph.efSearch = 150; // recommended value 150
    if (verbose) {
        printf("Finished loading KMode index.\n");
    }
    paramsChecked = true;
    codebookAdded = true;
    indexCreated = true;
}

void IndexGreat::WriteIndex(const std::string &indexPath)
{
    ASCEND_THROW_IF_NOT_MSG(KMode,
        "Currently in AKMode: Please pass in your index paths for both modes"
        " using the 2-parameter WriteIndex instead.\n");
    ASCEND_THROW_IF_NOT_MSG(paramsChecked,
        "No valid parameters found; If you created a Great instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(indexCreated,
                            "Base vectors not added; Please add base vectors first to create the index first.\n");
    indexHNSW->SaveIndex(indexPath);
    if (verbose) {
        printf("Finished writing KMode index.\n");
    }
}

void IndexGreat::WriteIndex(const std::string &AModeindexPath, const std::string &KModeindexPath)
{
    ASCEND_THROW_IF_NOT_MSG(AKMode,
        "Currently in KMode: Please pass in your index path for KMode using the single WriteIndex instead.\n");
    ASCEND_THROW_IF_NOT_MSG(paramsChecked,
        "No valid parameters found; If you created a Great instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data; Please add codebooks first.\n");
    ASCEND_THROW_IF_NOT_MSG(indexCreated,
                            "Base vectors not added; Please add base vectors first to create the index first.\n");
    indexVStar->WriteIndex(AModeindexPath);
    if (verbose) {
        printf("Finished writing AMode index...\n");
    }
    indexHNSW->SaveIndex(KModeindexPath);
    if (verbose) {
        printf("Finished writing KMode index.\n");
    }
}

APP_ERROR IndexGreat::AddCodeBooks(const std::string &codeBooksPath)
{
    ASCEND_THROW_IF_NOT_MSG(AKMode, "AddCodeBooks only supported in AKMode.\n");
    ASCEND_THROW_IF_NOT_MSG(
        paramsChecked,
        "No valid parameters found; if you created a Great instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    if (verbose) {
        printf("Adding codebooks to index ...\n");
    }
    // 将判断码本是否添加成功的布尔值置成false，若设置成功在结尾设置true
    codebookAdded = false;
    auto ret = indexVStar->AddCodeBooks(codeBooksPath);
    codebookAdded = true;
    return ret;
}

APP_ERROR IndexGreat::AddVectors(const std::vector<float> &baseRawData)
{
    ASCEND_THROW_IF_NOT_MSG(
        paramsChecked,
        "No valid parameters found; If you created a Great instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    if (AKMode) {
        ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data, please add codebooks first.\n");
    }

    ASCEND_THROW_IF_NOT_MSG(!baseRawData.empty(), "base vector cannot be empty.\n");
    ASCEND_THROW_IF_NOT_MSG(baseRawData.size() % static_cast<size_t>(dim) == 0, "base vector's size must be a multiple of dim.\n");

    uint64_t n = baseRawData.size() / static_cast<size_t>(dim);

    ASCEND_THROW_IF_NOT_MSG(nTotal == 0, "Progressive Add is currently not supported.\n");
    ASCEND_THROW_IF_NOT_FMT((NTOTAL_MIN <= n) && (n <= NTOTAL_MAX),
        "input base size %llu should be in [%llu, %llu].", n, NTOTAL_MIN, NTOTAL_MAX);

    int err = 0;
    nTotal += n;
    if (AKMode) {
        if (verbose) {
            printf("Adding base vector to AKMode Index...\n");
        }
        if (verbose) {
            printf("Intializing Vstar base: this may take some time...\n");
        }
        indexVStar->AddVectorsTmp(baseRawData, false);
        indexHNSW->train(n, baseRawData.data());
        indexHNSW->add(n, baseRawData.data());
    }
    if (KMode) {
        if (verbose) {
            printf("Adding base vector to KMode Index...\n");
        }
        indexHNSW->train(n, baseRawData.data());
        indexHNSW->add(n, baseRawData.data());
    }
    indexCreated = true;
    return err;
}

APP_ERROR IndexGreat::AddVectorsWithIds(const std::vector<float> &baseRawData, const std::vector<int64_t>& ids)
{
    ASCEND_THROW_IF_NOT_MSG(paramsChecked,
        "No valid parameters found; If you created a Great instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    if (AKMode) {
        ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data, please add codebooks first.\n");
    }

    ASCEND_THROW_IF_NOT_MSG(!baseRawData.empty(), "base vector cannot be empty.\n");
    ASCEND_THROW_IF_NOT_MSG(baseRawData.size() % static_cast<size_t>(dim) == 0, "base vector's size must be a multiple of dim.\n");
    
    size_t n = baseRawData.size() / static_cast<size_t>(dim);

    ASCEND_THROW_IF_NOT_MSG(nTotal == 0, "Progressive Add is currently not supported.\n");
    ASCEND_THROW_IF_NOT_FMT((NTOTAL_MIN <= n) && (n <= NTOTAL_MAX),
        "input base size %llu should be in [%llu, %llu].", n, NTOTAL_MIN, NTOTAL_MAX);

    ASCEND_THROW_IF_NOT_FMT(ids.size() == n,
                            "ids's and baseRawData's sizes do not match: id's size is %llu, baseRawData's size is %llu",
                            ids.size(), n);
    for (size_t i = 0; i < n; i++) {
        idMap[i] = ids[i];
        ASCEND_THROW_IF_NOT_MSG(idMapReverse.find(ids[i]) == idMapReverse.end(),
                                "Unique mapping between virtual and real id is required.\n");
        idMapReverse[ids[i]] = static_cast<int64_t>(i);
    }
    indexHNSW->SetIdMap(idMap);
    return AddVectors(baseRawData);
}

APP_ERROR IndexGreat::Search(const SearchImplParams &params)
{
    ASCEND_THROW_IF_NOT_MSG(
        paramsChecked,
        "No valid parameters found; If you created a Great instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    if (AKMode) {
        ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data; Please add codebooks first.\n");
    }
    ASCEND_THROW_IF_NOT_MSG(indexCreated,
                            "Base vectors not added; Please add base vectors first to create the index first.\n");
    ASCEND_THROW_IF_NOT_MSG(setSearchParamsSucceed, "Setting Search Parameter failed.\n");

    if (params.topK != topKGreat) {
        ASCEND_THROW_IF_NOT_FMT((0 < params.topK) && (params.topK <= 4096),
            "topK must be greater than 0 and less than or equal to 4096; you are setting it to %d\n", params.topK);
        if (AKMode) {
            indexVStar->SetTopK(params.topK);
        }
        topKGreat = params.topK;
    }
    {
        ASCEND_THROW_IF_NOT_MSG((0 < params.n) && (params.n <= 10000),
                                "SearchSize Must be greater than 0 and less than or equal to 10000.\n");
        ASCEND_THROW_IF_NOT_MSG(params.queryData.size() >= params.n * static_cast<size_t>(dim),
                                "Queries vector must have at least n * dim elements.\n");
        ASCEND_THROW_IF_NOT_MSG(params.dists.size() >= params.n * static_cast<size_t>(topKGreat),
                                "Dist vector must have at least n * topK vacancies.\n");
        ASCEND_THROW_IF_NOT_MSG(params.labels.size() >= params.n * static_cast<size_t>(topKGreat),
                                "Label vector must have at least n * topK vacancies.\n");
    }

    // assign all values in labels to -1 so that the result is compatiable with searchWithMask of Vstar
    std::fill(params.labels.begin(), params.labels.end(), -1);

    int err = 0;
    if (AKMode) {
        if (params.n == 1) {
            // if batchSize == 1, use KMode only
            indexHNSW->search(params.n, params.queryData.data(), params.topK, params.dists.data(),
                              params.labels.data());
        } else {
            size_t AQueryNum = static_cast<size_t>(params.n * mixSearchRatio);
            size_t KQueryNum = params.n - AQueryNum;
            std::vector<std::thread> threadPools;

            SearchImplParams paramsAMode(AQueryNum, params.queryData, params.topK, params.dists, params.labels);
            threadPools.emplace_back(
                static_cast<APP_ERROR (NpuIndexVStar::*)(const SearchImplParams&)>(&NpuIndexVStar::Search),
                indexVStar.get(), paramsAMode
            );
            if (KQueryNum <= KModeMaxSearch) {
                threadPools.emplace_back(
                    (&ascendsearch::IndexHNSWGraphPQHybrid::search),
                    indexHNSW.get(), KQueryNum, params.queryData.data() + AQueryNum * dim,
                    params.topK,
                    params.dists.data() + AQueryNum * params.topK,
                    params.labels.data() + AQueryNum * params.topK,
                    nullptr);
            } else {
                threadPools.emplace_back(
                    (&ascendSearchacc::IndexGreat::SearchBatchImpl),
                    this, KQueryNum, params.queryData.data() + AQueryNum * dim, params.topK,
                    params.dists.data() + AQueryNum * params.topK,
                    params.labels.data() + AQueryNum * params.topK);
            }
            for (auto& thread:threadPools) {
                thread.join();
            }
        }
    }
    if (KMode) {
        if (params.n <= KModeMaxSearch) {
            indexHNSW->search(params.n, params.queryData.data(), params.topK, params.dists.data(),
                              params.labels.data());
        } else {
            SearchBatchImpl(params.n, params.queryData.data(), params.topK, params.dists.data(), params.labels.data());
        }
    }
    if (idMap.size() > 0) {
        for (size_t i = 0; i < params.n * params.topK; i++) {
            params.labels[i] = idMap[params.labels[i]];  // map the virtual id back into real base id
        }
    }
    return err;
}

APP_ERROR IndexGreat::SearchWithMask(const SearchImplParams &params, const std::vector<uint8_t> &mask)
{
    ASCEND_THROW_IF_NOT_MSG(
        paramsChecked,
        "No valid parameters found; If you created a Great instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    if (AKMode) {
        ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data; Please add codebooks first.\n");
    }
    ASCEND_THROW_IF_NOT_MSG(indexCreated,
                            "Base vectors not added; Please add base vectors first to create the index first.\n");
    ASCEND_THROW_IF_NOT_MSG(setSearchParamsSucceed, "Setting Search Parameter failed.\n");
    size_t maskDim = (nTotal + 7) / 8;

    if (params.topK != topKGreat) {
        ASCEND_THROW_IF_NOT_FMT((0 < params.topK) && (params.topK <= 4096),
            "topK must be greater than 0 and less than or equal to 4096; you are setting it to %d\n", params.topK);
        if (AKMode) {
            indexVStar->SetTopK(params.topK);
        }
        topKGreat = params.topK;
    }
    {
        ASCEND_THROW_IF_NOT_MSG((0 < params.n) && (params.n <= 10000),
                                "SearchSize Must be greater than 0 and less than or equal to 10000.\n");
        ASCEND_THROW_IF_NOT_MSG(params.queryData.size() >= params.n * static_cast<size_t>(dim),
                                "Queries vector must have at least n * dim elements.\n");
        ASCEND_THROW_IF_NOT_MSG(params.dists.size() >= params.n * static_cast<size_t>(topKGreat),
                                "Dist vector must have at least n * topK vacancies.\n");
        ASCEND_THROW_IF_NOT_MSG(params.labels.size() >= params.n * static_cast<size_t>(topKGreat),
                                "Label vector must have at least n * topK vacancies.\n");
        ASCEND_THROW_IF_NOT_MSG(mask.size() >= params.n * maskDim,
                                "Mask Vector must have at least Ceil(NTotal/8) elements.\n");
    }

    // assign all values in labels to -1 so that the result is compatiable with searchWithMask of Vstar
    std::fill(params.labels.begin(), params.labels.end(), -1);
    
    int err = 0;
    const uint8_t* mask_data = mask.data();
    if (AKMode) {
        if (params.n == 1) {
            ascendsearch::HNSWGraphPQHybridSearchWithMaskParams HNSW_MaskParam(params.n, params.queryData.data(),
                                                                               params.topK,
                                                                               const_cast<uint8_t *>(mask_data),
                                                                               maskDim, 0, nTotal);
            indexHNSW->SearchWithMask(HNSW_MaskParam, params.dists.data(), params.labels.data());
        } else {  // AK-mixed
            size_t AQueryNum = static_cast<size_t>(params.n * mixSearchRatio);
            size_t KQueryNum = params.n - AQueryNum;
            std::vector<std::thread> threadPools;
            // std::vector<float>AModeQueryData(params.queryData.begin(), params.queryData.begin() + AQueryNum * dim);
            SearchImplParams paramsAMode(AQueryNum, params.queryData, params.topK, params.dists, params.labels);

            threadPools.emplace_back(
                static_cast<APP_ERROR (NpuIndexVStar::*)(const SearchImplParams&,
                    const std::vector<uint8_t>&)>(&NpuIndexVStar::Search),
                indexVStar.get(), paramsAMode, mask
            );
            if (KQueryNum <= KModeMaxSearch) {
                ascendsearch::HNSWGraphPQHybridSearchWithMaskParams HNSW_MaskParam(KQueryNum,
                    params.queryData.data() + AQueryNum * static_cast<size_t>(dim), params.topK,
                    const_cast<uint8_t *>(mask_data) + AQueryNum * maskDim, maskDim, 0, nTotal);
                threadPools.emplace_back(
                    (&ascendsearch::IndexHNSWGraphPQHybrid::SearchWithMask), indexHNSW.get(), HNSW_MaskParam,
                    params.dists.data() + AQueryNum * params.topK, params.labels.data() + AQueryNum * params.topK);
            } else {
                threadPools.emplace_back(
                    (&ascendSearchacc::IndexGreat::SearchBatchImplMask), this, KQueryNum,
                    params.queryData.data() + AQueryNum * dim, mask_data, params.topK,
                    params.dists.data() + AQueryNum * params.topK, params.labels.data() + AQueryNum * params.topK);
            }
            for (auto& thread:threadPools) {
                thread.join();
            }
        }
    }
    if (KMode) {
        if (params.n <= KModeMaxSearch) {
            ascendsearch::HNSWGraphPQHybridSearchWithMaskParams HNSW_MaskParam(params.n, params.queryData.data(),
                                                                               params.topK,
                                                                               const_cast<uint8_t *>(mask_data),
                                                                               maskDim, 0, nTotal);
            indexHNSW->SearchWithMask(HNSW_MaskParam, params.dists.data(), params.labels.data());
        } else {
            SearchBatchImplMask(params.n, params.queryData.data(), mask_data, params.topK, params.dists.data(),
                                params.labels.data());
        }
    }
    if (idMap.size() > 0) {
        for (size_t i = 0; i < params.n * params.topK; i++) {
            params.labels[i] = idMap[params.labels[i]];  // map the virtual id back into real base id
        }
    }
    return err;
}


void IndexGreat::SetSearchParams(const IndexGreatSearchParams &params)
{
    setSearchParamsSucceed = false; // 开始设置检索超参之前，将该参数设置为false

    std::string mode;
    if (AKMode) {
        mode = "AKMode";
    }
    if (KMode) {
        mode = "KMode";
    }

    // checking for mismatch
    ASCEND_THROW_IF_NOT_MSG(mode == params.mode, "Search Mode mismatched. Choose between 'KMode' and 'AKMode'.\n");

    if (AKMode) {
        SearchParams newSearchParams;
        newSearchParams.nProbeL1 = params.nProbeL1;
        newSearchParams.nProbeL2 = params.nProbeL2;
        newSearchParams.l3SegmentNum = params.l3SegmentNum;
        IndexVstarSearchParams vstarSearchParams(0, 0, 0);
        vstarSearchParams.params = newSearchParams;
        indexVStar->SetSearchParams(vstarSearchParams);

        ASCEND_THROW_IF_NOT_FMT(10 <= params.ef && params.ef <= 200,
            "search parameter ef must be between 10 and 200 inclusively; your search ef is %d\n", params.ef);
        indexHNSW->innerGraph.efSearch = params.ef;
    }

    if (KMode) {
        ASCEND_THROW_IF_NOT_FMT(10 <= params.ef && params.ef <= 200,
            "search parameter ef must be between 10 and 200 inclusively; your search ef is %d\n", params.ef);
        indexHNSW->innerGraph.efSearch = params.ef;
    }

    setSearchParamsSucceed = true; // 如果顺利设置检索超参，将该参数重置为true
}

IndexGreatSearchParams IndexGreat::GetSearchParams() const
{
    if (AKMode) {
        // assume all innerNpu has the same search params
        IndexVstarSearchParams AModeSearchParams = indexVStar->GetSearchParams();
        int KMode_ef = indexHNSW->innerGraph.efSearch;
        IndexGreatSearchParams greatSearchParam;
        greatSearchParam.mode = "AKMode";
        greatSearchParam.nProbeL1 = AModeSearchParams.params.nProbeL1;
        greatSearchParam.nProbeL2 = AModeSearchParams.params.nProbeL2;
        greatSearchParam.l3SegmentNum = AModeSearchParams.params.l3SegmentNum;
        greatSearchParam.ef = KMode_ef;
        return greatSearchParam;
    }
    if (KMode) {
        int KMode_ef = indexHNSW->innerGraph.efSearch;
        IndexGreatSearchParams greatSearchParam;
        greatSearchParam.mode = "KMode";
        greatSearchParam.ef = KMode_ef;
        return greatSearchParam;
    }
    return IndexGreatSearchParams{};
}

int IndexGreat::GetDim() const
{
    return dim;
}

uint64_t IndexGreat::GetNTotal() const
{
    if (nTotal == 0) {
        printf("Warning: Your nTotal is currently 0, which indicates you have not added a vector base to the index.\n");
    }
    return (uint64_t)nTotal;
}

void IndexGreat::Reset()
{
    printf("You are resetting this instance of IndexGreat; Any parameters inputted during initialization will be saved,"
           "all other data will be reset (added base vectors etc).\n");
    nTotal = 0;
    idMap.clear();
    idMapReverse.clear();
    codebookAdded = false;
    indexCreated = false;
    setSearchParamsSucceed = true;
    if (AKMode) {
        indexVStar->Reset();
        if (verbose) {
            printf("Finished resetting AMode...\n");
        }
        indexHNSW->reset();
        if (verbose) {
            printf("Finished resetting KMode!\n");
        }
    }
    if (KMode) {
        indexHNSW->reset();
        if (verbose) {
            printf("Finished resetting KMode!\n");
        }
    }
}

void IndexGreat::SearchBatchImpl(size_t n, const float *queryData, int topK, float *dists, int64_t *labels)
{
    for (size_t i = 0; i < n; i += KModeMaxSearch) {
        size_t searchable = std::min(n - i, KModeMaxSearch);
        indexHNSW->search(searchable, queryData + i * dim, topK, dists + i * topK, labels + i * topK);
    }
}

void IndexGreat::SearchBatchImplMask(size_t n, const float *queryData, const uint8_t* mask, int topK,
                                     float *dists, int64_t *labels)
{
    size_t maskDim = (nTotal + 7) / 8;
    for (size_t i = 0; i < n; i += KModeMaxSearch) {
        int64_t searchable = static_cast<int64_t>(std::min(n - i, KModeMaxSearch));
        ascendsearch::HNSWGraphPQHybridSearchWithMaskParams HNSW_MaskParam(searchable, queryData + i * static_cast<size_t>(dim), topK,
                                                                           const_cast<uint8_t *>(mask) + i * maskDim,
                                                                           maskDim, 0, nTotal);
        indexHNSW->SearchWithMask(HNSW_MaskParam, dists + i * topK, labels + i * topK);
    }
}

}  // namespace ascendSearchacc
