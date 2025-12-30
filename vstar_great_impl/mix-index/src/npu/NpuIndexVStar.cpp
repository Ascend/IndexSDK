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

#include "npu/NpuIndexVStar.h"

#include <algorithm>
#include <cmath>
#include <atomic>
#include <numeric>
#include <unistd.h>
#include <memory>
#include <vector>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>
#include <cerrno>
#include <iostream>

#include "omp.h"

#include "npu/NpuIndexIVFHSP.h"
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

namespace ascendSearchacc {

namespace {
    const std::vector<int> validDim = {128, 256, 512, 1024};
    const std::vector<int> validNlistL1 = {256, 512, 1024};
    const std::vector<int> validSubSpaceDimL1 = {32, 64, 128};
    constexpr uint64_t NTOTAL_MAX = 1e8;
    constexpr uint64_t NTOTAL_MIN = 1e4;
}

NpuIndexVStar::NpuIndexVStar(const IndexVstarInitParams &params)
{
    dim = params.dim;
    subSpaceDimL1 = params.subSpaceDimL1;
    nlistL1 = params.nlistL1;
    deviceList = params.deviceList;
    verbose = params.verbose;
    ASCEND_THROW_IF_NOT_MSG(std::find(validDim.begin(), validDim.end(), dim) != validDim.end(),
                            "Invalid dim: Should be in [128, 256, 512, 1024].\n");
    ASCEND_THROW_IF_NOT_MSG(std::find(validSubSpaceDimL1.begin(), validSubSpaceDimL1.end(),
        subSpaceDimL1) != validSubSpaceDimL1.end(),
        "Invalid subSpaceDimL1: Should be in [32, 64, 128].\n");
    ASCEND_THROW_IF_NOT_MSG(subSpaceDimL1 < dim, "Invalid subSpaceDimL1: Should be less than dim.\n");
    subSpaceDimL2 = static_cast<int>(subSpaceDimL1/2); // 测试结果表明subSpaceDimL2取subSpaceDimL1/2的效果较好
    ASCEND_THROW_IF_NOT_MSG(std::find(validNlistL1.begin(), validNlistL1.end(), nlistL1) != validNlistL1.end(),
                            "Invalid nlist1: Should be in [256, 512, 1024].\n");
    nlistL2 = (dim == 1024) ? 16 : 32; // 超参设置。dim为1024时，nlist设置为16，效果较好，其余设置为32
    ASCEND_THROW_IF_NOT_MSG(deviceList.size() == 1,
                            "Invalid deviceList: Should contain the id of exactly one NPU device you are using.\n");
    ASCEND_THROW_IF_NOT_FMT(0x8000000 <= params.ascendResources && params.ascendResources <= 0x80000000,
        "Invalid ascendResouces: Should be between 128MB and 2GB (convert number to bytes) inclusively,"
        "Your ascendResources is %lld\n", params.ascendResources);
 
    auto config = NpuIndexConfig(deviceList, params.ascendResources);
    innerHSP = std::make_unique<NpuIndexIVFHSP>(dim, subSpaceDimL1, subSpaceDimL2, nlistL1, nlistL2,
                                                MetricType::METRIC_L2, config);
    paramsChecked = true;
}

NpuIndexVStar::NpuIndexVStar(const std::vector<int> &deviceList, const bool verbose, const int64_t ascendResources)
{
    ASCEND_THROW_IF_NOT_MSG(deviceList.size() == 1,
                            "Invalid deviceList: Should contain the id of exactly one NPU device you are using.\n");
    ASCEND_THROW_IF_NOT_FMT(0x8000000 <= ascendResources && ascendResources <= 0x80000000,
        "Invalid ascendResouces: Should be between 128MB and 2GB (convert number to bytes) inclusively,"
        "Your ascendResources is %lld\n", ascendResources);
    this->verbose = verbose;
    auto config = NpuIndexConfig(deviceList, ascendResources);
    innerHSP = std::make_unique<NpuIndexIVFHSP>(config);
}

void NpuIndexVStar::LoadIndex(std::string indexPath, const NpuIndexVStar *loadedIndex)
{
    // if the user have already added codebooks or created index, give a warning about potential overwrites
    if (codebookAdded || indexCreated) {
        printf("Warning: You have loaded codebooks into the index or created an index already." \
            " Your existing data in the index will not be saved.\n");
    }
    ASCEND_THROW_IF_NOT_MSG(loadedIndex != this, "loaded index cannot be the same as the current index.\n");

    ascendSearchacc::checkSoftLink(indexPath);
    // 将判断流程是否继续的布尔值全部置成false，如果LoadIndex失败则阻止流程继续进行
    paramsChecked = false;
    codebookAdded = false;
    indexCreated = false;
    if (loadedIndex == nullptr) {
        innerHSP->LoadIndex(indexPath);
    } else {
        innerHSP->LoadIndex(indexPath, loadedIndex->innerHSP.get());
    }
    dim = innerHSP->GetDim();
    nlistL1 = innerHSP->GetNlistL1();
    nlistL2 = innerHSP->GetNlistL2();
    subSpaceDimL1 = innerHSP->GetSubDimL1();
    subSpaceDimL2 = innerHSP->GetSubDimL2();
    idMap = innerHSP->GetIdMap();
    for (auto &idPair: idMap) {
        int64_t realId = idPair.first;
        int64_t virtualId = idPair.second;
        idMapReverse.insert(std::make_pair(virtualId, realId)); // 在idMap的基础上，对idMapReverse进行重建
    }
    innerHSP->forwardIdsMapReverse(idMapReverse);
    paramsChecked = true;
    codebookAdded = true;
    indexCreated = true;
}

void NpuIndexVStar::WriteIndex(const std::string &indexPath)
{
    ASCEND_THROW_IF_NOT_MSG(paramsChecked,
                            "No valid parameters found; If you created a Vstar instance without specifying parameters,"
                            " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data; Please add codebooks first.\n");
    ASCEND_THROW_IF_NOT_MSG(indexCreated,
                            "Base vectors not added; Please add base vectors first to create the index first.\n");
    innerHSP->WriteIndex(indexPath);
}

APP_ERROR NpuIndexVStar::AddCodeBooks(const std::string &codeBooksPath)
{
    ASCEND_THROW_IF_NOT_MSG(paramsChecked,
        "No valid parameters found; if you created a Vstar instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    // assuming codebook is a single binary file of codebookL1 + codebookL2, with codebook L1 before codebookL2
    auto ret = 0;
    if (codebookAdded) {
        printf("Warning: You have already loaded codebooks into the index."
               " Your existing data in the index will not be saved.\n");
    }
    ascendSearchacc::checkSoftLink(codeBooksPath);
    // 将判断码本是否添加成功的布尔值置成false，若设置成功在结尾设置true
    codebookAdded = false;
    VstarIOReader codebookReader(codeBooksPath);
    
    // verify flags: 4 char
    char fourcc[4] = { 'C', 'B', 'S', 'P' };
    for (int i = 0; i < 4; i++) {  // verify flags: 4 char
        char checkingFlag = 'a';
        codebookReader.ReadAndCheck((&checkingFlag), sizeof(char));
        if (checkingFlag != fourcc[i]) {
            ASCEND_THROW_MSG("Codebook format is not correct.\n");
        }
    }
    // check input parameters
    int paramCheck = 0;
    codebookReader.ReadAndCheck((&paramCheck), sizeof(int));
    ASCEND_THROW_IF_NOT_FMT(
        paramCheck == dim,
        "Codebook's dim param does not match that of current index."
        " Codebook's dim is %d, current index's corresponding value is %d",
        paramCheck, dim);

    codebookReader.ReadAndCheck((&paramCheck), sizeof(int));
    ASCEND_THROW_IF_NOT_FMT(
        paramCheck == nlistL1,
        "Codebook's nlistL1 param does not match that of current index."
        " Codebook's nlistL1 is %d, current index's corresponding value is %d",
        paramCheck, nlistL1);

    codebookReader.ReadAndCheck((&paramCheck), sizeof(int));
    ASCEND_THROW_IF_NOT_FMT(
        paramCheck == subSpaceDimL1,
        "Codebook's subspaceDimL1 param does not match that of current index."
        " Codebook's subspaceDimL1 is %d, current index's corresponding value is %d",
        paramCheck, subSpaceDimL1);

    codebookReader.ReadAndCheck((&paramCheck), sizeof(int));
    ASCEND_THROW_IF_NOT_FMT(
        paramCheck == (nlistL2 * subSpaceDimL2),
        "Codebook's subspaceDimL2 or nlistL2 param do not match those of current index."
        " Codebook's subspaceDimL2 times nlistL2 is %d, current index's corresponding value is %d",
        paramCheck, subSpaceDimL2 * nlistL2);

    std::vector<float> codeBooksL1(nlistL1 * subSpaceDimL1 * dim);
    codebookReader.ReadAndCheck((codeBooksL1.data()),
                                sizeof(float) * nlistL1 * subSpaceDimL1 * dim);

    std::vector<float> codeBooksL2(nlistL1 * nlistL2 * subSpaceDimL2 * subSpaceDimL1);
    codebookReader.ReadAndCheck((codeBooksL2.data()),
                                sizeof(float) * nlistL1 * nlistL2 * subSpaceDimL2 * subSpaceDimL1);

    ret = innerHSP->AddCodeBooks(codeBooksL1, codeBooksL2);
    codebookAdded = true;
    return ret;
}

APP_ERROR NpuIndexVStar::AddCodeBooks(const NpuIndexVStar *loadedIndex)
{
    ASCEND_THROW_IF_NOT_MSG(
        paramsChecked,
        "No valid parameters found; if you created a Vstar instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(loadedIndex != nullptr, "loaded Index cannot be a null pointer.\n");
    ASCEND_THROW_IF_NOT_MSG(loadedIndex != this, "loaded index cannot be the same as the current index.\n");
    auto ret = 0;
    if (codebookAdded) {
        printf(
            "Warning: You have already loaded codebooks into the index."
            " Your existing data in the index will not be saved.\n");
    }
    ret = innerHSP->AddCodeBooks(loadedIndex->innerHSP.get());
    codebookAdded = true;
    return ret;
}

APP_ERROR NpuIndexVStar::AddVectors(const std::vector<float> &baseRawData)
{
    ASCEND_THROW_IF_NOT_MSG(
        paramsChecked,
        "No valid parameters found; If you created a Vstar instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data, please add codebooks first.\n");

    if (indexCreated && innerHSP->GetAddWithIds()) {
        ASCEND_THROW_MSG(
            "AddVectorsWithIds() and AddVectors() cannot be used together. You already added with ids.\n");
    }

    ASCEND_THROW_IF_NOT_MSG(!baseRawData.empty(), "base vector cannot be empty.\n");
    ASCEND_THROW_IF_NOT_MSG(baseRawData.size() % static_cast<size_t>(dim) == 0, "base vector's size must be a multiple of dim.\n");

    uint64_t n = baseRawData.size() / static_cast<size_t>(dim);

    ASCEND_THROW_IF_NOT_FMT((NTOTAL_MIN <= n) && (n <= NTOTAL_MAX),
        "input base size %llu should be in [%llu, %llu].", n, NTOTAL_MIN, NTOTAL_MAX);
    ASCEND_THROW_IF_NOT_FMT(innerHSP->GetNTotal() + n <= NTOTAL_MAX,
        "Index's ntotal %llu should be in [0, %llu].", innerHSP->GetNTotal() + n, NTOTAL_MAX);
    auto ret = 0;
    ret = innerHSP->AddVectorsVerbose(baseRawData);
    innerHSP->SetAddWithIds(false);
    indexCreated = true;
    return ret;
}

APP_ERROR NpuIndexVStar::AddVectorsWithIds(const std::vector<float> &baseRawData, const std::vector<int64_t>& ids)
{
    ASCEND_THROW_IF_NOT_MSG(paramsChecked,
        "No valid parameters found; If you created a Vstar instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data, please add codebooks first.\n");

    if (indexCreated && !innerHSP->GetAddWithIds()) {
        ASCEND_THROW_MSG(
            "AddVectorsWithIds() and AddVectors() cannot be used together. You already added without ids.\n");
    }

    ASCEND_THROW_IF_NOT_MSG(!baseRawData.empty(), "base vector cannot be empty.\n");
    ASCEND_THROW_IF_NOT_MSG(baseRawData.size() % static_cast<size_t>(dim) == 0, "base vector's size must be a multiple of dim.\n");

    size_t n = baseRawData.size() / static_cast<size_t>(dim);

    ASCEND_THROW_IF_NOT_FMT((NTOTAL_MIN <= n) && (n <= NTOTAL_MAX),
        "input base size %llu should be in [%llu, %llu]", n, NTOTAL_MIN, NTOTAL_MAX);
    ASCEND_THROW_IF_NOT_FMT(innerHSP->GetNTotal() + n <= NTOTAL_MAX,
        "Index's ntotal %llu should be in [0, %llu].", innerHSP->GetNTotal() + n, NTOTAL_MAX);

    int64_t uniqueBaseVecCounter = static_cast<int64_t>(innerHSP->GetUniqueBaseVecCounter());
    ASCEND_THROW_IF_NOT_FMT(ids.size() == n,
                            "ids's and baseRawData's sizes do not match: id's size is %llu, baseRawData's size is %llu",
                            ids.size(), n);
    for (int64_t i = uniqueBaseVecCounter; i < static_cast<int64_t>(n) + uniqueBaseVecCounter; i++) {
        idMap[i] = ids[i - uniqueBaseVecCounter];
        ASCEND_THROW_IF_NOT_MSG(idMapReverse.find(ids[i - uniqueBaseVecCounter]) == idMapReverse.end(),
                                "Unique mapping between virtual and real id is required.\n");
        idMapReverse[ids[i - uniqueBaseVecCounter]] = i;
    }
    innerHSP->forwardIdsMapReverse(idMapReverse);
    innerHSP->SetIdMap(idMap);
    auto ret = 0;
    ret = innerHSP->AddVectorsVerbose(baseRawData);
    innerHSP->SetAddWithIds(true);
    indexCreated = true;
    return ret;
}

APP_ERROR NpuIndexVStar::Search(const SearchImplParams &params)
{
    ASCEND_THROW_IF_NOT_MSG(
        paramsChecked,
        "No valid parameters found; If you created a Vstar instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data; Please add codebooks first.\n");
    ASCEND_THROW_IF_NOT_MSG(indexCreated,
                            "Base vectors not added; Please add base vectors first to create the index first.\n");
    ASCEND_THROW_IF_NOT_MSG(setSearchParamsSucceed, "Setting Search Parameter failed.\n");
    ASCEND_THROW_IF_NOT_MSG(this->innerHSP != nullptr, "Vstar's innerHSP structure is nullptr.\n");
    auto ret = 0;

    if (params.topK != topKVstar) {
        ASCEND_THROW_IF_NOT_FMT((0 < params.topK) && (params.topK <= 4096),
            "topK must be greater than 0 and less than or equal to 4096; you are setting it to %d\n", params.topK);
        topKVstar = params.topK;
        innerHSP->SetTopKDuringSearch(topKVstar);
    }

    {
        ASCEND_THROW_IF_NOT_MSG((0 < params.n) && (params.n <= 10000),
                                "SearchSize Must be greater than 0 and less than or equal to 10000.\n");
        ASCEND_THROW_IF_NOT_MSG(params.queryData.size() >= params.n * static_cast<size_t>(dim),
                                "Queries vector must have at least n * dim elements.\n");
        ASCEND_THROW_IF_NOT_MSG(params.dists.size() >= params.n * static_cast<size_t>(topKVstar),
                                "Dist vector must have at least n * topK vacancies.\n");
        ASCEND_THROW_IF_NOT_MSG(params.labels.size() >= params.n * static_cast<size_t>(topKVstar),
                                "Label vector must have at least n * topK vacancies.\n");
    }

    ret = innerHSP->Search(params.n, params.queryData.data(), params.topK, params.dists.data(), params.labels.data());
    if (idMap.size() > 0) {
        for (size_t i = 0; i < params.n * params.topK; i++) {
            params.labels[i] = idMap[params.labels[i]];
        }
    }
    return ret;
}

APP_ERROR NpuIndexVStar::Search(const SearchImplParams &params, const std::vector<uint8_t> &mask)
{
    ASCEND_THROW_IF_NOT_MSG(paramsChecked,
        "No valid parameters found; If you created a Vstar instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data; Please add codebooks first.\n");
    ASCEND_THROW_IF_NOT_MSG(indexCreated,
                            "Base vectors not added; Please add base vectors first to create the index first.\n");
    ASCEND_THROW_IF_NOT_MSG(setSearchParamsSucceed, "Setting Search Parameter failed.\n");
    ASCEND_THROW_IF_NOT_MSG(this->innerHSP != nullptr, "Vstar's innerHSP structure is nullptr.\n");

    if (params.topK != topKVstar) {
        ASCEND_THROW_IF_NOT_FMT((0 < params.topK) && (params.topK <= 4096),
            "topK must be greater than 0 and less than or equal to 4096; you are setting it to %d\n", params.topK);
        topKVstar = params.topK;
        innerHSP->SetTopKDuringSearch(topKVstar);
    }

    size_t maskDim = (innerHSP->GetNTotal() + 7) / 8;
    {
        ASCEND_THROW_IF_NOT_MSG((0 < params.n) && (params.n <= 10000),
                                "SearchSize Must be greater than 0 and less than or equal to 10000.\n");
        ASCEND_THROW_IF_NOT_MSG(params.queryData.size() >= params.n * static_cast<size_t>(dim),
                                "Queries vector must have at least n * dim elements.\n");
        ASCEND_THROW_IF_NOT_MSG(params.dists.size() >= params.n * static_cast<size_t>(topKVstar),
                                "Dist vector must have at least n * topK vacancies.\n");
        ASCEND_THROW_IF_NOT_MSG(params.labels.size() >= params.n * static_cast<size_t>(topKVstar),
                                "Label vector must have at least n * topK vacancies.\n");
        ASCEND_THROW_IF_NOT_MSG(mask.size() >= (params.n * maskDim),
                                "Mask Vector must have at least Ceil(NTotal/8) elements.\n");
    }

    auto ret = 0;
    const uint8_t *mask_data = mask.data();
    ret = innerHSP->Search(params.n, const_cast<uint8_t *>(mask_data),
                           params.queryData.data(), params.topK, params.dists.data(), params.labels.data());

    if (idMap.size() > 0) {
        for (size_t i = 0; i < params.n * params.topK; i++) {
            params.labels[i] = idMap[params.labels[i]];
        }
    }
    return ret;
}

APP_ERROR NpuIndexVStar::MultiSearch(const std::vector<NpuIndexVStar *> &indexes, const SearchImplParams &params,
                                     bool merge)
{
    ASCEND_THROW_IF_NOT_MSG(
        paramsChecked,
        "No valid parameters found; If you created a Vstar instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data; Please add codebooks first.\n");
    ASCEND_THROW_IF_NOT_MSG(indexCreated,
                            "Base vectors not added; Please add base vectors first to create the index first.\n");
    ASCEND_THROW_IF_NOT_MSG(setSearchParamsSucceed, "Setting Search Parameter failed.\n");
    ASCEND_THROW_IF_NOT_MSG(this->innerHSP != nullptr, "Vstar's innerHSP structure is nullptr.\n");

    {
        ASCEND_THROW_IF_NOT_MSG((0 < params.n) && (params.n <= 10000),
                                "SearchSize Must be greater than 0 and less than or equal to 10000.\n");
        ASCEND_THROW_IF_NOT_MSG(params.queryData.size() >= params.n * static_cast<size_t>(dim),
                                "Queries vector must have at least n * dim elements.\n");
    }

    if (params.topK != topKVstar) {
        ASCEND_THROW_IF_NOT_FMT((0 < params.topK) && (params.topK <= 4096),
            "topK must be greater than 0 and less than or equal to 4096; you are setting it to %d\n", params.topK);
        topKVstar = params.topK;
        innerHSP->SetTopKDuringSearch(topKVstar);
    }

    if (merge) {
        ASCEND_THROW_IF_NOT_MSG(params.dists.size() >= params.n * static_cast<size_t>(topKVstar),
                                "Merging: Dist vector must have at least n * topK vacancies.\n");
        ASCEND_THROW_IF_NOT_MSG(params.labels.size() >= params.n * static_cast<size_t>(topKVstar),
                                "Merging: Label vector must have at least n * topK vacancies.\n");
    } else {
        ASCEND_THROW_IF_NOT_MSG(params.dists.size() >= params.n * indexes.size() * static_cast<size_t>(topKVstar),
                                "Not Merging: Dist vector must have at least n * indexes.size() * topK vacancies.\n");
        ASCEND_THROW_IF_NOT_MSG(params.labels.size() >= params.n * indexes.size() * static_cast<size_t>(topKVstar),
                                "Not Merging: Label vector must have at least n * indexes.size() * topK vacancies.\n");
    }
    
    auto ret = 0;
    std::vector<NpuIndexIVFHSP *> innerHSPList;
    for (auto &index : indexes) {
        ASCEND_THROW_IF_NOT_MSG(index != nullptr, "Indices to be searched in MultiIndex cannot be nullptrs.\n");
        ASCEND_THROW_IF_NOT_MSG(index->IsIndexValid(),
            "Indices to be searched in MultiIndex must be loaded index with valid data to be searched.\n");
        innerHSPList.emplace_back(index->innerHSP.get());
    }
    ret = innerHSP->Search(innerHSPList, params.n, params.queryData.data(),
        params.topK, params.dists.data(), params.labels.data(), merge);
    return ret;
}

APP_ERROR NpuIndexVStar::MultiSearch(const std::vector<NpuIndexVStar *> &indexes, const SearchImplParams &params,
                                     const std::vector<uint8_t> &mask, bool merge)
{
    ASCEND_THROW_IF_NOT_MSG(
        paramsChecked,
        "No valid parameters found; If you created a Vstar instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data; Please add codebooks first.\n");
    ASCEND_THROW_IF_NOT_MSG(indexCreated,
                            "Base vectors not added; Please add base vectors first to create the index first.\n");
    ASCEND_THROW_IF_NOT_MSG(setSearchParamsSucceed, "Setting Search Parameter failed.\n");
    ASCEND_THROW_IF_NOT_MSG(this->innerHSP != nullptr, "Vstar's innerHSP structure is nullptr.\n");

    auto ret = 0;
    uint64_t maxNTotal = 0;
    std::vector<NpuIndexIVFHSP *> innerHSPList;
    for (auto &index : indexes) {
        ASCEND_THROW_IF_NOT_MSG(index != nullptr, "Indices to be searched in MultiIndex cannot be nullptrs.\n");
        ASCEND_THROW_IF_NOT_MSG(index->IsIndexValid(),
            "Indices to be searched in MultiIndex must be loaded index with valid data to be searched.\n");
        innerHSPList.emplace_back(index->innerHSP.get());
        maxNTotal = std::max(maxNTotal, index->GetNTotal());
    }
    size_t maxMaskSize = (maxNTotal + 7) / 8;

    {
        ASCEND_THROW_IF_NOT_MSG((0 < params.n) && (params.n <= 10000),
                                "SearchSize Must be greater than 0 and less than or equal to 10000.\n");
        ASCEND_THROW_IF_NOT_MSG(params.queryData.size() >= params.n * static_cast<size_t>(dim),
                                "Queries vector must have at least n * dim elements.\n");
        ASCEND_THROW_IF_NOT_MSG(mask.size() >= params.n * maxMaskSize,
                                "Mask Vector must have at least Ceil(NTotal/8) elements.\n");
    }

    if (params.topK != topKVstar) {
        ASCEND_THROW_IF_NOT_FMT((0 < params.topK) && (params.topK <= 4096),
            "topK must be greater than 0 and less than or equal to 4096; you are setting it to %d\n", params.topK);
        topKVstar = params.topK;
        innerHSP->SetTopKDuringSearch(topKVstar);
    }

    if (merge) {
        ASCEND_THROW_IF_NOT_MSG(params.dists.size() >= params.n * static_cast<size_t>(topKVstar),
                                "Merging: Dist vector must have at least n * topK vacancies.\n");
        ASCEND_THROW_IF_NOT_MSG(params.labels.size() >= params.n * static_cast<size_t>(topKVstar),
                                "Merging: Label vector must have at least n * topK vacancies.\n");
    } else {
        ASCEND_THROW_IF_NOT_MSG(params.dists.size() >= params.n * indexes.size() * static_cast<size_t>(topKVstar),
                                "Not Merging: Dist vector must have at least n * indexes.size() * topK vacancies.\n");
        ASCEND_THROW_IF_NOT_MSG(params.labels.size() >= params.n * indexes.size() * static_cast<size_t>(topKVstar),
                                "Not Merging: Label vector must have at least n * indexes.size() * topK vacancies.\n");
    }

    const uint8_t* mask_data = mask.data();
    ret = innerHSP->Search(innerHSPList, params.n, const_cast<uint8_t *>(mask_data),
        params.queryData.data(), params.topK, params.dists.data(), params.labels.data(), merge);
    return ret;
}

APP_ERROR NpuIndexVStar::DeleteVectors(const std::vector<int64_t> &ids)
{
    ASCEND_THROW_IF_NOT_MSG(
        paramsChecked,
        "No valid parameters found; If you created a Vstar instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data; Please add codebooks first.\n");
    ASCEND_THROW_IF_NOT_MSG(indexCreated,
                            "Base vectors not added; Please add base vectors first to create the index first.\n");
    auto ret = 0;
    ret = innerHSP->DeleteVectors(ids);
    idMap = innerHSP->GetIdMap();
    idMapReverse = innerHSP->GetIdsMapReverse();
    return ret;
}

APP_ERROR NpuIndexVStar::DeleteVectors(const int64_t &id)
{
    ASCEND_THROW_IF_NOT_MSG(
        paramsChecked,
        "No valid parameters found; If you created a Vstar instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data; Please add codebooks first.\n");
    ASCEND_THROW_IF_NOT_MSG(indexCreated,
                            "Base vectors not added; Please add base vectors first to create the index first.\n");
    auto ret = 0;
    ret = innerHSP->DeleteVectors(id);
    idMap = innerHSP->GetIdMap();
    idMapReverse = innerHSP->GetIdsMapReverse();
    return ret;
}

APP_ERROR NpuIndexVStar::DeleteVectors(const int64_t &startId, const int64_t &endId)
{
    ASCEND_THROW_IF_NOT_MSG(
        paramsChecked,
        "No valid parameters found; If you created a Vstar instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data; Please add codebooks first.\n");
    ASCEND_THROW_IF_NOT_MSG(indexCreated,
                            "Base vectors not added; Please add base vectors first to create the index first.\n");
    ASCEND_THROW_IF_NOT_FMT(endId >= startId, "endId is less than startId: endId is %lld, startId is %lld\n",
        endId, startId);
    auto ret = 0;
    ret = innerHSP->DeleteVectors(startId, endId);
    idMap = innerHSP->GetIdMap();
    idMapReverse = innerHSP->GetIdsMapReverse();
    return ret;
}

void NpuIndexVStar::SetSearchParams(const IndexVstarSearchParams &params)
{
    setSearchParamsSucceed = false; // 开始设置检索超参之前，将该参数设置为false

    if (verbose) {
        printf(
            "Warning: Before setting the search parameters, you should make sure you have generated compatible AscendC"
            " oprerators with the provided search parameters, otherwise an initialization error is expected\n");
    }
    ASCEND_THROW_IF_NOT_FMT(
        params.params.nProbeL1 > 16 && params.params.nProbeL1 <= nlistL1 && params.params.nProbeL1 % 8 == 0,
        "nProbeL1 must be 1) greater than 16; 2) less than nlistL1; 3) divisible by 8."
        " You are setting nProbeL1 to %d, which is invalid.\n",
        params.params.nProbeL1);
    ASCEND_THROW_IF_NOT_FMT(
        params.params.nProbeL2 > 16 && params.params.nProbeL2 <= (params.params.nProbeL1 * nlistL2) &&
        params.params.nProbeL2 % 8 == 0, // 被8整除，在ub搬运的时候就可以对齐
        "nProbeL2 must be 1) greater than 16; 2) less than (nProbeL1 * nlistL2); 3) divisible by 8."
        " You are setting nProbeL2 to %d, which is invalid.\n",
        params.params.nProbeL2);
    ASCEND_THROW_IF_NOT_FMT(
        params.params.l3SegmentNum > 100 && params.params.l3SegmentNum <= 5000 && params.params.l3SegmentNum % 8 == 0,
        "nProbeL2 must be 1) greater than 100 and less than 5000; 2) divisible by 8."
        " You are setting l3Segment to %d, which is invalid.\n",
        params.params.l3SegmentNum);

    innerHSP->SetSearchParams(params.params);

    setSearchParamsSucceed = true; // 如果顺利设置检索超参，将该参数重置为true
}

IndexVstarSearchParams NpuIndexVStar::GetSearchParams() const
{
    SearchParams existingSearchParams = innerHSP->GetSearchParams();
    IndexVstarSearchParams returnedVstarSearchParams(0, 0, 0);
    returnedVstarSearchParams.params = existingSearchParams;
    return returnedVstarSearchParams;
}

uint64_t NpuIndexVStar::GetNTotal() const
{
    uint64_t ntotal = 0;
    if (!paramsChecked) {
        if (verbose) {
            printf("Warning: You have not initialized your Vstar instance with valid parameters and"
                   " added your base vectors, so your ntotal is uninitialized (0).\n");
        }
        return 0;
    }
    ntotal = innerHSP->GetNTotal();
    return ntotal;
}

int NpuIndexVStar::GetDim() const
{
    if (!paramsChecked) {
        if (verbose) {
            printf("Warning: You have not initialized your Vstar instance with valid parameters and"
                   " added your base vectors, so your dim is uninitialized (0).\n");
        }
        return 0;
    }
    return dim;
}

void NpuIndexVStar::SetTopK(int topK)
{
    this->topKVstar = topK;
    ASCEND_THROW_IF_NOT_MSG(this->innerHSP != nullptr, "Vstar's innerHSP structure is nullptr.\n");
    this->innerHSP->SetTopKDuringSearch(topK);
}

APP_ERROR NpuIndexVStar::Reset()
{
    innerHSP.reset();
    idMap.clear();
    idMapReverse.clear();
    codebookAdded = false;
    indexCreated = false;
    setSearchParamsSucceed = true;
    if (dim > 0 && nlistL1 > 0 && nlistL2 > 0 && subSpaceDimL1 > 0 && subSpaceDimL2 > 0) {
        if (verbose) {
            printf("IndexVstar: Resetting an index with valid input parameters...\n");
        }
        auto config = NpuIndexConfig(deviceList);
        innerHSP = std::make_unique<NpuIndexIVFHSP>(dim, subSpaceDimL1, subSpaceDimL2, nlistL1, nlistL2,
                                                    MetricType::METRIC_L2, config);
        return 0;
    } else {
        if (verbose) {
            printf("IndexVstar: Resetting an index with no input parameters...\n");
        }
        auto config = NpuIndexConfig(deviceList);
        innerHSP = std::make_unique<NpuIndexIVFHSP>(config);
        dim = 0;
        nlistL1 = 0;
        nlistL2 = 0;
        subSpaceDimL1 = 0;
        subSpaceDimL2 = 0;
        return 0;
    }
}

APP_ERROR NpuIndexVStar::AddVectorsTmp(const std::vector<float> &baseRawData, const bool tmp)
{
    ASCEND_THROW_IF_NOT_MSG(paramsChecked,
        "No valid parameters found; If you created a Vstar instance without specifying parameters,"
        " please make sure you have loaded the index.\n");
    ASCEND_THROW_IF_NOT_MSG(codebookAdded, "No valid codebook data, please add codebooks first.\n");
    auto ret = 0;
    ret = innerHSP->AddVectorsVerbose(baseRawData, tmp);
    indexCreated = true;
    return ret;
}

bool NpuIndexVStar::IsIndexValid() const
{
    return (paramsChecked && codebookAdded && indexCreated);
}

}  // namespace ascendSearchacc
