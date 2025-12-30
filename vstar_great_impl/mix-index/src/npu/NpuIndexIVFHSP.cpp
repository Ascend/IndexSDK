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

#include "npu/NpuIndexIVFHSP.h"
#include <algorithm>
#include <cmath>
#include <atomic>
#include <numeric>
#include <unistd.h>
#include <memory>
#include <vector>
#include <limits>
#include <iostream>
#include <sys/stat.h>
#include <set>
#include <fcntl.h>
#include <cerrno>
#include <chrono>
#include <fstream>
#include <cmath>
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

#ifdef USE_ACL_NN_INTERFACE

#include "aclnn/acl_meta.h"
#include "aclnn_vstar_compute_l1.h"
#include "aclnn_vstar_compute_l2.h"
#include "aclnn_vstar_compute_l3.h"
#endif

namespace ascendSearchacc {
constexpr float EPSILON = 1e-6; // 浮点比较门槛

inline bool FloatEqual(float a, float b)
{
    return fabs(a - b) < EPSILON;
}

union FormatConvert {
    float16_t data;
    uint16_t value;
};
namespace {
const int THREADS_CNT_T = 0;
const int L2_QUERY_BATCH = 32;
const int L3_QUERY_BATCH = 32;
const int BASE_SIZE = 16384;
const uint16_t MAXVALUE_UINT16 = 0xFFFF;  // 65535
const int MAX_NLIST2 = 1024;
const std::vector<int> validDim = {128, 256, 512, 1024};
const std::vector<int> validNlistL1 = {256, 512, 1024};
const std::vector<int> validNlistL2 = {16, 32};
const std::vector<int> validSubSpaceDimL1 = {32, 64, 128};
const int MAXINDEXNUM = 150;
constexpr int DIM_UB = 1024;
constexpr int N_TOTAL_UB = 1e8;
}  // namespace

std::unique_ptr<ThreadPool> NpuIndexIVFHSP::pool = std::make_unique<ThreadPool>(omp_get_max_threads());

NpuIndexIVFHSP::NpuIndexIVFHSP(NpuIndexConfig config) : NpuIndexIVF(config)
{
    this->searchParam = std::make_unique<SearchParams>();
    searchL1OpAttrs = std::shared_ptr<AscendTensor<int64_t, DIMS_1> >(
        new AscendTensor<int64_t, DIMS_1>({ static_cast<int>(TopkIvfSpL1AttrIdx::TOPK_L1_ATTR_IDX_COUNT) }));
    searchL2OpAttrs = std::shared_ptr<AscendTensor<int64_t, DIMS_1> >(
        new AscendTensor<int64_t, DIMS_1>({ static_cast<int>(TopkIvfSpL2AttrIdx::TOPK_L2_ATTR_IDX_COUNT) }));
    searchMultiL2OpAttrs = std::shared_ptr<AscendTensor<int64_t, DIMS_1> >(
        new AscendTensor<int64_t, DIMS_1>({ static_cast<int>(IvfMultiSpTopkL2AttrIdx::TOPK_L2_ATTR_IDX_COUNT) }));
    searchL3OpAttrs = std::shared_ptr<AscendTensor<int64_t, DIMS_1> >(
        new AscendTensor<int64_t, DIMS_1>({ static_cast<int>(TopkIvfSpL3AttrIdx::TOPK_L3_ATTR_IDX_COUNT) }));
    defaultStream = resources->getDefaultStream();
    aiCpuStream = resources->getAlternateStreams()[0];
}

NpuIndexIVFHSP::NpuIndexIVFHSP(int dim, int subSpaceDim1, int subSpaceDim2, int nList1, int nList2,
                               MetricType metricType, NpuIndexConfig config)
    : NpuIndexIVF(dim, nList1, metricType, config),
      threadPool(std::make_unique<AscendThreadPool>(THREADS_CNT_T)),
      sqHandler(std::make_unique<faiss::ScalarQuantizer>(subSpaceDim1, faiss::ScalarQuantizer::QuantizerType::QT_8bit)),
      nListL2(nList2),
      subSpaceDimL1(subSpaceDim1),
      subSpaceDimL2(subSpaceDim2),
      dimStored(subSpaceDim1),
      minAddressOfBaseNpu(nullptr)
{
    ASCEND_THROW_IF_NOT(std::find(validDim.begin(), validDim.end(), dim) != validDim.end());
    ASCEND_THROW_IF_NOT(subSpaceDimL1 > 0 && subSpaceDimL1 % CUBE_ALIGN == 0 && subSpaceDimL1 < dim);
    ASCEND_THROW_IF_NOT(subSpaceDimL2 > 0 && subSpaceDimL2 % CUBE_ALIGN == 0 && subSpaceDimL2 < subSpaceDimL1);
    ASCEND_THROW_IF_NOT(std::find(validNlistL1.begin(), validNlistL1.end(), nList1) != validNlistL1.end());
    ASCEND_THROW_IF_NOT(std::find(validNlistL2.begin(), validNlistL2.end(), nList2) != validNlistL2.end());

    std::initializer_list<int> vDiffNpuDims = {dimStored};
    vDiff2Npu = std::make_shared<AscendTensor<float16_t, DIMS_1> >(vDiffNpuDims);
    vDiffNpu = std::make_shared<AscendTensor<float16_t, DIMS_1> >(vDiffNpuDims);

    bucketOffsetInPage.resize(nList1 * nList2, -1);
    std::initializer_list<int> addressOffsetOfBucketDims = {nList1 * nList2 * 6};
    addressOffsetOfBucket = std::make_shared<AscendTensor<uint64_t, DIMS_1> >(addressOffsetOfBucketDims);
    addressOffsetOfBucketCPU.resize(nList1 * nList2 * 6);

    this->searchParam = std::make_unique<SearchParams>();
    searchL1OpAttrs = std::shared_ptr<AscendTensor<int64_t, DIMS_1> >(
        new AscendTensor<int64_t, DIMS_1>({ static_cast<int>(TopkIvfSpL1AttrIdx::TOPK_L1_ATTR_IDX_COUNT) }));
    searchL2OpAttrs = std::shared_ptr<AscendTensor<int64_t, DIMS_1> >(
        new AscendTensor<int64_t, DIMS_1>({ static_cast<int>(TopkIvfSpL2AttrIdx::TOPK_L2_ATTR_IDX_COUNT) }));
    searchMultiL2OpAttrs = std::shared_ptr<AscendTensor<int64_t, DIMS_1> >(
        new AscendTensor<int64_t, DIMS_1>({ static_cast<int>(IvfMultiSpTopkL2AttrIdx::TOPK_L2_ATTR_IDX_COUNT) }));
    searchL3OpAttrs = std::shared_ptr<AscendTensor<int64_t, DIMS_1> >(
        new AscendTensor<int64_t, DIMS_1>({ static_cast<int>(TopkIvfSpL3AttrIdx::TOPK_L3_ATTR_IDX_COUNT) }));

    defaultStream = resources->getDefaultStream();
    aiCpuStream = resources->getAlternateStreams()[0];

    codeWordsByBucket.resize(nList * nListL2, std::vector<uint8_t>());
    idxByBucket.resize(nList * nListL2, std::vector<int64_t>());
    normL2ByBucket.resize(nList * nListL2, std::vector<float>());

    this->Init();
}

NpuIndexIVFHSP::~NpuIndexIVFHSP() noexcept
{
    APP_LOG_INFO("NpuIndexIVFHSP DeConstruct Start.\n");
#ifndef USE_ACL_NN_INTERFACE
    l1DistOps.clear();
    l2DistOps.clear();
    l3DistOps.clear();
    l3DistWithMaskOps.clear();
#endif
    l1TopKOps.clear();
    l2TopKOps.clear();
    l2TopKWithMaskOps.clear();
    l3TopKOps.clear();
    defaultStream = nullptr;
    aiCpuStream = nullptr;
    if (aclrtSetDevice(ascendConfig.deviceList[0]) != ACL_SUCCESS) {
        APP_LOG_ERROR("Set device failed. deviceId is %d", ascendConfig.deviceList[0]);
    }
    APP_LOG_INFO("NpuIndexIVFHSP DeConstruct End.\n");
}

APP_ERROR NpuIndexIVFHSP::Init()
{
    // reset ascend opp
    APP_LOG_INFO("NpuIndexIVFHSP Init Ops Start.\n");
#ifndef USE_ACL_NN_INTERFACE
    ACL_REQUIRE_OK(ResetL1DistOp());
    ACL_REQUIRE_OK(ResetL2DistOp());
    ACL_REQUIRE_OK(ResetL3DistOp());
    ACL_REQUIRE_OK(ResetL3DistWithMaskOp());
#endif
    ACL_REQUIRE_OK(ResetMatMul());
    ACL_REQUIRE_OK(ResetL1TopKOp());
    ACL_REQUIRE_OK(ResetL2TopKOp());
    ACL_REQUIRE_OK(ResetMultiL2TopKOp());

    vts = std::make_shared<std::vector<VisitedTable>>(MAXINDEXNUM);
    ACL_REQUIRE_OK(ResetMultiL3TopKOp(1));
    ACL_REQUIRE_OK(ResetL2TopKWithMaskOp());
    ACL_REQUIRE_OK(ResetL3TopKOp());
    APP_LOG_INFO("NpuIndexIVFHSP Init Ops End.\n NpuIndexIVFHSP Init Attrs.\n");
    UpdateOpAttrs();
    APP_LOG_INFO("NpuIndexIVFHSP Init Attrs End.\n");
    return APP_ERR_OK;
}

void NpuIndexIVFHSP::SetSearchParams(SearchParams params)
{
    this->searchParam->nProbeL1 = params.nProbeL1;
    this->searchParam->nProbeL2 = params.nProbeL2;
    this->searchParam->l3SegmentNum = params.l3SegmentNum;
    UpdateOpAttrs();
#ifndef USE_ACL_NN_INTERFACE
    ACL_REQUIRE_OK(ResetL1DistOp());
    ACL_REQUIRE_OK(ResetL2DistOp());
    ACL_REQUIRE_OK(ResetL3DistOp());
    ACL_REQUIRE_OK(ResetL3DistWithMaskOp());
#endif
    ACL_REQUIRE_OK(ResetMatMul());
    ACL_REQUIRE_OK(ResetL1TopKOp());
    ACL_REQUIRE_OK(ResetL2TopKOp());
    ACL_REQUIRE_OK(ResetMultiL2TopKOp());
    vts = std::make_shared<std::vector<VisitedTable>>(MAXINDEXNUM);
    ACL_REQUIRE_OK(ResetMultiL3TopKOp(1));
    ACL_REQUIRE_OK(ResetL2TopKWithMaskOp());
    ACL_REQUIRE_OK(ResetL3TopKOp());
}

SearchParams NpuIndexIVFHSP::GetSearchParams() const
{
    SearchParams existingSearchParams;
    existingSearchParams.nProbeL1 = this->searchParam->nProbeL1;
    existingSearchParams.nProbeL2 = this->searchParam->nProbeL2;
    existingSearchParams.l3SegmentNum = this->searchParam->l3SegmentNum;

    return existingSearchParams;
}

std::shared_ptr<const AscendTensor<float16_t, DIMS_4> > NpuIndexIVFHSP::GetCodeBooksL1NPU() const
{
    return std::static_pointer_cast<const AscendTensor<float16_t, DIMS_4> >(codeBooksShapedL1Npu);
}

std::shared_ptr<const AscendTensor<float16_t, DIMS_4> > NpuIndexIVFHSP::GetCodeBooksL2NPU() const
{
    return std::static_pointer_cast<const AscendTensor<float16_t, DIMS_4> >(codeBooksShapedL2Npu);
}

const float *NpuIndexIVFHSP::GetCodeBooksL1CPU() const
{
    return codeBooksL1Cpu.data();
}

const float *NpuIndexIVFHSP::GetCodeBooksL2CPU() const
{
    return codeBooksL2Cpu.data();
}

void NpuIndexIVFHSP::WriteIndex(std::string indexPath)
{
    VstarIOWriter indexWriter(indexPath);

    // write Index Flag
    char fourcc[4] = { 'I', 'W', 'V', 'S' };
    for (int i = 0; i < 4; i++) {
        indexWriter.WriteAndCheck(&(fourcc[i]), sizeof(char));
    }

    // write Index Params
    indexWriter.WriteAndCheck((&nList), sizeof(nList));
    indexWriter.WriteAndCheck((&nListL2), sizeof(nListL2));
    indexWriter.WriteAndCheck((&subSpaceDimL1), sizeof(subSpaceDimL1));
    indexWriter.WriteAndCheck((&subSpaceDimL2), sizeof(subSpaceDimL2));
    indexWriter.WriteAndCheck((&dim), sizeof(dim));
    indexWriter.WriteAndCheck((&dimStored), sizeof(dimStored));
    indexWriter.WriteAndCheck((&ntotal), sizeof(ntotal));
    indexWriter.WriteAndCheck((&trained), sizeof(trained));
    indexWriter.WriteAndCheck((&metricType), sizeof(metricType));

    std::vector<float16_t> vDiff1Cpu(dimStored);
    std::vector<float16_t> vDiff2Cpu(dimStored);
    auto ret = aclrtMemcpy(vDiff1Cpu.data(), vDiff1Cpu.size() * sizeof(float16_t), vDiffNpu->data(),
                           vDiffNpu->getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    ACL_REQUIRE_OK(ret);
    ret = aclrtMemcpy(vDiff2Cpu.data(), vDiff2Cpu.size() * sizeof(float16_t), vDiff2Npu->data(),
                      vDiff2Npu->getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    ACL_REQUIRE_OK(ret);
    int vDiff1CpuSize = static_cast<int>(vDiff1Cpu.size());
    indexWriter.WriteAndCheck((&vDiff1CpuSize), sizeof(vDiff1CpuSize));
    indexWriter.WriteAndCheck((vDiff1Cpu.data()), sizeof(float16_t) * vDiff1Cpu.size());
    int vDiff2CpuSize = static_cast<int>(vDiff2Cpu.size());
    indexWriter.WriteAndCheck((&vDiff2CpuSize), sizeof(vDiff2CpuSize));
    indexWriter.WriteAndCheck((vDiff2Cpu.data()), sizeof(float16_t) * vDiff2Cpu.size());
    indexWriter.WriteAndCheck((sqHandler->trained.data()), sizeof(float) * sqHandler->trained.size());

    // write codebook
    indexWriter.WriteAndCheck((codeBooksL1Cpu.data()), sizeof(float) * codeBooksL1Cpu.size());
    indexWriter.WriteAndCheck((codeBooksL2Cpu.data()), sizeof(float) * codeBooksL2Cpu.size());

    // write precomputeList data
    int precomputeListSize = static_cast<int>(precomputeNormL2Npu->size());
    std::vector<float16_t> precomputeList(precomputeListSize);
    ret = aclrtMemcpy(precomputeList.data(), precomputeNormL2Npu->size() * sizeof(float16_t),
                      precomputeNormL2Npu->data(), precomputeNormL2Npu->size() * sizeof(float16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ACL_REQUIRE_OK(ret);
    indexWriter.WriteAndCheck((&precomputeListSize), sizeof(precomputeListSize));
    indexWriter.WriteAndCheck((precomputeList.data()), sizeof(float16_t) * precomputeList.size());
    // write base data
    for (int i = 0; i < nList; i++) {
        int idsListSize = static_cast<int>(ids[i]->size());
        indexWriter.WriteAndCheck((&idsListSize), sizeof(idsListSize));
        if (idsListSize > 0) {
            std::vector<int64_t> idsList(idsListSize);
            ret = aclrtMemcpy(idsList.data(), idsList.size() * sizeof(int64_t), ids[i]->data(),
                              idsList.size() * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
            ACL_REQUIRE_OK(ret);
            indexWriter.WriteAndCheck((idsList.data()), sizeof(int64_t) * idsList.size());
        }
    }

    for (int i = 0; i < nList; i++) {
        size_t baseListSize = baseShaped[i]->size();
        indexWriter.WriteAndCheck((&baseListSize), sizeof(baseListSize));
        if (baseListSize > 0) {
            std::vector<uint8_t> baseList(baseListSize);
            ret = aclrtMemcpy(baseList.data(), baseList.size() * sizeof(uint8_t), baseShaped[i]->data(),
                              baseList.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
            ACL_REQUIRE_OK(ret);
            indexWriter.WriteAndCheck((baseList.data()), sizeof(uint8_t) * baseList.size());
        }
    }

    int listSize = static_cast<int>(bucketOffsetInPage.size());
    indexWriter.WriteAndCheck((&listSize), sizeof(listSize));
    indexWriter.WriteAndCheck((bucketOffsetInPage.data()), sizeof(size_t) * bucketOffsetInPage.size());

    // 07/25: Saving byBucket data (Consider remove saving data above; they can be recomputed in AddIntoNpuStore)
    for (size_t i = 0; i < idxByBucket.size(); ++i) {
        int globalBucketSize = static_cast<int>(idxByBucket[i].size());
        indexWriter.WriteAndCheck((&globalBucketSize), sizeof(globalBucketSize));
        indexWriter.WriteAndCheck((idxByBucket[i].data()), globalBucketSize * sizeof(int64_t));
        indexWriter.WriteAndCheck((normL2ByBucket[i].data()), globalBucketSize * sizeof(float));
        indexWriter.WriteAndCheck((codeWordsByBucket[i].data()), globalBucketSize * dimStored * sizeof(uint8_t));
    }

    indexWriter.WriteAndCheck((&isAddWithIds), sizeof(isAddWithIds));
    int idMapSize = static_cast<int>(idMap.size());
    indexWriter.WriteAndCheck((&idMapSize), sizeof(idMapSize));

    for (const auto &idPair : idMap) {
        indexWriter.WriteAndCheck((&idPair.first), sizeof(idPair.first));
        indexWriter.WriteAndCheck((&idPair.second), sizeof(idPair.second));
    }

    indexWriter.WriteAndCheck((&uniqueBaseVecCounter), sizeof(uniqueBaseVecCounter));
}

void NpuIndexIVFHSP::LoadIndex(std::string indexPath, const NpuIndexIVFHSP *loadedIndex)
{
    VstarIOReader indexReader(indexPath);

    // load Index Flag
    char fourcc[4] = { 'I', 'W', 'V', 'S' };
    for (int i = 0; i < 4; i++) { // fourcc hava 4 elements
        char checkingFlag = 'a';
        indexReader.ReadAndCheck((&checkingFlag), sizeof(char));
        if (checkingFlag != fourcc[i]) {
            ASCEND_THROW_MSG("Index format is not correct.\n");
        }
    }

    // load Index Params
    // check if parameters match if there already exist parameters; otherwise check the validity of parameters
    ReadAndCheck(nList, ReadAndCheckType::nlist1, indexReader);
    ReadAndCheck(nListL2, ReadAndCheckType::nlist2, indexReader);
    ReadAndCheck(subSpaceDimL1, ReadAndCheckType::subdiml1, indexReader);
    ReadAndCheck(subSpaceDimL2, ReadAndCheckType::subdiml2, indexReader);
    ReadAndCheck(dim, ReadAndCheckType::dim, indexReader);
    ReadAndCheck(dimStored, ReadAndCheckType::subdiml1, indexReader);
    ReadAndCheck(ntotal, ReadAndCheckType::validRange, indexReader);
    ReadAndCheck(trained, ReadAndCheckType::False, indexReader);

    indexReader.ReadAndCheck((&metricType), sizeof(metricType));

    // resize vectors based on data
    codeWordsByBucket.resize(nList * nListL2, std::vector<uint8_t>());
    idxByBucket.resize(nList * nListL2, std::vector<int64_t>());
    normL2ByBucket.resize(nList * nListL2, std::vector<float>());

    // load SQ Params
    int vDiff1ListSize = 0;
    indexReader.ReadAndCheck((&vDiff1ListSize), sizeof(vDiff1ListSize));
    loadedValueSanityCheck(vDiff1ListSize, DIM_UB);

    std::vector<float16_t> vDiff1Cpu(vDiff1ListSize);
    indexReader.ReadAndCheck((vDiff1Cpu.data()), sizeof(float16_t) * vDiff1Cpu.size());
    int vDiff2ListSize = 0;
    indexReader.ReadAndCheck((&vDiff2ListSize), sizeof(vDiff2ListSize));
    loadedValueSanityCheck(vDiff2ListSize, DIM_UB);

    std::vector<float16_t> vDiff2Cpu(vDiff2ListSize);
    indexReader.ReadAndCheck((vDiff2Cpu.data()), sizeof(float16_t) * vDiff2Cpu.size());
    std::initializer_list<int> vDiffNpuDims = {vDiff1ListSize};
    vDiffNpu = std::make_shared<AscendTensor<float16_t, DIMS_1> >(vDiffNpuDims);
    auto ret = aclrtMemcpy(vDiffNpu->data(), vDiffNpu->getSizeInBytes(), vDiff1Cpu.data(),
                           sizeof(float16_t) * vDiff1Cpu.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_REQUIRE_OK(ret);
    std::initializer_list<int> vDiff2NpuDims = {vDiff2ListSize};
    vDiff2Npu = std::make_shared<AscendTensor<float16_t, DIMS_1> >(vDiff2NpuDims);
    ret = aclrtMemcpy(vDiff2Npu->data(), vDiff2Npu->getSizeInBytes(), vDiff2Cpu.data(),
                      sizeof(float16_t) * vDiff2Cpu.size(), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_REQUIRE_OK(ret);

    sqHandler = std::make_unique<faiss::ScalarQuantizer>(dimStored, faiss::ScalarQuantizer::QuantizerType::QT_8bit);
    sqHandler->trained.resize(dimStored * 2);
    indexReader.ReadAndCheck((sqHandler->trained.data()), sizeof(float) * sqHandler->trained.size());

    // load codebook
    std::vector<float> codeBooksL1CpuTmp(1L * nList * subSpaceDimL1 * dim);
    indexReader.ReadAndCheck((codeBooksL1CpuTmp.data()), sizeof(float) * codeBooksL1CpuTmp.size());

    std::vector<float> codeBooksL2CpuTmp(1L * nList * nListL2 * subSpaceDimL2 * subSpaceDimL1);
    indexReader.ReadAndCheck((codeBooksL2CpuTmp.data()), sizeof(float) * codeBooksL2CpuTmp.size());

    if (nullptr == loadedIndex) {
        AddCodeBooks(codeBooksL1CpuTmp, codeBooksL2CpuTmp);
    } else {
        AddCodeBooks(loadedIndex);
    }

    // load precomputeNormL2Npu data
    int precomputeListSize = 0;
    indexReader.ReadAndCheck((&precomputeListSize), sizeof(precomputeListSize));
    loadedValueSanityCheck(precomputeListSize, N_TOTAL_UB);

    std::vector<float16_t> precomputeList(precomputeListSize);
    indexReader.ReadAndCheck((precomputeList.data()), sizeof(float16_t) * precomputeList.size());
    precomputeNormL2Npu = std::make_unique<DeviceVector<float16_t> >(MemorySpace::DEVICE);
    precomputeNormL2Npu->append(precomputeList.data(), precomputeList.size(), true);

    // load maskByteNpu data
    size_t maskByteSize = precomputeNormL2Npu->size();
    std::vector<uint8_t> maskByte(maskByteSize);
    maskByteNpu = std::make_unique<DeviceVector<uint8_t> >(MemorySpace::DEVICE);
    maskByteNpu->append(maskByte.data(), maskByte.size(), true);
    maskByteCpu.resize(maskByteSize);

    // load isMaskOffset data
    int isMaskOffsetSize = nList * nListL2 * 2;
    std::vector<uint64_t> isMaskOffsetVec(isMaskOffsetSize, 0);
    isMaskOffset = std::make_unique<DeviceVector<uint64_t> >(MemorySpace::DEVICE);
    isMaskOffset->append(isMaskOffsetVec.data(), isMaskOffsetVec.size(), true);
    isMaskOffsetCpu.resize(isMaskOffsetSize, 0);

    // load ids data
    for (int i = 0; i < nList; i++) {
        int idsListSize = 0;
        indexReader.ReadAndCheck((&idsListSize), sizeof(idsListSize));
        loadedValueSanityCheck(idsListSize, N_TOTAL_UB);

        std::vector<int64_t> idsList(idsListSize);
        indexReader.ReadAndCheck((idsList.data()), sizeof(int64_t) * idsList.size());
        ids.emplace_back(std::make_unique<DeviceVector<int64_t> >(MemorySpace::DEVICE));
        ids[i]->append(idsList.data(), idsListSize, true);
        idsCPU.emplace_back(idsList);
    }

    for (int i = 0; i < nList; i++) {
        size_t baseListSize = 0;
        indexReader.ReadAndCheck((&baseListSize), sizeof(baseListSize));
        loadedValueSanityCheck(baseListSize, N_TOTAL_UB);

        std::vector<uint8_t> baseList(baseListSize);
        indexReader.ReadAndCheck((baseList.data()), sizeof(uint8_t) * baseList.size());
        baseShaped.emplace_back(std::make_unique<DeviceVector<unsigned char> >(MemorySpace::DEVICE));
        baseShaped[i]->append(baseList.data(), baseList.size(), true);
    }

    // load bucketOffsetInPage
    int listSize = 0;
    indexReader.ReadAndCheck((&listSize), sizeof(listSize));
    loadedValueSanityCheck(listSize, N_TOTAL_UB);

    bucketOffsetInPage.resize(listSize);
    indexReader.ReadAndCheck((bucketOffsetInPage.data()), sizeof(size_t) * bucketOffsetInPage.size());

    // check whether we've reached EOF (index before progressive add functionality does not store byBucket data)
    for (size_t i = 0; i < idxByBucket.size(); ++i) {
        int globalBucketSize = 0;
        indexReader.ReadAndCheck((&globalBucketSize), sizeof(globalBucketSize));
        loadedValueSanityCheck(globalBucketSize, N_TOTAL_UB);

        idxByBucket[i].resize(globalBucketSize);
        indexReader.ReadAndCheck((idxByBucket[i].data()), globalBucketSize * sizeof(int64_t));

        normL2ByBucket[i].resize(globalBucketSize);
        indexReader.ReadAndCheck((normL2ByBucket[i].data()), globalBucketSize * sizeof(float));

        codeWordsByBucket[i].resize(globalBucketSize * dimStored);
        indexReader.ReadAndCheck((codeWordsByBucket[i].data()), globalBucketSize * dimStored * sizeof(uint8_t));
    }

    indexReader.ReadAndCheck((&isAddWithIds), sizeof(isAddWithIds));

    int idMapSize = 0;
    indexReader.ReadAndCheck((&idMapSize), sizeof(idMapSize));
    loadedValueSanityCheck(idMapSize, N_TOTAL_UB);

    for (int i = 0; i < idMapSize; ++i) {
        int64_t realId = 0;
        int64_t virtualId = 0;
        indexReader.ReadAndCheck((&realId), sizeof(realId));
        indexReader.ReadAndCheck((&virtualId), sizeof(virtualId));
        idMap.insert(std::make_pair(realId, virtualId));
    }

    // 为了保证老版本索引可用性，如果uniqueBaseVecCounter无法被读取成功，不报错
    indexReader.ReadWithoutCheck((&uniqueBaseVecCounter), sizeof(uniqueBaseVecCounter));

    std::vector<uint64_t> paddingPageOffset(this->nList + 1);
    paddingPageOffset[0] = 0;
    for (int i = 0; i < nList; i++) {
        size_t pageSize = bucketOffsetInPage[i * nListL2 + nListL2 - 1];
        size_t paddingPageSize = static_cast<size_t>(utils::divUp(pageSize, BASE_SEG_SIZE)) * BASE_SEG_SIZE;
        paddingPageOffset[i + 1] = paddingPageOffset[i] + paddingPageSize;
    }

    std::initializer_list<int> addressOffsetOfBucketDims = { nList * nListL2 * 6 };
    addressOffsetOfBucket = std::make_shared<AscendTensor<uint64_t, DIMS_1> >(addressOffsetOfBucketDims);
    addressOffsetOfBucketCPU.resize(nList * nListL2 * 6);
    UpdateBucketAddressInfo(ntotal, paddingPageOffset);
    ACL_REQUIRE_OK(this->Init());
}

void NpuIndexIVFHSP::UpdateOpAttrs()
{
    (*searchL1OpAttrs)[static_cast<int>(TopkIvfSpL1AttrIdx::TOPK_L1_ATTR_ASC_IDX)] = 0;
    (*searchL1OpAttrs)[static_cast<int>(TopkIvfSpL1AttrIdx::TOPK_L1_ATTR_K_IDX)] = this->searchParam->nProbeL1;
    (*searchL1OpAttrs)[static_cast<int>(TopkIvfSpL1AttrIdx::TOPK_L1_ATTR_QUICK_HEAP)] = 0;

    (*searchL2OpAttrs)[static_cast<int>(TopkIvfSpL2AttrIdx::TOPK_L2_ATTR_ASC_IDX)] = 0;
    (*searchL2OpAttrs)[static_cast<int>(TopkIvfSpL2AttrIdx::TOPK_L2_ATTR_K_IDX)] = this->searchParam->nProbeL2;
    (*searchL2OpAttrs)[static_cast<int>(TopkIvfSpL2AttrIdx::TOPK_L2_ATTR_NLIST2_IDX)] = this->nListL2;
    (*searchL2OpAttrs)[static_cast<int>(TopkIvfSpL2AttrIdx::TOPK_L2_ATTR_SEG_SIZE_IDX)] = BASE_SEG_SIZE;
    (*searchL2OpAttrs)[static_cast<int>(TopkIvfSpL2AttrIdx::TOPK_L2_ATTR_BUCKET_NUM_IDX)] =
        static_cast<int>(this->nListL2 * nList);
    (*searchL2OpAttrs)[static_cast<int>(TopkIvfSpL2AttrIdx::TOPK_L2_ATTR_DIM_STORE_IDX)] = dimStored;
    (*searchL2OpAttrs)[static_cast<int>(TopkIvfSpL2AttrIdx::TOPK_L2_ATTR_SEG_NUM_IDX)] =
        this->searchParam->l3SegmentNum;

    (*searchMultiL2OpAttrs)[static_cast<int>(IvfMultiSpTopkL2AttrIdx::TOPK_L2_ATTR_ASC_IDX)] = 0;
    (*searchMultiL2OpAttrs)[static_cast<int>(IvfMultiSpTopkL2AttrIdx::TOPK_L2_ATTR_K_IDX)] =
        this->searchParam->nProbeL2;
    (*searchMultiL2OpAttrs)[static_cast<int>(IvfMultiSpTopkL2AttrIdx::TOPK_L2_ATTR_NLIST2_IDX)] = this->nListL2;

    (*searchL3OpAttrs)[static_cast<int>(TopkIvfSpL3AttrIdx::TOPK_L3_ATTR_ASC_IDX)] = 1;
    (*searchL3OpAttrs)[static_cast<int>(TopkIvfSpL3AttrIdx::TOPK_L3_ATTR_K_IDX)] = topKHSP;
    (*searchL3OpAttrs)[2] = BASE_SEG_SIZE;
    (*searchL3OpAttrs)[3] = this->searchParam->l3SegmentNum;
    (*searchL3OpAttrs)[4] = BASE_SEG_SIZE;
    (*searchL3OpAttrs)[5] = this->searchParam->nProbeL2;
    (*searchL3OpAttrs)[6] = 0;
}

/**
 * @brief 检索时，允许用户改变topK值，进而改变AICPU的topK值
 *
 */
void NpuIndexIVFHSP::SetTopKDuringSearch(int topKSearch)
{
    (*searchL3OpAttrs)[static_cast<int>(TopkIvfSpL3AttrIdx::TOPK_L3_ATTR_K_IDX)] = topKSearch;
    topKHSP = topKSearch;
}

/**
 *
 * @param codeBooksL1 nlistL1 * subSpaceDimL1 * dim
 * @param codeBooksL2 nlistL1 * nlistL2 * subSpaceDimL2 * subSpaceDimL1
 * @return
 */
APP_ERROR NpuIndexIVFHSP::AddCodeBooks(const std::vector<float> &codeBooksL1, const std::vector<float> &codeBooksL2)
{
    ASCEND_THROW_IF_NOT_FMT((static_cast<size_t>(nList) * static_cast<size_t>(subSpaceDimL1) * static_cast<size_t>(dim)) == codeBooksL1.size(),
                            "the size of codeBooksL1 is incorrect, require:%zu.",
                            static_cast<size_t>(nList) * static_cast<size_t>(subSpaceDimL1) * static_cast<size_t>(dim));
    ASCEND_THROW_IF_NOT_FMT((static_cast<size_t>(nList) * static_cast<size_t>(nListL2) * static_cast<size_t>(subSpaceDimL2) * static_cast<size_t>(subSpaceDimL1)) ==
                                codeBooksL2.size(),
                            "the size of codeBooksL2 is incorrect, require:%zu.",
                            static_cast<size_t>(nList) * static_cast<size_t>(nListL2) * static_cast<size_t>(subSpaceDimL2) * static_cast<size_t>(subSpaceDimL1));
 
    int n1 = nList * subSpaceDimL1;

    // Calling AddCodeBooks through vector parameters, thus allocating memory on NPU
    std::initializer_list<int> codeBooksShapedL1NpuDims = {
        utils::divUp(n1, CUBE_ALIGN), utils::divUp(this->dim, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN };
    codeBooksShapedL1Npu = std::make_shared<AscendTensor<float16_t, DIMS_4> >(codeBooksShapedL1NpuDims);

    std::initializer_list<int> codeBooksShapedL2NpuDims = { utils::divUp(nList * nListL2 * subSpaceDimL2, CUBE_ALIGN),
                                                            utils::divUp(subSpaceDimL1, CUBE_ALIGN),
                                                            CUBE_ALIGN, CUBE_ALIGN };
    codeBooksShapedL2Npu = std::make_shared<AscendTensor<float16_t, DIMS_4> >(codeBooksShapedL2NpuDims);

    // / Add  CodeBooks To CPU Memory
    this->codeBooksL1Cpu.assign(codeBooksL1.begin(), codeBooksL1.end());
    this->codeBooksL2Cpu.assign(codeBooksL2.begin(), codeBooksL2.end());

    // / Add CodeBooks To Npu Memory
    auto &mem = resources->getMemoryManager();

    std::vector<float16_t> codeBooksL1Tmp(codeBooksL1.size());

    std::transform(codeBooksL1.begin(), codeBooksL1.end(), codeBooksL1Tmp.begin(), [](float value) -> float16_t {
        FormatConvert convert;
        convert.value = fp16(value).data;
        return convert.data;
    });

    std::vector<float16_t> codeBooksL1CpuShaped(codeBooksShapedL1Npu->getSizeInBytes() / sizeof(float16_t));
    ZzFormatReshape(codeBooksL1Tmp, codeBooksL1CpuShaped, static_cast<size_t>(nList) * subSpaceDimL1, dim,
                    codeBooksShapedL1Npu->getSize(2), codeBooksShapedL1Npu->getSize(3));
    auto ret = aclrtMemcpy(codeBooksShapedL1Npu->data(), codeBooksShapedL1Npu->getSizeInBytes(),
                           codeBooksL1CpuShaped.data(), codeBooksL1CpuShaped.size() * sizeof(float16_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
    APP_LOG_INFO("CodeBookL1 Add Finished.");

    int n2 = codeBooksShapedL2Npu->getSize(0) * codeBooksShapedL2Npu->getSize(2);  // nlist * nlistL2 * subSpaceDimL2

    AscendTensor<float16_t, DIMS_2> codeBooksL2Fp16Npu(mem, { n2, subSpaceDimL1 },
                                                       defaultStream);  // assert n2 % CUBE_ALIGN == 0

    std::vector<float16_t> codeBooksL2Tmp(codeBooksL2.size());
    std::transform(codeBooksL2.begin(), codeBooksL2.end(), codeBooksL2Tmp.begin(), [](float value) -> float16_t {
        FormatConvert convert;
        convert.value = fp16(value).data;
        return convert.data;
    });
    std::vector<float16_t> codeBooksL2CpuShaped(codeBooksShapedL2Npu->getSizeInBytes() / sizeof(float16_t));
    ZzFormatReshape(codeBooksL2Tmp, codeBooksL2CpuShaped, static_cast<size_t>(nList) * nListL2 * subSpaceDimL2,
                    subSpaceDimL1, codeBooksShapedL2Npu->getSize(2), codeBooksShapedL2Npu->getSize(3));

    ret = aclrtMemcpy(codeBooksShapedL2Npu->data(), codeBooksShapedL2Npu->getSizeInBytes(), codeBooksL2CpuShaped.data(),
                      codeBooksL2CpuShaped.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
    APP_LOG_INFO("CodeBookL2 Add Finished.");
    return APP_ERR_OK;
}

/**
 * @brief iN MultiIndex Scenario, allow pointers to AscendTensors to another index's codebook
 *
 * @param codeBooksL1Shared
 * @return APP_ERROR
 */
APP_ERROR NpuIndexIVFHSP::AddCodeBooks(const NpuIndexIVFHSP *loadedIndex)
{
    ASCEND_THROW_IF_NOT_MSG(loadedIndex != nullptr, "Empty pointer to loaded Index\n");
    ASCEND_THROW_IF_NOT_MSG(loadedIndex->GetCodeBooksL1NPU() != nullptr,
                            "Empty pointer to loaded Index's L1 codebook on NPU.\n");
    ASCEND_THROW_IF_NOT_MSG(loadedIndex->GetCodeBooksL2NPU() != nullptr,
                            "Empty pointer to loaded Index's L2 codebook on NPU.\n");

    ASCEND_THROW_IF_NOT_FMT((static_cast<size_t>(nList) * static_cast<size_t>(subSpaceDimL1) * static_cast<size_t>(dim)) ==
                                (loadedIndex->GetCodeBooksL1NPU()->getSizeInBytes() / sizeof(float16_t)),
                            "the size of codeBooksL1 is incorrect, require:%zu.",
                            static_cast<size_t>(nList) * static_cast<size_t>(subSpaceDimL1) * static_cast<size_t>(dim));
    ASCEND_THROW_IF_NOT_FMT((static_cast<size_t>(nList) * static_cast<size_t>(nListL2) * static_cast<size_t>(subSpaceDimL2) * static_cast<size_t>(subSpaceDimL1)) ==
                                (loadedIndex->GetCodeBooksL2NPU()->getSizeInBytes() / sizeof(float16_t)),
                            "the size of codeBooksL2 is incorrect, require:%zu.",
                            static_cast<size_t>(nList) * static_cast<size_t>(nListL2) * static_cast<size_t>(subSpaceDimL2) * static_cast<size_t>(subSpaceDimL1));

    ASCEND_THROW_IF_NOT_MSG(loadedIndex->GetCodeBooksL1CPU() != nullptr,
                            "Empty pointer to loaded Index's L1 codebook on CPU.\n");
    ASCEND_THROW_IF_NOT_MSG(loadedIndex->GetCodeBooksL2CPU() != nullptr,
                            "Empty pointer to loaded Index's L2 codebook on CPU.\n");

    codeBooksL1Cpu.assign(loadedIndex->GetCodeBooksL1CPU(),
                          loadedIndex->GetCodeBooksL1CPU() + (static_cast<size_t>(nList) * subSpaceDimL1 * dim));
    codeBooksL2Cpu.assign(loadedIndex->GetCodeBooksL2CPU(),
                          loadedIndex->GetCodeBooksL2CPU() +
                          (static_cast<size_t>(nList) * nListL2 * subSpaceDimL2 * subSpaceDimL1));

    codeBooksShapedL1Npu = std::const_pointer_cast<AscendTensor<float16_t, DIMS_4> >(loadedIndex->GetCodeBooksL1NPU());
    codeBooksShapedL2Npu = std::const_pointer_cast<AscendTensor<float16_t, DIMS_4> >(loadedIndex->GetCodeBooksL2NPU());

    return APP_ERR_OK;
}

// step1 按1级粗桶分页
// 基于大页内存存储
// 先基于baseRawData获取当前全量数据的一个划分，存储再临时变量中，然后再结合已有的数据进行重新划分，以支持增量增加的功能。
/**
 * step1 按1级粗桶分页
 * @param baseRawData
 * @return
 */

APP_ERROR NpuIndexIVFHSP::AddVectorsVerbose(const std::vector<float> &baseRawData, bool verbose)
{
    if (aclrtSetDevice(ascendConfig.deviceList[0]) != ACL_SUCCESS) {
        APP_LOG_ERROR("Set device failed. deviceId is %d", ascendConfig.deviceList[0]);
        return ACL_ERROR_FAILURE;
    }
    auto nb = baseRawData.size() / static_cast<size_t>(this->dim);
    ASCEND_THROW_IF_NOT(nb * static_cast<size_t>(this->dim) == baseRawData.size());
    APP_LOG_INFO("NpuIndexIVFHSP::AddVectors Operation. n = %d, dim=%d, dimStored=%d.\n", nb, this->dim,
                 this->dimStored);
    if (nb == 0)
        return APP_ERR_OK;

    /* *get indices incrementally from ntotal */
    ASCEND_THROW_IF_NOT_FMT(uniqueBaseVecCounter <= UINT64_MAX - nb,
        "uniqueBaseVecCounter[%llu] should be <= UINT64_MAX - nb[%llu].\n", uniqueBaseVecCounter, nb);
    std::vector<int64_t> indices(nb);
    std::iota(indices.begin(), indices.end(), uniqueBaseVecCounter);
    uniqueBaseVecCounter += nb;

    // / Get L2 Bucket Id of each base vector
    std::vector<int64_t> assign(nb, -1);
    std::vector<float> codeWords(nb * static_cast<size_t>(dimStored));
    std::vector<float> precomputeNormL2(nb);
    GetVectorsAssignNPU(nb, const_cast<float *>(baseRawData.data()), assign.data(), codeWords.data(),
                        precomputeNormL2.data(), 1, verbose);

    // / ScalaQuantization
    std::vector<uint8_t> codeWordSQ(nb * static_cast<size_t>(this->dimStored));
    ComputeSQCode(nb, codeWords, codeWordSQ);

    // / resort: move base vectors together by the assign information
    for (size_t k = 0; k < nb; k++) {
        codeWordsByBucket[assign[k]].insert(codeWordsByBucket[assign[k]].end(), codeWordSQ.data() + k * dimStored,
                                            codeWordSQ.data() + (k + 1) * dimStored);
        idxByBucket[assign[k]].emplace_back(indices[k]);
        normL2ByBucket[assign[k]].emplace_back(precomputeNormL2[k]);
    }

    // / Add to Index Data Store
    AddIntoNpuDataStore(nb, codeWordsByBucket, idxByBucket, normL2ByBucket);
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::Train(const std::vector<float> &data)
{
    // / 在cpu侧train
    sqHandler->train(data.size() / dimStored, data.data());
    // / train完成将量化参数更新到Npu侧
    std::vector<float> trained(this->dimStored * 2);
    float *vmin = trained.data();
    float *vdiff = trained.data() + this->dimStored;
    switch (sqHandler->qtype) {
        case faiss::ScalarQuantizer::QuantizerType::QT_8bit:
            trained.assign(sqHandler->trained.begin(), sqHandler->trained.end());
            break;
        case faiss::ScalarQuantizer::QuantizerType::QT_8bit_uniform:
            for (int i = 0; i < this->dimStored; i++) {
                *(vmin + i) = sqHandler->trained[0];
                *(vdiff + i) = sqHandler->trained[1];
            }
            break;
        default:
            ASCEND_THROW_FMT("Not Supported qtype(%d).", sqHandler->qtype);
            break;
    }
    for (int i = 0; i < dimStored; i++) {
        trained[i] = vmin[i] * 255.0 + vdiff[i] * 0.5;
    }
    std::transform(trained.begin(), trained.end(), trained.begin(), [](float x) { return -2.0 * x / 255.0; });
    std::vector<float16_t> trainedFP16(this->dimStored * 2);
    std::transform(trained.begin(), trained.end(), trainedFP16.begin(), [](float v) {
        auto convert = FormatConvert();
        convert.value = fp16(v).data;
        return convert.data;
    });
    auto ret = aclrtMemcpy(vDiff2Npu->data(), vDiff2Npu->getSizeInBytes(), trainedFP16.data(),
                           this->dimStored * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy vDiff2 to device failed %d", static_cast<int>(ret));
    ret = aclrtMemcpy(vDiffNpu->data(), vDiffNpu->getSizeInBytes(), trainedFP16.data() + dimStored,
                      this->dimStored * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy vDiff to device failed %d", static_cast<int>(ret));
    return APP_ERR_OK;
}

void NpuIndexIVFHSP::ComputeSQCode(size_t n, const std::vector<float> &data, std::vector<uint8_t> &sqCodes)
{
    if (!this->trained) {
        Train(data);
    }
    sqHandler->compute_codes(data.data(), sqCodes.data(), n);
}

bool NpuIndexIVFHSP::CalculateOffsetL3(const std::vector<NpuIndexIVFHSP *> &indexes, int n, int i,
                                       std::vector<uint64_t> &labelL2Cpu,
                                       std::vector<uint64_t> &outOffset,  // n * nprobel2 * 5
                                       std::vector<uint64_t> &outIdsOffset)
{
    // 3. L3 Search
    auto baseAddressOffsetTmp = indexes[i]->addressOffsetOfBucketCPU.data();
    auto normL2OffsetTmp = indexes[i]->addressOffsetOfBucketCPU.data() + nList * nListL2 * 2;
    auto idOffsetTmp = indexes[i]->addressOffsetOfBucketCPU.data() + nList * nListL2 * 4;

    auto *vt = &(vts->at(i));
    for (int j = 0; j < n; j++) {
        auto labelsL2 = labelL2Cpu.data() + j * this->searchParam->nProbeL2;
        auto outsumSegmentNum = outOffset.data() + j * this->searchParam->nProbeL2 * 5;
        auto outBaseOffset = outOffset.data() + j * this->searchParam->nProbeL2 * 5 + this->searchParam->nProbeL2;
        auto outNormL2Offset = outOffset.data() + j * this->searchParam->nProbeL2 * 5 + this->searchParam->nProbeL2 * 3;
        auto outIdOffset = outIdsOffset.data() + j * this->searchParam->nProbeL2 * 2;

        // get selected buckets's Offset into outOffset
        uint64_t minNormL2Offset = normL2OffsetTmp[labelsL2[0] * 2];
        uint64_t maxNormL2Offset = normL2OffsetTmp[labelsL2[0] * 2 + 1];
        for (int i = 0; i < this->searchParam->nProbeL2; ++i) {
            outNormL2Offset[i * 2] = normL2OffsetTmp[labelsL2[i] * 2];
            outNormL2Offset[i * 2 + 1] = normL2OffsetTmp[labelsL2[i] * 2 + 1];
            minNormL2Offset = std::min(minNormL2Offset, outNormL2Offset[i * 2]);
            maxNormL2Offset = std::max(maxNormL2Offset, outNormL2Offset[i * 2 + 1]);
        }

        vt->advance();
        int segCnt = 0;
        for (int k = 0; k < this->searchParam->nProbeL2; k++) {
            auto globalIdx = labelsL2[k];
            auto startOffset = outNormL2Offset[k * 2];
            auto endOffset = outNormL2Offset[k * 2 + 1];
            auto startIdx = (startOffset - minNormL2Offset) / BASE_SEG_SIZE;
            auto endIdx = (endOffset - minNormL2Offset) / BASE_SEG_SIZE;
            uint64_t actualK0 = startIdx;
            uint64_t actualK1 = startIdx;
            bool flag = false;
            for (auto m = startIdx; m < endIdx; m++) {
                if (segCnt == this->searchParam->l3SegmentNum) {
                    break;
                }
                if (!vt->get(m)) {
                    // 如果当前segment没有被占领
                    vt->set(m);
                    segCnt++;
                    flag = true;
                    actualK1 = m + 1;
                } else {
                    // 如果当前segment被占领
                    if (!flag) {
                        actualK0++;
                    } else {
                        actualK1 = m;
                        break;
                    }
                }
            }
            if (actualK0 > actualK1) {
                actualK1 = actualK0;
            }

            outBaseOffset[k * 2] = (actualK0 - startIdx) * static_cast<uint64_t>(BASE_SEG_SIZE) * static_cast<uint64_t>(dimStored)+
                                   baseAddressOffsetTmp[globalIdx * 2];
            outBaseOffset[k * 2 + 1] = (actualK1 - startIdx) * static_cast<uint64_t>(BASE_SEG_SIZE) * static_cast<uint64_t>(dimStored) +
                                       baseAddressOffsetTmp[globalIdx * 2];
            outNormL2Offset[k * 2] = actualK0 * BASE_SEG_SIZE + minNormL2Offset;
            outNormL2Offset[k * 2 + 1] = actualK1 * BASE_SEG_SIZE + minNormL2Offset;

            outIdOffset[k * 2] = (actualK0 - startIdx) * BASE_SEG_SIZE * sizeof(int64_t) + idOffsetTmp[globalIdx * 2];
            outIdOffset[k * 2 + 1] = (actualK1 - startIdx) * BASE_SEG_SIZE * sizeof(int64_t) +
                                     idOffsetTmp[globalIdx * 2];
            if (outIdOffset[k * 2] > idOffsetTmp[globalIdx * 2 + 1]) {
                outIdOffset[k * 2] = idOffsetTmp[globalIdx * 2 + 1];
            }
            if (outIdOffset[k * 2 + 1] > idOffsetTmp[globalIdx * 2 + 1]) {
                outIdOffset[k * 2 + 1] = idOffsetTmp[globalIdx * 2 + 1];
            }
        }
        outsumSegmentNum[0] = (outNormL2Offset[1] - outNormL2Offset[0]) / BASE_SEG_SIZE;
        for (int q = 1; q < this->searchParam->nProbeL2; ++q) {
            outsumSegmentNum[q] = (outNormL2Offset[q * 2 + 1] - outNormL2Offset[q * 2]) / BASE_SEG_SIZE +
                                  outsumSegmentNum[q - 1];
        }
    }
    return true;
}

bool NpuIndexIVFHSP::CalculateOffsetL3WithMask(const std::vector<NpuIndexIVFHSP *> &indexes, int n, const uint8_t *mask,
                                               int i, std::vector<uint64_t> &labelL2Cpu,
                                               std::vector<uint64_t> &outOffset,  // n * nprobel2 * 5
                                               std::vector<uint64_t> &outIdsOffset)
{
    // 3. L3 Search
    auto baseAddressOffsetTmp = indexes[i]->addressOffsetOfBucketCPU.data();
    auto normL2OffsetTmp = indexes[i]->addressOffsetOfBucketCPU.data() + nList * nListL2 * 2;
    auto idOffsetTmp = indexes[i]->addressOffsetOfBucketCPU.data() + nList * nListL2 * 4;
    auto idsCpuAddressOfBucketVecCurr = indexes[i]->idsCpuAddressOfBucketVec.data();

    auto *vt = &(vts->at(i));
    for (int j = 0; j < n; j++) {
        auto labelsL2 = labelL2Cpu.data() + j * this->searchParam->nProbeL2;
        auto outsumSegmentNum = outOffset.data() + j * this->searchParam->nProbeL2 * 5;
        auto outBaseOffset = outOffset.data() + j * this->searchParam->nProbeL2 * 5 + this->searchParam->nProbeL2;
        auto outNormL2Offset = outOffset.data() + j * this->searchParam->nProbeL2 * 5 + this->searchParam->nProbeL2 * 3;
        auto outIdOffset = outIdsOffset.data() + j * this->searchParam->nProbeL2 * 2;
 
        // get selected buckets's Offset into outOffset
        uint64_t minNormL2Offset = normL2OffsetTmp[labelsL2[0] * 2];
        uint64_t maxNormL2Offset = normL2OffsetTmp[labelsL2[0] * 2 + 1];
        for (int m = 0; m < this->searchParam->nProbeL2; ++m) {
            outNormL2Offset[m * 2] = normL2OffsetTmp[labelsL2[m] * 2];
            outNormL2Offset[m * 2 + 1] = normL2OffsetTmp[labelsL2[m] * 2 + 1];
            minNormL2Offset = std::min(minNormL2Offset, outNormL2Offset[m * 2]);
            maxNormL2Offset = std::max(maxNormL2Offset, outNormL2Offset[m * 2 + 1]);
        }
 
        vt->advance();
        int segCnt = 0;
        for (int k = 0; k < this->searchParam->nProbeL2; k++) {
            auto globalIdx = labelsL2[k];
            auto startOffset = outNormL2Offset[k * 2];
            auto endOffset = outNormL2Offset[k * 2 + 1];
            auto startIdx = (startOffset - minNormL2Offset) / BASE_SEG_SIZE;
            auto endIdx = (endOffset - minNormL2Offset) / BASE_SEG_SIZE;
            uint64_t actualK0 = startIdx;
            uint64_t actualK1 = startIdx;
            bool flag = false;
            for (auto m = startIdx; m < endIdx; m++) {
                if (segCnt == this->searchParam->l3SegmentNum) {
                    break;
                }
                if (!vt->get(m)) {
                    // 如果当前segment没有被占领
                    vt->set(m);
                    segCnt++;
                    flag = true;
                    actualK1 = m + 1;
                } else {
                    // 如果当前segment被占领
                    if (!flag) {
                        actualK0++;
                    } else {
                        actualK1 = m;
                        break;
                    }
                }
            }
            if (actualK0 > actualK1) {
                actualK1 = actualK0;
            }
 
            outBaseOffset[k * 2] =(actualK0 - startIdx) * static_cast<uint64_t>(BASE_SEG_SIZE) * static_cast<uint64_t>(dimStored) +
                                   baseAddressOffsetTmp[globalIdx * 2];
            outBaseOffset[k * 2 + 1] = (actualK1 - startIdx) * static_cast<uint64_t>(BASE_SEG_SIZE) * static_cast<uint64_t>(dimStored) +
                                       baseAddressOffsetTmp[globalIdx * 2];
            outNormL2Offset[k * 2] = actualK0 * BASE_SEG_SIZE + minNormL2Offset;
            outNormL2Offset[k * 2 + 1] = actualK1 * BASE_SEG_SIZE + minNormL2Offset;
 
            outIdOffset[k * 2] = (actualK0 - startIdx) * BASE_SEG_SIZE * sizeof(int64_t) + idOffsetTmp[globalIdx * 2];
            outIdOffset[k * 2 + 1] = (actualK1 - startIdx) * BASE_SEG_SIZE * sizeof(int64_t) +
                                     idOffsetTmp[globalIdx * 2];
            if (outIdOffset[k * 2] > idOffsetTmp[globalIdx * 2 + 1]) {
                outIdOffset[k * 2] = idOffsetTmp[globalIdx * 2 + 1];
            }
            if (outIdOffset[k * 2 + 1] > idOffsetTmp[globalIdx * 2 + 1]) {
                outIdOffset[k * 2 + 1] = idOffsetTmp[globalIdx * 2 + 1];
            }
            {
                uint64_t start = outIdOffset[2 * k];
                uint64_t end = outIdOffset[2 * k + 1];
                uint64_t offset = (end - start) / sizeof(uint64_t);
                std::vector<uint64_t> realIds(offset, 0);
                int64_t *idStart = static_cast<int64_t* >(reinterpret_cast<void *>(
                    start - idOffsetTmp[globalIdx * 2] + idsCpuAddressOfBucketVecCurr[globalIdx * 2]));

                if (offset > 0) {
                    auto ret = memcpy_s(realIds.data(), sizeof(int64_t) * offset, idStart, sizeof(int64_t) * offset);
                    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
                    for (uint64_t m = 0; m < offset; m++) {
                        uint64_t realIdX = realIds[m] / 8;
                        uint64_t realIdY = realIds[m] % 8;
                        if (mask[realIdX] & (1 << realIdY)) {
                            indexes[i]->maskByteCpu[outNormL2Offset[k * 2] + m] = 1;
                        } else {
                            indexes[i]->maskByteCpu[outNormL2Offset[k * 2] + m] = 0;
                        }
                    }
                }
            }
        }
        outsumSegmentNum[0] = (outNormL2Offset[1] - outNormL2Offset[0]) / BASE_SEG_SIZE;
        for (int q = 1; q < this->searchParam->nProbeL2; ++q) {
            outsumSegmentNum[q] = (outNormL2Offset[q * 2 + 1] - outNormL2Offset[q * 2]) / BASE_SEG_SIZE +
                                  outsumSegmentNum[q - 1];
        }
    }
    return true;
}

/**
 * Add Reordered Data into NPUStore(baseShaped)
 * @param num
 * @param baseCodesByBucket
 * @param idsByBucket
 * @return
 */
APP_ERROR NpuIndexIVFHSP::AddIntoNpuDataStore(size_t num, std::vector<std::vector<uint8_t> > &baseCodesByBucket,
                                              std::vector<std::vector<int64_t> > &idsByBucket,
                                              std::vector<std::vector<float> > &normL2ByBucket)
{
    auto stream = resources->getDefaultStream();
    auto &mem = resources->getMemoryManager();

    // remove existing content
    if (!baseShaped.empty()) {
        baseShaped.clear();
    }
    if (!ids.empty()) {
        ids.clear();
        idsCPU.clear();
    }
    precomputeNormL2Npu.reset();

    if (baseShaped.empty()) {
        auto AddFunc = [&stream, &mem, this](int currPageId, int packCnt, std::vector<uint8_t> &pageDataTmp) {
            // padding, align with BASE_SEG_SIZE
            auto newPackCnt = static_cast<size_t>(utils::divUp(packCnt, BASE_SEG_SIZE)) * BASE_SEG_SIZE;
            std::vector<uint8_t> pageDataShapedCpu(newPackCnt * static_cast<size_t>(this->dimStored));
            ZzFormatReshape(pageDataTmp, pageDataShapedCpu, packCnt, this->dimStored, CUBE_ALIGN,
                            CUBE_ALIGN);  // convert pageData into Zzformat and store it into pageDataShapedCpu
            baseShaped.emplace_back(std::make_unique<DeviceVector<unsigned char> >(MemorySpace::DEVICE));
            baseShaped[currPageId]->resize(pageDataShapedCpu.size(), true);  // resize device vector to the size of
                                                                             // pageDataShapedCpu
            if (newPackCnt > 0) {
                auto codeDataNpu =
                    AscendTensor<uint8_t, DIMS_4>(static_cast<uint8_t *>(baseShaped[currPageId]->data()),
                                                  { utils::divUp(static_cast<int>(newPackCnt), CUBE_ALIGN),
                                                    utils::divUp(this->dimStored, CUBE_ALIGN), CUBE_ALIGN,
                                                    CUBE_ALIGN });
                auto ret = aclrtMemcpy(codeDataNpu.data(), codeDataNpu.getSizeInBytes(), pageDataShapedCpu.data(),
                                       pageDataShapedCpu.size() * sizeof(uint8_t), ACL_MEMCPY_HOST_TO_DEVICE);
                ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy pageDataTmp to device failed %d",
                                        static_cast<int>(ret));
            } else {
                APP_LOG_WARNING("currPageId(%d) is empty!", currPageId);
            }
        };

        std::vector<uint8_t> pageDataTmp;
        std::vector<int64_t> pageIdsTmp;
        std::vector<float16_t> normL2Tmp;
        std::vector<float16_t> normL2All;
        size_t pageSize = 0;
        std::vector<uint64_t> paddingPageOffset(this->nList + 1);
        paddingPageOffset[0] = 0;
        for (size_t i = 0; i < idsByBucket.size(); i++) {
            auto bucketSize = idsByBucket[i].size();  // size of the L2 bucket
            if (bucketSize > 0) {
                pageDataTmp.insert(pageDataTmp.end(), baseCodesByBucket[i].begin(), baseCodesByBucket[i].end());
                pageIdsTmp.insert(pageIdsTmp.end(), idsByBucket[i].begin(), idsByBucket[i].end());
                normL2Tmp.insert(normL2Tmp.end(), normL2ByBucket[i].begin(), normL2ByBucket[i].end());
                pageSize += bucketSize;
            }
            bucketOffsetInPage[i] = pageSize;  // Within a page (L1 bucket), the offset of buckets (therefore L2 bucket)
            auto idxL1 = static_cast<int>(i / static_cast<size_t>(nListL2));
            auto idxL2 =  static_cast<int>(i % static_cast<size_t>(nListL2));
            if (idxL2 == nListL2 - 1) {  // if we are getting to the last L2 bucket of the L1 bucket
                // add into npu page
                AddFunc(idxL1, pageSize, pageDataTmp);  // add into page with id == idx1 all the Zz formated codeword
                                                        // data that belong to it

                ids.emplace_back(std::make_unique<DeviceVector<int64_t> >(MemorySpace::DEVICE));
                ids[idxL1]->append(pageIdsTmp.data(), pageSize, true);  // add into page with id == idx1 all the base
                                                                        // vector ids that belong to it
                idsCPU.emplace_back(pageIdsTmp);
                size_t paddingPageSize = static_cast<size_t>(utils::divUp(pageSize, BASE_SEG_SIZE)) * BASE_SEG_SIZE;
                normL2Tmp.resize(paddingPageSize);
                normL2All.insert(normL2All.end(), normL2Tmp.begin(), normL2Tmp.end());
                paddingPageOffset[idxL1 + 1] = paddingPageOffset[idxL1] + paddingPageSize;
                pageSize = 0;
                pageDataTmp.resize(0);
                pageIdsTmp.resize(0);
                normL2Tmp.resize(0);
            }
        }
        precomputeNormL2Npu = std::make_unique<DeviceVector<float16_t> >(MemorySpace::DEVICE);
        precomputeNormL2Npu->append(normL2All.data(), normL2All.size(), true);

        // assign size to maskByteNpu based on precomputeNormL2Npu
        size_t maskByteSize = precomputeNormL2Npu->size();
        std::vector<uint8_t> maskByte(maskByteSize);
        maskByteNpu = std::make_unique<DeviceVector<uint8_t>>(MemorySpace::DEVICE);
        maskByteNpu->append(maskByte.data(), maskByte.size(), true);
        maskByteCpu.resize(maskByteSize);

        int isMaskOffsetSize = nList * nListL2 * 2;
        std::vector<uint64_t> isMaskOffsetVec(isMaskOffsetSize, 0);
        isMaskOffset = std::make_unique<DeviceVector<uint64_t>>(MemorySpace::DEVICE);
        isMaskOffset->append(isMaskOffsetVec.data(), isMaskOffsetVec.size(), true);
        isMaskOffsetCpu.resize(isMaskOffsetSize, 0);

        UpdateBucketAddressInfo(num, paddingPageOffset);
    } else {
        printf("Not possible!\n");
    }

    ntotal += num;

    return APP_ERR_OK;
}

/**
 * 计算二级分桶分段数，并更新二级分桶的起始偏移位置和终止偏移位置（单位:BASE_SEG_SIZE）
 * @param num
 * @param idsByBucket
 */
void NpuIndexIVFHSP::UpdateBucketAddressInfo(size_t, std::vector<uint64_t> &paddingPageOffset)
{
    // reset addressOffSetOfBucket
    addressOffsetOfBucket.reset();
    std::initializer_list<int> addressOffsetOfBucketDims = { nList * nListL2 * 6 };
    addressOffsetOfBucket = std::make_shared<AscendTensor<uint64_t, DIMS_1> >(addressOffsetOfBucketDims);

    minAddressOfBaseNpu = baseShaped[0]->data();
    for (size_t i = 1; i < baseShaped.size(); i++) {
        minAddressOfBaseNpu = std::min(minAddressOfBaseNpu, baseShaped[i]->data());
    }  // baseShaped stores the quantized codewords for all vectors within a L1 bucket; minAddressOfBaseNpu stores the
       // address of the pointer with the smallest address

    size_t bucketNum = static_cast<size_t>(nList * nListL2);
    std::vector<uint64_t> baseAddressOffsetOfBucketVec(bucketNum * 2);    // start offset and end offset
    std::vector<uint64_t> normL2AddressOffsetOfBucketVec(bucketNum * 2);  // start offset and end offset
    std::vector<uint64_t> idsAddressOfBucketVec(bucketNum * 2);           // start offset and end offset
    idsCpuAddressOfBucketVec.resize(bucketNum * 2);

    for (size_t lstId = 0; lstId < bucketNum; lstId++) {
        // 计算二级分桶内的分段数
        auto idxL1 = static_cast<int>(lstId / static_cast<size_t>(nListL2));
        auto idxL2 = static_cast<int>(lstId % static_cast<size_t>(nListL2));

        bool bucketSizeFlag = true;
        // 对桶是否为空进行校验
        if (bucketOffsetInPage[lstId] == 0) {
            bucketSizeFlag = false;
        } else if (lstId > 0 && bucketOffsetInPage[lstId - 1] == bucketOffsetInPage[lstId]) {
            bucketSizeFlag = false;
        }
        uint64_t startSegOffset = 0;
        if (idxL2 > 0) {
            startSegOffset = bucketOffsetInPage[lstId - 1] / static_cast<size_t>(BASE_SEG_SIZE);  // how many buckets are within idxL1 bucket
                                                                             // before current idxL2 bucket
        }
        uint64_t endSegOffset = startSegOffset;
        if (bucketSizeFlag) {
            endSegOffset = utils::divUp(bucketOffsetInPage[lstId],
                                        static_cast<size_t>(BASE_SEG_SIZE));  // this ensures that the interval between startSegOffset and
                                                         // endSegOffset times 64 can store all L2 buckets of idxL2
        }

        baseAddressOffsetOfBucketVec[lstId * 2] = baseShaped[idxL1]->data() - minAddressOfBaseNpu +
                                                  startSegOffset * static_cast<uint64_t>(BASE_SEG_SIZE) * static_cast<uint64_t>(dimStored);
        baseAddressOffsetOfBucketVec[lstId * 2 + 1] = baseShaped[idxL1]->data() - minAddressOfBaseNpu +
                                                      endSegOffset * static_cast<uint64_t>(BASE_SEG_SIZE) * static_cast<uint64_t>(dimStored);

        idsAddressOfBucketVec[lstId * 2] =
            reinterpret_cast<uint64_t>(ids[idxL1]->data() + startSegOffset * static_cast<uint64_t>(BASE_SEG_SIZE));
        idsCpuAddressOfBucketVec[lstId * 2] = reinterpret_cast<uint64_t>(
            idsCPU[idxL1].data() + startSegOffset * static_cast<uint64_t>(BASE_SEG_SIZE));

        normL2AddressOffsetOfBucketVec[lstId * 2] = paddingPageOffset[idxL1] + startSegOffset * static_cast<uint64_t>(BASE_SEG_SIZE);
        normL2AddressOffsetOfBucketVec[lstId * 2 + 1] = paddingPageOffset[idxL1] + endSegOffset * static_cast<uint64_t>(BASE_SEG_SIZE);

        if (endSegOffset * static_cast<uint64_t>(BASE_SEG_SIZE) > bucketOffsetInPage[idxL1 * nListL2 + nListL2 - 1]) {
            idsAddressOfBucketVec[lstId * 2 + 1] =
                reinterpret_cast<uint64_t>(ids[idxL1]->data() + bucketOffsetInPage[lstId]);
            idsCpuAddressOfBucketVec[lstId * 2 + 1] =
                reinterpret_cast<uint64_t>(idsCPU[idxL1].data() + bucketOffsetInPage[lstId]);
        } else {
            idsAddressOfBucketVec[lstId * 2 + 1] =
                reinterpret_cast<uint64_t>(ids[idxL1]->data() + endSegOffset * static_cast<uint64_t>(BASE_SEG_SIZE));
            idsCpuAddressOfBucketVec[lstId * 2 + 1] =
                reinterpret_cast<uint64_t>(idsCPU[idxL1].data() + endSegOffset * static_cast<uint64_t>(BASE_SEG_SIZE));
        }
    }

    std::copy(baseAddressOffsetOfBucketVec.begin(), baseAddressOffsetOfBucketVec.end(),
              addressOffsetOfBucketCPU.data());
    auto ret = aclrtMemcpy(addressOffsetOfBucket->data(), addressOffsetOfBucket->getSizeInBytes(),
                           baseAddressOffsetOfBucketVec.data(), baseAddressOffsetOfBucketVec.size() * sizeof(uint64_t),
                           ACL_MEMCPY_HOST_TO_DEVICE);  // spanning distance of each l2 bucket (address difference)
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy baseAddressOffsetOfBucket to device failed %d", ret);
    std::copy(normL2AddressOffsetOfBucketVec.begin(), normL2AddressOffsetOfBucketVec.end(),
              addressOffsetOfBucketCPU.data() + bucketNum * 2);
    ret = aclrtMemcpy(addressOffsetOfBucket->data() + bucketNum * 2,
                      addressOffsetOfBucket->getSizeInBytes() - bucketNum * 2 * sizeof(uint64_t),
                      normL2AddressOffsetOfBucketVec.data(), normL2AddressOffsetOfBucketVec.size() * sizeof(uint64_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    // spanning distance of l2 norm for each l2 bucket (as opposed to
    // above, each bucket takes address of divUp(bucketVecNum, SEG_SIZE)
    // * dimStored, l2 norm only takes divUp(bucketVecNum, SEG_SIZE)
    // space)
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy normL2AddressOffsetOfBucket to device failed %d", ret);
    std::copy(idsAddressOfBucketVec.begin(), idsAddressOfBucketVec.end(),
              addressOffsetOfBucketCPU.data() + bucketNum * 4);
    ret = aclrtMemcpy(addressOffsetOfBucket->data() + bucketNum * 4,
                      addressOffsetOfBucket->getSizeInBytes() - bucketNum * 4 * sizeof(uint64_t),
                      idsAddressOfBucketVec.data(), idsAddressOfBucketVec.size() * sizeof(uint64_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy idsAddressOfBucketVec to device failed %d",
                            ret);  // beginning and end address of l2 bucket within l1 bucket
}

void NpuIndexIVFHSP::GetVectorsAssignNPU(size_t nb, float *baseRawData, int64_t *assign, float *codeWords,
                                         float *precomputeNormL2, int boostType, bool verbose)
{
    if (boostType == 1) {
        GetVectorsAssignNPUL1(nb, baseRawData, assign, codeWords, precomputeNormL2, verbose);
    }
}

void NpuIndexIVFHSP::GetVectorsAssignNPUL1(size_t nb, float *baseRawData, int64_t *assign, float *codeWords,
                                           float *precomputeNormL2, bool verbose)
{
    auto &mem = resources->getMemoryManager();
    AscendTensor<float, DIMS_3> codeBooksL1Npu(mem, { this->nList, this->subSpaceDimL1, this->dim }, defaultStream);
    auto ret = aclrtMemcpy(codeBooksL1Npu.data(), codeBooksL1Npu.getSizeInBytes(), codeBooksL1Cpu.data(),
                           codeBooksL1Npu.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", static_cast<int>(ret));

    size_t progressBatch = PROGRESSBATCH;
    size_t progressCnt = utils::divUp(nb, progressBatch);
    for (size_t p = 0; p < progressCnt; p++) {
        size_t start = p * progressBatch;
        size_t end = start + std::min(nb - start, progressBatch);
        size_t batchSize = end - start;

        AscendTensor<float, DIMS_3> codeWordsL1TmpNpu(mem, { PROGRESSBATCH, nList, subSpaceDimL1 }, defaultStream);
        std::vector<float> codeWordsL1Tmp(batchSize * static_cast<size_t>(nList) * static_cast<size_t>(subSpaceDimL1));
        std::vector<float16_t> codeWordsL1TmpHalf(batchSize * static_cast<size_t>(nList) * static_cast<size_t>(subSpaceDimL1));
        std::vector<float16_t> distanceL2Tmp(batchSize * static_cast<size_t>(nListL2));

        AscendTensor<float, DIMS_2> queryTmpNpu(mem, { PROGRESSBATCH, dim }, defaultStream);

        ret = aclrtMemcpy(queryTmpNpu.data(), (end - start) * dim * sizeof(float), baseRawData + start * dim,
                          (end - start) * dim * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", static_cast<int>(ret));
        RunMatMul(queryTmpNpu, codeBooksL1Npu, codeWordsL1TmpNpu);
        ret = aclrtSynchronizeStream(defaultStream);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "RunMatMul Add at L2 dists failed %d", ret);
        ret = aclrtMemcpy(codeWordsL1Tmp.data(), codeWordsL1Tmp.size() * sizeof(float), codeWordsL1TmpNpu.data(),
                          codeWordsL1Tmp.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy codeBooksL1Npu to Host failed %d", ret);
        // #pragma omp parallel
        {
            std::vector<float> codeWordsL1Tmp_iter(nList * subSpaceDimL1);
            std::vector<float> codeWordsL2Tmp(nListL2 * subSpaceDimL2);
            // #pragma omp for
            for (size_t i = start; i < end; i++) {
                ret = memcpy_s(codeWordsL1Tmp_iter.data(), nList * subSpaceDimL1 * sizeof(float),
                               codeWordsL1Tmp.data() + (i - start) * nList * subSpaceDimL1,
                               nList * subSpaceDimL1 * sizeof(float));
                ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
                int maxInd = -1;
                float maxNorm = -1;
                for (int k = 0; k < nList; k++) {
                    float normTmp = fvec_norm_L2sqr(codeWordsL1Tmp_iter.data() + k * subSpaceDimL1, subSpaceDimL1);
                    if (normTmp > maxNorm) {
                        maxNorm = normTmp;
                        maxInd = k;
                    }
                }
                precomputeNormL2[i] = maxNorm;
                ret = memcpy_s(codeWords + i * dimStored, dimStored * sizeof(float),
                               codeWordsL1Tmp_iter.data() + maxInd * subSpaceDimL1, dimStored * sizeof(float));
                ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
                MatMul(codeWordsL2Tmp.data(), codeWordsL1Tmp_iter.data() + maxInd * subSpaceDimL1,
                       codeBooksL2Cpu.data() + maxInd * nListL2 * subSpaceDimL2 * subSpaceDimL1, 1, subSpaceDimL1,
                       nListL2 * subSpaceDimL2, true);
                int maxIndL2 = -1;
                float maxNormL2 = -1;
                for (int k = 0; k < nListL2; k++) {
                    float normTmp = fvec_norm_L2sqr(codeWordsL2Tmp.data() + k * subSpaceDimL2, subSpaceDimL2);
                    if (normTmp > maxNormL2) {
                        maxNormL2 = normTmp;
                        maxIndL2 = k;
                    }
                }
                assign[i] = maxInd * nListL2 + maxIndL2;
            }
        }
        if (verbose) {
            printf("\r-----Progress:[%.2f]>>>>>>>>: %zu/%zu", end * 100.0 / nb, end, nb);
            fflush(stdout);
        }
    }
    printf("\n");
}

int NpuIndexIVFHSP::GetAddPageSize(size_t byteSizePerTerm)
{
    int addPageSize = static_cast<int>(ADD_PAGE_BYTE_SIZE / byteSizePerTerm);
    return std::max(addPageSize, 1);
}

APP_ERROR NpuIndexIVFHSP::DataTypeTransform(AscendTensor<float, DIMS_2> &floatDataNpu,
                                            AscendTensor<float16_t, DIMS_2> &fp16DataNpu)
{
#ifndef USE_ACL_NN_INTERFACE
    auto &mem = resources->getMemoryManager();
    auto stream = resources->getDefaultStream();
    AscendTensor<uint16_t, DIMS_2> flag(mem, { 8, 16 }, stream);
    int nq = floatDataNpu.getSize(0);
    AscendOperator *op = nullptr;
    if (fpToFp16Ops.find(nq) != fpToFp16Ops.end()) {
        op = fpToFp16Ops[nq].get();
    }
    RunFpToFp16(op, floatDataNpu, fp16DataNpu, flag);
    auto ret = aclrtSynchronizeStream(stream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtSynchronizeStream default stream: %i\n",
                             ret);
#endif
    return APP_ERR_OK;
}

template <typename T, aclDataType ACL_T>
APP_ERROR NpuIndexIVFHSP::ZzFormatReshape(AscendTensor<T, DIMS_2> &srcNpu, AscendTensor<T, DIMS_4> &dstNpu)
{
    int n = srcNpu.getSize(0);
    int dim = srcNpu.getSize(1);
    ASCEND_THROW_IF_NOT(n % CUBE_ALIGN == 0);
    ASCEND_THROW_IF_NOT(dim % CUBE_ALIGN == 0);
    std::string opName = "TransdataShapedSp";
    auto &mem = resources->getMemoryManager();
    auto stream = resources->getDefaultStream();
    AscendTensor<int64_t, DIMS_1> attr(mem,
                                       { static_cast<int32_t>(TransDataShapedAttrIdx::TRANSDATA_SHAPED_ATTR_COUNT) },
                                       stream);
    attr[static_cast<int32_t>(TransDataShapedAttrIdx::TRANSDATA_SHAPED_ATTR_NTOTAL_IDX)] = 0;
    LaunchOpTwoInOneOut<T, DIMS_2, ACL_T, int64_t, DIMS_1, ACL_INT64, T, DIMS_4, ACL_T>(opName, stream, srcNpu, attr,
                                                                                        dstNpu);
    auto ret = aclrtSynchronizeStream(stream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtSynchronizeStream ZzFormatReshape stream failed: %i\n", ret);
    return APP_ERR_OK;
}

template <typename T>
void NpuIndexIVFHSP::ZzFormatReshape(std::vector<T> &src, std::vector<T> &dst, size_t n, int dim, int nAlign,
                                     int dimAlign)
{
    ASCEND_THROW_IF_NOT(nAlign > 0 && nAlign % CUBE_ALIGN == 0);
    ASCEND_THROW_IF_NOT(dimAlign > 0 && dimAlign % CUBE_ALIGN == 0);
    ASCEND_THROW_IF_NOT(src.size() == n * static_cast<size_t>(dim));
    ASCEND_THROW_IF_NOT(dim % dimAlign == 0);
    size_t nMovCnt = static_cast<size_t>(utils::divUp(n, nAlign));
    size_t dimMovCnt = static_cast<size_t>(utils::divUp(dim, dimAlign));
    ASCEND_THROW_IF_NOT(dst.size() >= nMovCnt * dimMovCnt * static_cast<size_t>(nAlign) * static_cast<size_t>(dimAlign));
#pragma omp parallel for
    for (size_t i = 0; i < nMovCnt; i++) {
        for (int k = 0; k < nAlign; k++) {
            for (size_t j = 0; j < dimMovCnt; j++) {
                auto ret = memcpy_s(dst.data() + i * nAlign * dim + k * dimAlign + j * nAlign * dimAlign,
                                    dimAlign * sizeof(T),
                                    src.data() + i * nAlign * dim + k * dim + j * dimAlign, dimAlign * sizeof(T));
                ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
            }
        }
    }
}

size_t NpuIndexIVFHSP::GetSearchPagedSize(size_t n, int k) const
{
    // How many vectors fit into searchPageSize?
    size_t maxNumVecsForPageSize = SEARCH_PAGE_SIZE / (static_cast<size_t>(dim) * sizeof(float));
    size_t maxRetVecsForPageSize = SEARCH_PAGE_SIZE / (static_cast<size_t>(k) * (sizeof(uint16_t) + sizeof(uint64_t)));
    maxNumVecsForPageSize = std::min(maxNumVecsForPageSize, maxRetVecsForPageSize);

    // Always add at least 1 vector, if we have huge vectors
    maxNumVecsForPageSize = std::max(maxNumVecsForPageSize, static_cast<size_t>(1));
    return std::min(n, maxNumVecsForPageSize);
}

APP_ERROR NpuIndexIVFHSP::Search(size_t nq, float *queryData, int topK, float *dists, int64_t *labels) const
{
    if (aclrtSetDevice(ascendConfig.deviceList[0]) != ACL_SUCCESS) {
        APP_LOG_ERROR("Set device failed. deviceId is %d", ascendConfig.deviceList[0]);
        return ACL_ERROR_FAILURE;
    }
    APPERR_RETURN_IF(nq == 0, APP_ERR_OK);
    size_t totalSize = static_cast<size_t>(nq) * static_cast<size_t>(dim) * sizeof(float);
    size_t totalOutSize = nq * static_cast<size_t>(topK) * (sizeof(uint16_t) + sizeof(uint64_t));

    if (totalSize > SEARCH_PAGE_SIZE || nq > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
        size_t tileSize = GetSearchPagedSize(nq, topK);

        for (size_t i = 0; i < nq; i += tileSize) {
            size_t curNum = std::min(tileSize, nq - i);
            const_cast<NpuIndexIVFHSP &>(*this).SearchImpl(curNum, queryData + i * static_cast<size_t>(dim), topK,
                                                           dists + i * static_cast<size_t>(topK),
                                                           labels + i * static_cast<size_t>(topK));
        }
    } else {
        const_cast<NpuIndexIVFHSP &>(*this).SearchImpl(nq, queryData, topK, dists, labels);
    }
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::Search(const std::vector<NpuIndexIVFHSP *> &indexes, size_t nq, float *queryData, int topK,
                                 float *dists, int64_t *labels, bool merge)
{
    if (aclrtSetDevice(ascendConfig.deviceList[0]) != ACL_SUCCESS) {
        APP_LOG_ERROR("Set device failed. deviceId is %d", ascendConfig.deviceList[0]);
        return ACL_ERROR_FAILURE;
    }
    APPERR_RETURN_IF(nq == 0, APP_ERR_OK);

    size_t indexSize = indexes.size();
    ACL_REQUIRE_OK(ResetMultiL3TopKOp(indexSize));

    size_t totalSize = static_cast<size_t>(nq) * static_cast<size_t>(dim) * sizeof(float);
    size_t totalOutSize = nq * static_cast<size_t>(topK) * (sizeof(uint16_t) + sizeof(uint64_t));

    if (merge) {
        if (totalSize > SEARCH_PAGE_SIZE || nq > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
            size_t tileSize = GetSearchPagedSize(nq, topK);
            for (size_t i = 0; i < nq; i += tileSize) {
                size_t curNum = std::min(tileSize, nq - i);
                const_cast<NpuIndexIVFHSP &>(*this).SearchImpl(indexes, curNum,
                                                               queryData + i * static_cast<size_t>(dim),
                                                               topK, dists + i * static_cast<size_t>(topK),
                                                               labels + i * static_cast<size_t>(topK), merge);
            }
        } else {
            const_cast<NpuIndexIVFHSP &>(*this).SearchImpl(indexes, nq, queryData, topK, dists, labels, merge);
        }
    } else {
        if (totalSize > SEARCH_PAGE_SIZE || nq > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
            size_t tileSize = GetSearchPagedSize(nq, topK);
            for (size_t i = 0; i < nq; i += tileSize) {
                size_t curNum = std::min(tileSize, nq - i);
                const_cast<NpuIndexIVFHSP &>(*this).SearchImpl(indexes, curNum,
                                                               queryData + i * static_cast<size_t>(dim),
                                                               topK, dists + i * indexSize * static_cast<size_t>(topK),
                                                               labels + i * indexSize * static_cast<size_t>(topK),
                                                               merge);
            }
        } else {
            const_cast<NpuIndexIVFHSP &>(*this).SearchImpl(indexes, nq, queryData, topK, dists, labels, merge);
        }
    }

    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::Search(const std::vector<NpuIndexIVFHSP *> &indexes, size_t nq, uint8_t *mask,
                                 float *queryData, int topK, float *dists, int64_t *labels, bool merge)
{
    if (aclrtSetDevice(ascendConfig.deviceList[0]) != ACL_SUCCESS) {
        APP_LOG_ERROR("Set device failed. deviceId is %d", ascendConfig.deviceList[0]);
        return ACL_ERROR_FAILURE;
    }
    APPERR_RETURN_IF(nq == 0, APP_ERR_OK);
 
    size_t indexSize = indexes.size();
    ACL_REQUIRE_OK(ResetMultiL3TopKOp(indexSize));
 
    size_t totalSize = static_cast<size_t>(nq) * static_cast<size_t>(dim) * sizeof(float);
    size_t totalOutSize = nq * static_cast<size_t>(topK) * (sizeof(uint16_t) + sizeof(uint64_t));

    if (merge) {
        if (totalSize > SEARCH_PAGE_SIZE || nq > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
        size_t tileSize = GetSearchPagedSize(nq, topK);
        for (size_t i = 0; i < nq; i += tileSize) {
            size_t curNum = std::min(tileSize, nq - i);
            const_cast<NpuIndexIVFHSP &>(*this).SearchImpl(indexes, curNum, mask,
                                                           queryData + i * static_cast<size_t>(dim),
                                                           topK, dists + i * static_cast<size_t>(topK),
                                                           labels + i * static_cast<size_t>(topK), merge);
            }
        } else {
            const_cast<NpuIndexIVFHSP &>(*this).SearchImpl(indexes, nq, mask, queryData, topK, dists, labels, merge);
        }
    } else {
        if (totalSize > SEARCH_PAGE_SIZE || nq > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
        size_t tileSize = GetSearchPagedSize(nq, topK);
        for (size_t i = 0; i < nq; i += tileSize) {
            size_t curNum = std::min(tileSize, nq - i);
            const_cast<NpuIndexIVFHSP &>(*this).SearchImpl(indexes, curNum, mask,
                                                           queryData + i * static_cast<size_t>(dim),
                                                           topK, dists + i * indexSize * static_cast<size_t>(topK),
                                                           labels + i * indexSize * static_cast<size_t>(topK), merge);
            }
        } else {
            const_cast<NpuIndexIVFHSP &>(*this).SearchImpl(indexes, nq, mask, queryData, topK, dists, labels, merge);
        }
    }
 
    return APP_ERR_OK;
}
 

APP_ERROR NpuIndexIVFHSP::Search(size_t nq, uint8_t *mask, float *queryData, int topK, float *dists,
                                 int64_t *labels) const
{
    if (aclrtSetDevice(ascendConfig.deviceList[0]) != ACL_SUCCESS) {
        APP_LOG_ERROR("Set device failed. deviceId is %d", ascendConfig.deviceList[0]);
        return ACL_ERROR_FAILURE;
    }

    APPERR_RETURN_IF(nq == 0, APP_ERR_OK);
    size_t totalSize = static_cast<size_t>(nq) * static_cast<size_t>(dim) * sizeof(float);
    size_t totalOutSize = nq * static_cast<size_t>(topK) * (sizeof(uint16_t) + sizeof(uint64_t));

    if (totalSize > SEARCH_PAGE_SIZE || nq > SEARCH_VEC_SIZE || totalOutSize > SEARCH_PAGE_SIZE) {
        size_t tileSize = GetSearchPagedSize(nq, topK);
        for (size_t i = 0; i < nq; i += tileSize) {
            size_t curNum = std::min(tileSize, nq - i);
            const_cast<NpuIndexIVFHSP &>(*this).SearchImpl(curNum, mask, queryData + i * static_cast<size_t>(dim), topK,
                                                           dists + i * static_cast<size_t>(topK),
                                                           labels + i * static_cast<size_t>(topK));
        }
    } else {
        const_cast<NpuIndexIVFHSP &>(*this).SearchImpl(nq, mask, queryData, topK, dists, labels);
    }
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::Search(std::vector<float> &queryData, int topK, std::vector<float> &dists,
                                 std::vector<int64_t> &labels) const
{
    size_t nq = queryData.size() / dim;
    APPERR_RETURN_IF(nq == 0, APP_ERR_OK);
    ASCEND_THROW_IF_NOT(nq * dim == queryData.size());
    ASCEND_THROW_IF_NOT(topK > 0);
    ASCEND_THROW_IF_NOT(nq * static_cast<size_t>(topK) == dists.size());
    ASCEND_THROW_IF_NOT(nq * static_cast<size_t>(topK) == labels.size());

    return Search(nq, queryData.data(), topK, dists.data(), labels.data());
}

/**
 *  思考下如何把search的流水搞起来
 * @param n
 * @param x
 * @param k
 * @param distances
 * @param labels
 */
void NpuIndexIVFHSP::SearchImpl(int n, const uint8_t *mask, const float *x, int k, float *distances, int64_t *labels)
{
    auto &mem = resources->getMemoryManager();
    this->maskFlag = true;

    AscendTensor<float, DIMS_2> queryNpu(mem, { n, dim }, defaultStream);

    auto ret = aclrtMemcpy(queryNpu.data(), queryNpu.getSizeInBytes(), x, n * dim * sizeof(float),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy queryNpu error %d", static_cast<int>(ret));

    AscendTensor<uint8_t, DIMS_1> maskBitNpu(mem, { static_cast<int>(static_cast<uint64_t>(n) * ((ntotal + 7) / 8)) }, defaultStream);

    ret = aclrtMemcpy(maskBitNpu.data(), maskBitNpu.getSizeInBytes(), mask,  n * ((ntotal + 7) / 8) * sizeof(uint8_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy maskBitNpu error %d", static_cast<int>(ret));

    int searchCnt = 0;
    std::vector<float16_t> distHalf(n * k);
    int batchSize = 1;
    while (n - searchCnt >= batchSize) {
        AscendTensor<float, DIMS_2> queryTmpNpu(queryNpu.data() + searchCnt * dim, { batchSize, dim });
        AscendTensor<uint8_t, DIMS_1> maskBitTmpNpu(maskBitNpu.data() + static_cast<uint64_t>(searchCnt) * ((ntotal + 7) / 8),
                                                    { static_cast<int>(static_cast<uint64_t>(batchSize) * (ntotal + 7) / 8) });
        ret = SearchBatchImpl(batchSize, maskBitTmpNpu, queryTmpNpu, k, distHalf.data() + searchCnt * k,
                              labels + searchCnt * k);

        ASCEND_THROW_IF_NOT(ret == APP_ERR_OK);
        searchCnt += batchSize;
    }

    // convert result data from fp16 to float
    transform(distHalf.begin(), distHalf.end(), distances, [](float16_t temp) {
        auto convert = FormatConvert();
        convert.data = temp;
        return static_cast<float>(fp16(convert.value));
    });
}

void NpuIndexIVFHSP::SearchImpl(int n, const float *x, int k, float *distances, int64_t *labels)
{
    auto &mem = resources->getMemoryManager();
    this->maskFlag = false;

    AscendTensor<float, DIMS_2> queryNpu(mem, { n, dim }, defaultStream);
    auto ret = aclrtMemcpy(queryNpu.data(), queryNpu.getSizeInBytes(), x, n * dim * sizeof(float),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", static_cast<int>(ret));

    int searchCnt = 0;
    std::vector<float16_t> distHalf(n * k);
    for (auto batchSize : opAccessBatchList) {
        while (n - searchCnt >= batchSize) {
            AscendTensor<float, DIMS_2> queryTmpNpu(queryNpu.data() + searchCnt * dim, { batchSize, dim });
            ret = SearchBatchImpl(batchSize, queryTmpNpu, k, distHalf.data() + searchCnt * k, labels + searchCnt * k);
            ASCEND_THROW_IF_NOT(ret == APP_ERR_OK);
            searchCnt += batchSize;
        }
    }
    // convert result data from fp16 to float
    transform(distHalf.begin(), distHalf.end(), distances, [](float16_t temp) {
        auto convert = FormatConvert();
        convert.data = temp;
        return static_cast<float>(fp16(convert.value));
    });
}

void NpuIndexIVFHSP::SearchImpl(const std::vector<NpuIndexIVFHSP *> &indexes, int n, const float *x, int k,
                                float *distances, int64_t *labels, bool merge)
{
    auto &mem = resources->getMemoryManager();
    this->maskFlag = false;

    AscendTensor<float, DIMS_2> queryNpu(mem, { n, dim }, defaultStream);
    auto ret = aclrtMemcpy(queryNpu.data(), queryNpu.getSizeInBytes(), x, n * dim * sizeof(float),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy error %d", static_cast<int>(ret));

    int searchCnt = 0;
    if (merge) {
        std::vector<float16_t> distHalf(n * k);
        for (auto batchSize : opAccessBatchList) {
            while (n - searchCnt >= batchSize) {
                AscendTensor<float, DIMS_2> queryTmpNpu(queryNpu.data() + searchCnt * dim, { batchSize, dim });
                ret = SearchBatchImpl(indexes, batchSize, queryTmpNpu, k, distHalf.data() + searchCnt * k,
                                      labels + searchCnt * k, merge);

                ASCEND_THROW_IF_NOT(ret == APP_ERR_OK);
                searchCnt += batchSize;
            }
        }
        // convert result data from fp16 to float
        transform(distHalf.begin(), distHalf.end(), distances, [](float16_t temp) {
            auto convert = FormatConvert();
            convert.data = temp;
            return static_cast<float>(fp16(convert.value));
        });
    } else {
        std::vector<float16_t> distHalf(n * static_cast<int>(indexes.size()) * k);
        for (auto batchSize : opAccessBatchList) {
            while (n - searchCnt >= batchSize) {
                AscendTensor<float, DIMS_2> queryTmpNpu(queryNpu.data() + searchCnt * dim, { batchSize, dim });
                ret = SearchBatchImpl(indexes, batchSize, queryTmpNpu, k,
                                      distHalf.data() + searchCnt * indexes.size() * k,
                                      labels + searchCnt * indexes.size() * k, merge);

                ASCEND_THROW_IF_NOT(ret == APP_ERR_OK);
                searchCnt += batchSize;
            }
        }
        // convert result data from fp16 to float
        transform(distHalf.begin(), distHalf.end(), distances, [](float16_t temp) {
            auto convert = FormatConvert();
            convert.data = temp;
            return static_cast<float>(fp16(convert.value));
        });
    }
}

void NpuIndexIVFHSP::SearchImpl(const std::vector<NpuIndexIVFHSP *> &indexes, int n, const uint8_t *mask,
                                const float *x, int k, float *distances, int64_t *labels, bool merge)
{
    auto &mem = resources->getMemoryManager();
    this->maskFlag = true;
 
    AscendTensor<float, DIMS_2> queryNpu(mem, { n, dim }, defaultStream);
 
    auto ret = aclrtMemcpy(queryNpu.data(), queryNpu.getSizeInBytes(), x, n * dim * sizeof(float),
                           ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy queryNpu error %d", static_cast<int>(ret));
 
    int searchCnt = 0;
    if (merge) {
        std::vector<float16_t> distHalf(n * k);
        int batchSize = 1;
        while (n - searchCnt >= batchSize) {
            AscendTensor<float, DIMS_2> queryTmpNpu(queryNpu.data() + searchCnt * dim, { batchSize, dim });
            ret = SearchBatchImpl(indexes, batchSize, mask + searchCnt * ((ntotal + 7) / 8),
                                  queryTmpNpu, k, distHalf.data() + searchCnt * k, labels + searchCnt * k, merge);
            ASCEND_THROW_IF_NOT(ret == APP_ERR_OK);
            searchCnt += batchSize;
        }
    // convert result data from fp16 to float
        transform(distHalf.begin(), distHalf.end(), distances, [](float16_t temp) {
            auto convert = FormatConvert();
            convert.data = temp;
            return static_cast<float>(fp16(convert.value));
        });
    } else {
        std::vector<float16_t> distHalf(n * static_cast<int>(indexes.size()) * k);
        int batchSize = 1;
        {
            std::vector<float16_t> distHalfTemp(indexes.size() * static_cast<size_t>(batchSize) * static_cast<size_t>(k));
            std::vector<int64_t> labelsTemp(indexes.size() * static_cast<size_t>(batchSize) * static_cast<size_t>(k));
            while (n - searchCnt >= batchSize) {
                AscendTensor<float, DIMS_2> queryTmpNpu(queryNpu.data() + searchCnt * dim, { batchSize, dim });
                ret = SearchBatchImpl(indexes, batchSize, mask + searchCnt * ((ntotal + 7) / 8), queryTmpNpu, k,
                                      distHalfTemp.data(), labelsTemp.data(), merge);
                ASCEND_THROW_IF_NOT(ret == APP_ERR_OK);
                for (size_t indexID = 0; indexID < indexes.size(); indexID++) {
                    std::copy(distHalfTemp.begin() + indexID * batchSize * k,
                              distHalfTemp.begin() + indexID * batchSize * k + batchSize * k,
                              distHalf.begin() + indexID * n * k + searchCnt * k);
                    std::copy(labelsTemp.begin() + indexID * batchSize * k,
                              labelsTemp.begin() + indexID * batchSize * k + batchSize * k,
                              labels+indexID * n * k + searchCnt * k);
                }
                searchCnt += batchSize;
            }
        }
        // convert result data from fp16 to float
        transform(distHalf.begin(), distHalf.end(), distances, [](float16_t temp) {
            auto convert = FormatConvert();
            convert.data = temp;
            return static_cast<float>(fp16(convert.value));
        });
    }
}
 

APP_ERROR NpuIndexIVFHSP::SearchBatchImpl(int n, AscendTensor<uint8_t, DIMS_1> &maskBitNpu,
                                          AscendTensor<float, DIMS_2> &queryNpu, int k, float16_t *distances,
                                          int64_t *labels)
{
    auto &mem = resources->getMemoryManager();

    // L1 search, to find nprobe IVF list
    // L1 topk op output: top nprobe dist and top nprobe label
    AscendTensor<float16_t, DIMS_2> queryCodes(mem, { n, nList * subSpaceDimL1 }, defaultStream);
    AscendTensor<uint16_t, DIMS_2> l1KIndicesNpu(mem, { n, searchParam->nProbeL1 }, defaultStream);
    auto ret = SearchBatchImplL1(queryNpu, queryCodes, l1KIndicesNpu);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfhsp search l1 failed: %i\n", ret);

    // 2. L2 search, search codes in nprobe IVF list to find topk results
    // L2 output, if SUBCENTER_NUM is dynymic need to cat tiles
    // 目前距离计算和topk串行，下一步，针对大batch，进行流水线优化

    AscendTensor<uint64_t, DIMS_2> addressOffsetL3(mem, { n, searchParam->nProbeL2 * 6 }, defaultStream);
    AscendTensor<uint64_t, DIMS_2> idAdressL3(mem, { n, searchParam->nProbeL2 * 2 }, defaultStream);

    ret = SearchBatchImplL2(maskBitNpu, queryCodes, l1KIndicesNpu, addressOffsetL3, idAdressL3);

    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfhsp search l2 failed: %i\n", ret);

    // 3. L3 Search
    AscendTensor<float16_t, DIMS_2> outDists(mem, { n, k }, defaultStream);
    AscendTensor<int64_t, DIMS_2> outlabels(mem, { n, k }, defaultStream);

    ret = SearchBatchImplL3(queryCodes, addressOffsetL3, idAdressL3, outDists, outlabels);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfhsp search l3 failed: %i\n", ret);

    ret = aclrtMemcpy(distances, n * k * sizeof(float16_t), outDists.data(), outDists.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy distance to host failed:%d\n", ret);
    ret = aclrtMemcpy(labels, n * k * sizeof(int64_t), outlabels.data(), outlabels.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy labels to host failed:%d\n", ret);

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n * k; ++i) {
        if (FloatEqual(distances[i], 100.0)) {
            labels[i] = -1;
        }
    }

    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::SearchBatchImpl(const std::vector<NpuIndexIVFHSP *> &indexes, int n, const uint8_t *mask,
                                          AscendTensor<float, DIMS_2> &queryNpu, int k, float16_t *distances,
                                          int64_t *labels, bool merge)
{
    auto &mem = resources->getMemoryManager();
    int indexSize = static_cast<int>(indexes.size());
 
    // L1 search, to find nprobe IVF list
    // L1 topk op output: top nprobe dist and top nprobe label
    AscendTensor<float16_t, DIMS_2> queryCodes(mem, { n, nList * subSpaceDimL1 }, defaultStream);
    AscendTensor<uint16_t, DIMS_2> l1KIndicesNpu(mem, { n, searchParam->nProbeL1 }, defaultStream);
    auto ret = SearchBatchImplL1(queryNpu, queryCodes, l1KIndicesNpu);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfhsp search l1 failed: %i\n", ret);
 
    // 2. L2 search, search codes in nprobe IVF list to find topk results
    // L2 output, if SUBCENTER_NUM is dynymic need to cat tiles
    // 目前距离计算和topk串行，下一步，针对大batch，进行流水线优化
    AscendTensor<uint64_t, DIMS_2> labelL2(mem, { n, searchParam->nProbeL2 }, defaultStream);
    std::vector<uint64_t> labelL2Cpu(n * searchParam->nProbeL2);
    // IvfSpL2DistOp Output
    ret = SearchBatchImplMultiL2(queryCodes, l1KIndicesNpu, labelL2);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfhsp multi search l2 failed: %i\n", ret);
 
    ret = aclrtMemcpy(labelL2Cpu.data(), labelL2Cpu.size() * sizeof(uint64_t), labelL2.data(), labelL2.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ACL_REQUIRE_OK(ret);
 
    // Host Calculate L3 data Address Offset
    AscendTensor<uint64_t, DIMS_3> addressOffsetL3(mem, { indexSize, n, searchParam->nProbeL2 * 6 }, defaultStream);
    AscendTensor<uint64_t, DIMS_3> idAdressL3(mem, { indexSize, n, searchParam->nProbeL2 * 2 }, defaultStream);
    std::vector<std::vector<uint64_t> > outOffsetList(indexSize,
                                                      std::vector<uint64_t>(n * this->searchParam->nProbeL2 * 5));
    std::vector<std::vector<uint64_t> > outIdsOffsetList(indexSize,
                                                         std::vector<uint64_t>(n * this->searchParam->nProbeL2 * 2));
    std::vector<std::future<bool> > results;
    for (int i = 0; i < indexSize; i++) {
        auto result = pool->enqueue(&NpuIndexIVFHSP::CalculateOffsetL3WithMask, this, std::ref(indexes), n, mask, i,
                                    std::ref(labelL2Cpu), std::ref(outOffsetList[i]), std::ref(outIdsOffsetList[i]));
        results.emplace_back(std::move(result));
    }
 
    // Delivering TopKL3 Operators and listening to L3Distance flags
    AscendTensor<float16_t, DIMS_3> distResult(mem, { indexSize, n, searchParam->l3SegmentNum * BASE_SEG_SIZE },
                                               defaultStream);
    AscendTensor<float16_t, DIMS_3> vcMinDistResult(mem, { indexSize, n,
        2 * searchParam->l3SegmentNum * 2 * BASE_SEG_SIZE / VCMIN_SEG_SIZE }, defaultStream);
    std::vector<float16_t> vcMinDistResultVec(2 * indexSize * n * searchParam->l3SegmentNum * 2 * BASE_SEG_SIZE /
                                              VCMIN_SEG_SIZE, 0.0f);
    ret = aclrtMemcpy(vcMinDistResult.data(), vcMinDistResult.getSizeInBytes(), vcMinDistResultVec.data(),
                      vcMinDistResultVec.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy vcMinDistResult to device failed %d", ret);
 
    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { indexSize, CORE_NUM, FLAG_SIZE }, defaultStream);
    std::vector<uint16_t> opFlagVec(indexSize * CORE_NUM * FLAG_SIZE, 0);
    ret = aclrtMemcpy(opFlag.data(), opFlag.getSizeInBytes(), opFlagVec.data(), opFlagVec.size() * sizeof(uint16_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy opFlag to device failed %d", ret);
    AscendTensor<float16_t, DIMS_3> outDists(mem, { indexSize, n, k }, aiCpuStream);
    AscendTensor<int64_t, DIMS_3> outlabels(mem, { indexSize, n, k }, aiCpuStream);
    RunMultiL3TopKOp(distResult, vcMinDistResult, opFlag, idAdressL3, *searchL3OpAttrs, addressOffsetL3, outDists,
                     outlabels);
 
    // waiting and postProcess ‘CalculateOffsetL3’ result
    std::vector<uint64_t> addressOffsetL3CpuContinue(indexSize * n * searchParam->nProbeL2 * 6);
    std::vector<uint64_t> outIdsOffsetListCpuContinue(indexSize * n * this->searchParam->nProbeL2 * 2);
 
    // data flow: labelL2Cpu (n * nProbeL2) ->
    for (int i = 0; i < indexSize; i++) {
        results[i].get();
        auto ret = aclrtMemcpy(indexes[i]->maskByteNpu->data(), maskByteCpu.size() * sizeof(uint8_t),
                               indexes[i]->maskByteCpu.data(), maskByteCpu.size() * sizeof(uint8_t),
                               ACL_MEMCPY_HOST_TO_DEVICE);  // spanning distance of each l2 bucket (address difference)
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy maskByteCpu to device failed %d", ret);
 
        for (int j = 0; j < n; j++) {
            ret = memcpy_s(addressOffsetL3CpuContinue.data() + i * n * searchParam->nProbeL2 * 6 +
                j * searchParam->nProbeL2 * 6, searchParam->nProbeL2 * sizeof(uint64_t),
                labelL2Cpu.data() + j * searchParam->nProbeL2, searchParam->nProbeL2 * sizeof(uint64_t));
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
            ret = memcpy_s(addressOffsetL3CpuContinue.data() + i * n * searchParam->nProbeL2 * 6 +
                j * searchParam->nProbeL2 * 6 + searchParam->nProbeL2, searchParam->nProbeL2 * 5 * sizeof(uint64_t),
                outOffsetList[i].data() + j * searchParam->nProbeL2 * 5, searchParam->nProbeL2 * 5 * sizeof(uint64_t));
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
        }
 
        ret = memcpy_s(outIdsOffsetListCpuContinue.data() + i * n * searchParam->nProbeL2 * 2,
            n * searchParam->nProbeL2 * 2 * sizeof(uint64_t), outIdsOffsetList[i].data(),
            n * searchParam->nProbeL2 * 2 * sizeof(uint64_t));
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
    }
    ret = aclrtMemcpy(addressOffsetL3.data(), addressOffsetL3.getSizeInBytes(), addressOffsetL3CpuContinue.data(),
                      addressOffsetL3.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_REQUIRE_OK(ret);
    ret = aclrtMemcpy(idAdressL3.data(), idAdressL3.getSizeInBytes(), outIdsOffsetListCpuContinue.data(),
                      idAdressL3.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_REQUIRE_OK(ret);
 
    // Delivering L3Distance Operators
    for (int i = 0; i < indexSize; i++) {
        ret = SearchBatchImplMultiL3(indexes, i, queryCodes, addressOffsetL3, idAdressL3, distResult, vcMinDistResult,
                                     opFlag);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfhsp search l3 failed: %i\n", ret);
    }
 
    // waiting all stream finish and copy results to Host
    ret = aclrtSynchronizeStream(aiCpuStream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream aicpu stream failed: %d\n", ret);
 
    std::vector<float16_t> multiIndexDists(indexSize * n * k);
    std::vector<int64_t> multiIndexLabels(indexSize * n * k);
 
    if (merge) {
        ret = aclrtMemcpy(multiIndexDists.data(), indexSize * n * k * sizeof(float16_t), outDists.data(),
                          indexSize * n * k * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy distance to host failed:%d\n", ret);
        ret = aclrtMemcpy(multiIndexLabels.data(), indexSize * n * k * sizeof(int64_t), outlabels.data(),
                          indexSize * n * k * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    } else {
        ret = aclrtMemcpy(distances, indexSize * n * k * sizeof(float16_t), outDists.data(),
                          indexSize * n * k * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy distance to host failed:%d\n", ret);
        ret = aclrtMemcpy(labels, indexSize * n * k * sizeof(int64_t), outlabels.data(),
                          indexSize * n * k * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    }
 
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy labels to host failed:%d\n", ret);
 
    if (merge) {
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<float16_t, int64_t> > dist_label_pairs(indexSize * k);
            for (int id_index = 0; id_index < indexSize; ++id_index) {
                for (int id_k = 0; id_k < k; ++id_k) {
                    auto offset = id_index * n * k + i * k + id_k;
                    if (indexes[id_index]->isAddWithIds) {
                        dist_label_pairs[id_index * k + id_k] =
                            std::make_pair(multiIndexDists[offset],
                                           indexes[id_index]->GetIdMap().at(multiIndexLabels[offset]));
                    } else {
                        dist_label_pairs[id_index * k + id_k] =
                            std::make_pair(multiIndexDists[offset],
                                           multiIndexLabels[offset]);
                    }
                }
            }
 
            std::sort(dist_label_pairs.begin(), dist_label_pairs.end(),
                      [](const std::pair<float16_t, int64_t> &a, const std::pair<float16_t, int64_t> &b) {
                          return a.first < b.first;  // 按照距离升序排序
                      });

            for (int id_k = 0; id_k < k; ++id_k) {
                distances[i * k + id_k] = dist_label_pairs[id_k].first;
                labels[i * k + id_k] = dist_label_pairs[id_k].second;
            }
        }

#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i <  n * k; ++i) {
            if (FloatEqual(distances[i], 100.0)) { // 100 is the predefined distance we set for labels that are masked
                labels[i] = -1;
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int id_index = 0; id_index < indexSize; ++id_index) {
                for (int id_k = 0; id_k < k; ++id_k) {
                    auto offset = id_index * n * k + i * k + id_k;
                    if (indexes[id_index]->isAddWithIds) {
                        labels[offset] = indexes[id_index]->GetIdMap().at(labels[offset]);
                    }
                }
            }
        }
#pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < indexSize * n * k; ++i) {
            if (FloatEqual(distances[i], 100.0)) { // 100 is the predefined distance we set for labels that are masked
                labels[i] = -1;
            }
        }
    }
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::SearchBatchImpl(int n, AscendTensor<float, DIMS_2> &queryNpu, int k, float16_t *distances,
                                          int64_t *labels)
{
    auto &mem = resources->getMemoryManager();

    // L1 search, to find nprobe IVF list
    // L1 topk op output: top nprobe dist and top nprobe label
    AscendTensor<float16_t, DIMS_2> queryCodes(mem, { n, nList * subSpaceDimL1 }, defaultStream);
    AscendTensor<uint16_t, DIMS_2> l1KIndicesNpu(mem, { n, searchParam->nProbeL1 }, defaultStream);
    auto ret = SearchBatchImplL1(queryNpu, queryCodes, l1KIndicesNpu);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfhsp search l1 failed: %i\n", ret);

    // 2. L2 search, search codes in nprobe IVF list to find topk results
    // L2 output, if SUBCENTER_NUM is dynymic need to cat tiles
    // 目前距离计算和topk串行，下一步，针对大batch，进行流水线优化

    AscendTensor<uint64_t, DIMS_2> addressOffsetL3(mem, { n, searchParam->nProbeL2 * 6 }, defaultStream);
    AscendTensor<uint64_t, DIMS_2> idAdressL3(mem, { n, searchParam->nProbeL2 * 2 }, defaultStream);
    ret = SearchBatchImplL2(queryCodes, l1KIndicesNpu, addressOffsetL3, idAdressL3);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfhsp search l2 failed: %i\n", ret);

    // 3. L3 Search

    AscendTensor<float16_t, DIMS_2> outDists(mem, { n, k }, defaultStream);
    AscendTensor<int64_t, DIMS_2> outlabels(mem, { n, k }, defaultStream);

    ret = SearchBatchImplL3(queryCodes, addressOffsetL3, idAdressL3, outDists, outlabels);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfhsp search l3 failed: %i\n", ret);

    ret = aclrtMemcpy(distances, n * k * sizeof(float16_t), outDists.data(), outDists.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy distance to host failed:%d\n", ret);
    ret = aclrtMemcpy(labels, n * k * sizeof(int64_t), outlabels.data(), outlabels.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy labels to host failed:%d\n", ret);

    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::SearchBatchImpl(const std::vector<NpuIndexIVFHSP *> &indexes, int n,
                                          AscendTensor<float, DIMS_2> &queryNpu, int k, float16_t *distances,
                                          int64_t *labels, bool merge)
{
    auto &mem = resources->getMemoryManager();
    int indexSize = static_cast<int>(indexes.size());

    // L1 search, to find nprobe IVF list
    // L1 topk op output: top nprobe dist and top nprobe label
    AscendTensor<float16_t, DIMS_2> queryCodes(mem, { n, nList * subSpaceDimL1 }, defaultStream);
    AscendTensor<uint16_t, DIMS_2> l1KIndicesNpu(mem, { n, searchParam->nProbeL1 }, defaultStream);
    auto ret = SearchBatchImplL1(queryNpu, queryCodes, l1KIndicesNpu);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfhsp search l1 failed: %i\n", ret);

    // 2. L2 search, search codes in nprobe IVF list to find topk results
    // L2 output, if SUBCENTER_NUM is dynymic need to cat tiles
    // 目前距离计算和topk串行，下一步，针对大batch，进行流水线优化
    AscendTensor<uint64_t, DIMS_2> labelL2(mem, { n, searchParam->nProbeL2 }, defaultStream);
    std::vector<uint64_t> labelL2Cpu(n * searchParam->nProbeL2);
    // IvfSpL2DistOp Output
    ret = SearchBatchImplMultiL2(queryCodes, l1KIndicesNpu, labelL2);
    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfhsp multi search l2 failed: %i\n", ret);

    ret = aclrtMemcpy(labelL2Cpu.data(), labelL2Cpu.size() * sizeof(uint64_t), labelL2.data(), labelL2.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ACL_REQUIRE_OK(ret);

    // Host Calculate L3 data Address Offset
    AscendTensor<uint64_t, DIMS_3> addressOffsetL3(mem, { indexSize, n, searchParam->nProbeL2 * 6 }, defaultStream);
    AscendTensor<uint64_t, DIMS_3> idAdressL3(mem, { indexSize, n, searchParam->nProbeL2 * 2 }, defaultStream);
    std::vector<std::vector<uint64_t> > outOffsetList(indexSize,
                                                      std::vector<uint64_t>(n * this->searchParam->nProbeL2 * 5));
    std::vector<std::vector<uint64_t> > outIdsOffsetList(indexSize,
                                                         std::vector<uint64_t>(n * this->searchParam->nProbeL2 * 2));
    std::vector<std::future<bool> > results;
    for (int i = 0; i < indexSize; i++) {
        auto result = pool->enqueue(&NpuIndexIVFHSP::CalculateOffsetL3, this, std::ref(indexes), n, i,
                                    std::ref(labelL2Cpu), std::ref(outOffsetList[i]), std::ref(outIdsOffsetList[i]));
        results.emplace_back(std::move(result));
    }

    // Delivering TopKL3 Operators and listening to L3Distance flags
    AscendTensor<float16_t, DIMS_3> distResult(mem, { indexSize, n, searchParam->l3SegmentNum * BASE_SEG_SIZE },
                                               defaultStream);
    AscendTensor<float16_t, DIMS_3> vcMinDistResult(mem, { indexSize, n,
        2 * searchParam->l3SegmentNum * 2 * BASE_SEG_SIZE / VCMIN_SEG_SIZE }, defaultStream);
    std::vector<float16_t> vcMinDistResultVec(2 * indexSize * n * searchParam->l3SegmentNum * 2 * BASE_SEG_SIZE /
                                              VCMIN_SEG_SIZE, 0.0f);
    ret = aclrtMemcpy(vcMinDistResult.data(), vcMinDistResult.getSizeInBytes(), vcMinDistResultVec.data(),
                      vcMinDistResultVec.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy vcMinDistResult to device failed %d", static_cast<int>(ret));

    AscendTensor<uint16_t, DIMS_3> opFlag(mem, { indexSize, CORE_NUM, FLAG_SIZE }, defaultStream);
    std::vector<uint16_t> opFlagVec(indexSize * CORE_NUM * FLAG_SIZE, 0);
    ret = aclrtMemcpy(opFlag.data(), opFlag.getSizeInBytes(), opFlagVec.data(), opFlagVec.size() * sizeof(uint16_t),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy opFlag to device failed %d", static_cast<int>(ret));
    AscendTensor<float16_t, DIMS_3> outDists(mem, { indexSize, n, k }, aiCpuStream);
    AscendTensor<int64_t, DIMS_3> outlabels(mem, { indexSize, n, k }, aiCpuStream);
    RunMultiL3TopKOp(distResult, vcMinDistResult, opFlag, idAdressL3, *searchL3OpAttrs, addressOffsetL3, outDists,
                     outlabels);

    // waiting and postProcess ‘CalculateOffsetL3’ result
    std::vector<uint64_t> addressOffsetL3CpuContinue(indexSize * n * searchParam->nProbeL2 * 6);
    std::vector<uint64_t> outIdsOffsetListCpuContinue(indexSize * n * this->searchParam->nProbeL2 * 2);

    // data flow: labelL2Cpu (n * nProbeL2) ->
    for (int i = 0; i < indexSize; i++) {
        results[i].get();
        for (int j = 0; j < n; j++) {
            ret = memcpy_s(addressOffsetL3CpuContinue.data() + i * n * searchParam->nProbeL2 * 6 +
                j * searchParam->nProbeL2 * 6, searchParam->nProbeL2 * sizeof(uint64_t),
                labelL2Cpu.data() + j * searchParam->nProbeL2, searchParam->nProbeL2 * sizeof(uint64_t));
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
            ret = memcpy_s(addressOffsetL3CpuContinue.data() + i * n * searchParam->nProbeL2 * 6 +
                j * searchParam->nProbeL2 * 6 + searchParam->nProbeL2, searchParam->nProbeL2 * 5 * sizeof(uint64_t),
                outOffsetList[i].data() + j * searchParam->nProbeL2 * 5, searchParam->nProbeL2 * 5 * sizeof(uint64_t));
            ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
        }

        ret = memcpy_s(outIdsOffsetListCpuContinue.data() + i * n * searchParam->nProbeL2 * 2,
            n * searchParam->nProbeL2 * 2 * sizeof(uint64_t), outIdsOffsetList[i].data(),
            n * searchParam->nProbeL2 * 2 * sizeof(uint64_t));
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Mem operator error %d", static_cast<int>(ret));
    }
    ret = aclrtMemcpy(addressOffsetL3.data(), addressOffsetL3.getSizeInBytes(), addressOffsetL3CpuContinue.data(),
                      addressOffsetL3.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_REQUIRE_OK(ret);
    ret = aclrtMemcpy(idAdressL3.data(), idAdressL3.getSizeInBytes(), outIdsOffsetListCpuContinue.data(),
                      idAdressL3.getSizeInBytes(), ACL_MEMCPY_HOST_TO_DEVICE);
    ACL_REQUIRE_OK(ret);

    // Delivering L3Distance Operators
    for (int i = 0; i < indexSize; i++) {
        ret = SearchBatchImplMultiL3(indexes, i, queryCodes, addressOffsetL3, idAdressL3, distResult, vcMinDistResult,
                                     opFlag);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "ivfhsp search l3 failed: %i\n", ret);
    }

    // waiting all stream finish and copy results to Host
    ret = aclrtSynchronizeStream(aiCpuStream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream aicpu stream failed: %d\n", ret);

    std::vector<float16_t> multiIndexDists(indexSize * n * k);
    std::vector<int64_t> multiIndexLabels(indexSize * n * k);

    if (merge) {
        ret = aclrtMemcpy(multiIndexDists.data(), indexSize * n * k * sizeof(float16_t), outDists.data(),
                          indexSize * n * k * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy distance to host failed:%d\n", ret);
        ret = aclrtMemcpy(multiIndexLabels.data(), indexSize * n * k * sizeof(int64_t), outlabels.data(),
                          indexSize * n * k * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    } else {
        ret = aclrtMemcpy(distances, indexSize * n * k * sizeof(float16_t), outDists.data(),
                          indexSize * n * k * sizeof(float16_t), ACL_MEMCPY_DEVICE_TO_HOST);
        APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy distance to host failed:%d\n", ret);
        ret = aclrtMemcpy(labels, indexSize * n * k * sizeof(int64_t), outlabels.data(),
                          indexSize * n * k * sizeof(int64_t), ACL_MEMCPY_DEVICE_TO_HOST);
    }

    APPERR_RETURN_IF_NOT_FMT(ret == APP_ERR_OK, APP_ERR_INNER_ERROR, "copy labels to host failed:%d\n", ret);

    if (merge) {
        for (int i = 0; i < n; ++i) {
            std::vector<std::pair<float16_t, int64_t> > dist_label_pairs(indexSize * k);
            for (int id_index = 0; id_index < indexSize; ++id_index) {
                for (int id_k = 0; id_k < k; ++id_k) {
                    auto offset = id_index * n * k + i * k + id_k;
                    if (indexes[id_index]->isAddWithIds) {
                        dist_label_pairs[id_index * k + id_k] =
                            std::make_pair(multiIndexDists[offset],
                                           indexes[id_index]->GetIdMap().at(multiIndexLabels[offset]));
                    } else {
                        dist_label_pairs[id_index * k + id_k] =
                        std::make_pair(multiIndexDists[offset], multiIndexLabels[offset]);
                    }
                }
            }

            std::sort(dist_label_pairs.begin(), dist_label_pairs.end(),
                      [](const std::pair<float16_t, int64_t> &a, const std::pair<float16_t, int64_t> &b) {
                          return a.first < b.first;  // 按照距离升序排序
                      });

            for (int id_k = 0; id_k < k; ++id_k) {
                distances[i * k + id_k] = dist_label_pairs[id_k].first;
                labels[i * k + id_k] = dist_label_pairs[id_k].second;
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            for (int id_index = 0; id_index < indexSize; ++id_index) {
                for (int id_k = 0; id_k < k; ++id_k) {
                    auto offset = id_index * n * k + i * k + id_k;
                    if (indexes[id_index]->isAddWithIds) {
                        labels[offset] = indexes[id_index]->GetIdMap().at(labels[offset]);
                    }
                }
            }
        }
    }
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::SearchBatchImplL1(AscendTensor<float, DIMS_2> &queriesNpu,
                                            AscendTensor<float16_t, DIMS_2> &queryCodes,
                                            AscendTensor<uint16_t, DIMS_2> &l1KIndiceNpu)
{
    APP_LOG_INFO("NpuIndexIVFHSP SearchBatchImplL1 operation started.\n");
    auto &mem = resources->getMemoryManager();

    int batch = queriesNpu.getSize(0);
    AscendTensor<float16_t, DIMS_2> dists(mem, { batch, nList }, defaultStream);
    AscendTensor<uint32_t, DIMS_2> opSize(mem, { CORE_NUM, SIZE_ALIGN }, defaultStream);
    opSize[0][0] = nList;
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, { CORE_NUM, FLAG_SIZE }, defaultStream);
    std::vector<uint16_t> opFlagVec(CORE_NUM * FLAG_SIZE, 0);
    auto ret = aclrtMemcpy(opFlag.data(), opFlagVec.size() * sizeof(uint16_t), opFlagVec.data(),
                           opFlagVec.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    APPERR_RETURN_IF_NOT_LOG(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "Failed to copy attr to device");

    AscendTensor<float16_t, DIMS_2> l1KDistsNpu(mem, { batch, searchParam->nProbeL1 }, defaultStream);

    RunL1DistOp(batch, queriesNpu, *codeBooksShapedL1Npu, queryCodes, dists, opFlag);
    RunL1TopKOp(batch, dists, opSize, opFlag, *searchL1OpAttrs, l1KDistsNpu, l1KIndiceNpu);
    ret = aclrtSynchronizeStream(defaultStream);

    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR, "aclrtSynchronizeStream default stream: %i\n",
                             ret);
    ret = aclrtSynchronizeStream(aiCpuStream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream aicpu stream failed: %i\n", ret);

    APP_LOG_INFO("NpuIndexIVFHSP SearchBatchImplL1 operation end.\n");
    return APP_ERR_OK;
}
APP_ERROR NpuIndexIVFHSP::SearchBatchImplL2(AscendTensor<uint8_t, DIMS_1> &maskBitNpu,
                                            AscendTensor<float16_t, DIMS_2> &queryCodesNpu,
                                            AscendTensor<uint16_t, DIMS_2> &l1KIndicesNpu,
                                            AscendTensor<uint64_t, DIMS_2> &addressOffsetOfBucketL3,
                                            AscendTensor<uint64_t, DIMS_2> &idAdressL3)
{
    APP_LOG_INFO("NpuIndexIVFHSP SearchBatchImplL2 operation started.\n");
    auto &mem = resources->getMemoryManager();
    int nq = queryCodesNpu.getSize(0);

    // IvfSpL2DistOp Output
    AscendTensor<float16_t, DIMS_2> dists(mem, { nq, searchParam->nProbeL1 * nListL2 }, defaultStream);
    AscendTensor<float16_t, DIMS_2> distsRes(mem, { nq, searchParam->nProbeL2 }, defaultStream);
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, { CORE_NUM, FLAG_SIZE }, defaultStream);
    std::vector<uint16_t> opFlagVec(CORE_NUM * FLAG_SIZE, 0);
    auto ret = aclrtMemcpy(opFlag.data(), opFlag.getSizeInBytes(), opFlagVec.data(),
                           opFlagVec.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy opFlag to device failed %d", static_cast<int>(ret));

    RunL2DistOp(queryCodesNpu, *codeBooksShapedL2Npu, l1KIndicesNpu, dists, opFlag);
    RunL2TopKWithMaskOp(maskBitNpu, dists, l1KIndicesNpu, opFlag, *searchL2OpAttrs, distsRes, addressOffsetOfBucketL3,
                        idAdressL3);

    ret = aclrtSynchronizeStream(defaultStream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream aicore stream failed: %i\nq", ret);

    ret = aclrtSynchronizeStream(aiCpuStream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream aicpu stream failed: %d\n", ret);

    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::SearchBatchImplL2(AscendTensor<float16_t, DIMS_2> &queryCodesNpu,
                                            AscendTensor<uint16_t, DIMS_2> &l1KIndicesNpu,
                                            AscendTensor<uint64_t, DIMS_2> &addressOffsetOfBucketL3,
                                            AscendTensor<uint64_t, DIMS_2> &idAdressL3)
{
    APP_LOG_INFO("NpuIndexIVFHSP SearchBatchImplL2 operation started.\n");
    auto &mem = resources->getMemoryManager();
    int nq = queryCodesNpu.getSize(0);

    // IvfSpL2DistOp Output
    AscendTensor<float16_t, DIMS_2> dists(mem, { nq, searchParam->nProbeL1 * nListL2 }, defaultStream);
    AscendTensor<float16_t, DIMS_2> distsRes(mem, { nq, searchParam->nProbeL2 }, defaultStream);
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, { CORE_NUM, FLAG_SIZE }, defaultStream);
    std::vector<uint16_t> opFlagVec(CORE_NUM * FLAG_SIZE, 0);
    auto ret = aclrtMemcpy(opFlag.data(), opFlag.getSizeInBytes(), opFlagVec.data(),
                           opFlagVec.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy opFlag to device failed %d", static_cast<int>(ret));

    RunL2DistOp(queryCodesNpu, *codeBooksShapedL2Npu, l1KIndicesNpu, dists, opFlag);
    RunL2TopKOp(dists, l1KIndicesNpu, opFlag, *searchL2OpAttrs, distsRes, addressOffsetOfBucketL3, idAdressL3);
    ret = aclrtSynchronizeStream(defaultStream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream aicore stream failed: %i\nq", ret);

    ret = aclrtSynchronizeStream(aiCpuStream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream aicpu stream failed: %d\n", ret);

    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::SearchBatchImplMultiL2(AscendTensor<float16_t, DIMS_2> &queryCodesNpu,
                                                 AscendTensor<uint16_t, DIMS_2> &l1KIndicesNpu,
                                                 AscendTensor<uint64_t, DIMS_2> &indicesL2)
{
    auto &mem = resources->getMemoryManager();
    auto n = queryCodesNpu.getSize(0);
    AscendTensor<float16_t, DIMS_2> dists(mem, { n, searchParam->nProbeL1 * nListL2 }, defaultStream);
    AscendTensor<float16_t, DIMS_2> distsRes(mem, { n, searchParam->nProbeL2 }, aiCpuStream);
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, { CORE_NUM, FLAG_SIZE }, defaultStream);
    std::vector<uint16_t> opFlagVec(CORE_NUM * FLAG_SIZE, 0);
    auto ret = aclrtMemcpy(opFlag.data(), opFlag.getSizeInBytes(), opFlagVec.data(),
                           opFlagVec.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);

    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy opFlag to device failed %d", static_cast<int>(ret));
    RunL2DistOp(queryCodesNpu, *codeBooksShapedL2Npu, l1KIndicesNpu, dists, opFlag);
    RunMultiL2TopKOp(dists, l1KIndicesNpu, opFlag, *searchL2OpAttrs, distsRes, indicesL2);

    ret = aclrtSynchronizeStream(defaultStream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream aicore stream failed: %i\nq", ret);
    ret = aclrtSynchronizeStream(aiCpuStream);
    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream aicpu stream failed: %d\n", ret);
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::SearchBatchImplL3(AscendTensor<float16_t, DIMS_2> &queryCodes,
                                            AscendTensor<uint64_t, DIMS_2> &addressOffsetOfBucketL3,
                                            AscendTensor<uint64_t, DIMS_2> &idAddressOfBucketL3,
                                            AscendTensor<float16_t, DIMS_2> &outDists,
                                            AscendTensor<int64_t, DIMS_2> &outIndices)
{
    auto &mem = resources->getMemoryManager();
    int n = queryCodes.getSize(0);
    // tensor for operator flags
    AscendTensor<uint16_t, DIMS_2> opFlag(mem, { CORE_NUM, FLAG_SIZE }, defaultStream);
    std::vector<uint16_t> opFlagVec(CORE_NUM * FLAG_SIZE, 0);
    int ret = aclrtMemcpy(opFlag.data(), opFlag.getSizeInBytes(), opFlagVec.data(), opFlagVec.size() * sizeof(uint16_t),
                          ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy opFlag to device failed %d", static_cast<int>(ret));

    AscendTensor<float16_t, DIMS_2> distResult(mem, { n, searchParam->l3SegmentNum * BASE_SEG_SIZE }, defaultStream);
    AscendTensor<float16_t, DIMS_2> vcMinDistResult(mem, { n, 2 * searchParam->l3SegmentNum * 2 *
                                                           BASE_SEG_SIZE / VCMIN_SEG_SIZE }, defaultStream);
    std::vector<float16_t> vcMinDistResultVec(2 * n * searchParam->l3SegmentNum * 2 * BASE_SEG_SIZE / VCMIN_SEG_SIZE,
                                              0.0f);  // 赋值为0，为了后续非对齐搬出时利用原子加; 申请2倍应有内存，使atomicAdd不会产生越界错误
    ret = aclrtMemcpy(vcMinDistResult.data(), vcMinDistResult.getSizeInBytes(), vcMinDistResultVec.data(),
                      vcMinDistResultVec.size() * sizeof(float16_t), ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "copy vcMinDistResult to device failed %d", static_cast<int>(ret));

    // attr Tensor
    AscendTensor<int32_t, DIMS_2> attr_nlistL1(mem, { 1, this->nList }, defaultStream);
    AscendTensor<int32_t, DIMS_2> attr_nlistL2(mem, { 1, this->nListL2 }, defaultStream);
    AscendTensor<int32_t, DIMS_2> attr_segmentL3(mem, { 1, this->searchParam->l3SegmentNum }, defaultStream);

    // maskByteNpu
    if (this->maskFlag) {
        RunL3DistWithMaskOp(queryCodes, addressOffsetOfBucketL3, distResult, vcMinDistResult, opFlag, attr_nlistL1,
                            attr_nlistL2, attr_segmentL3);
    }
    if (!this->maskFlag) {
        RunL3DistOp(queryCodes, addressOffsetOfBucketL3, distResult, vcMinDistResult, opFlag, attr_nlistL1,
                    attr_nlistL2, attr_segmentL3);
    }

    RunL3TopKOp(distResult, vcMinDistResult, opFlag, idAddressOfBucketL3, *searchL3OpAttrs, outDists, outIndices);
    ret = aclrtSynchronizeStream(defaultStream);

    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream default stream failed: %d\n", ret);
    ret = aclrtSynchronizeStream(aiCpuStream);

    APPERR_RETURN_IF_NOT_FMT(ret == ACL_SUCCESS, APP_ERR_INNER_ERROR,
                             "aclrtSynchronizeStream aicpu stream failed: %d\n", ret);

    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::SearchBatchImplMultiL3(const std::vector<NpuIndexIVFHSP *> &indexes, int i,
                                                 AscendTensor<float16_t, DIMS_2> &queryCodes,
                                                 AscendTensor<uint64_t, DIMS_3> &addressOffsetOfBucketL3,
                                                 AscendTensor<uint64_t, DIMS_3> &,
                                                 AscendTensor<float16_t, DIMS_3> &distResult,
                                                 AscendTensor<float16_t, DIMS_3> &vcMinDistResult,
                                                 AscendTensor<uint16_t, DIMS_3> &opFlag)
{
    auto &mem = resources->getMemoryManager();

    AscendTensor<int32_t, DIMS_2> attr_nlistL1(mem, { 1, this->nList }, defaultStream);
    AscendTensor<int32_t, DIMS_2> attr_nlistL2(mem, { 1, this->nListL2 }, defaultStream);
    AscendTensor<int32_t, DIMS_2> attr_segmentL3(mem, { 1, this->searchParam->l3SegmentNum }, defaultStream);

    if (this->maskFlag) {
        RunMultiL3DistWithMaskOp(indexes, i, queryCodes, addressOffsetOfBucketL3, distResult, vcMinDistResult, opFlag,
                                 attr_nlistL1, attr_nlistL2, attr_segmentL3);
    }
    if (!this->maskFlag) {
        RunMultiL3DistOp(indexes, i, queryCodes, addressOffsetOfBucketL3, distResult, vcMinDistResult, opFlag,
                         attr_nlistL1, attr_nlistL2, attr_segmentL3);
    }
    return APP_ERR_OK;
}

/**********************************************************************************************
 * Ascend Operators
 **********************************************************************************************/
void NpuIndexIVFHSP::RunL1TopKOp(int batch, AscendTensor<float16_t, DIMS_2> &dists,
                                 AscendTensor<uint32_t, DIMS_2> &opSize, AscendTensor<uint16_t, DIMS_2> &opFlag,
                                 AscendTensor<int64_t, DIMS_1> &attrs, AscendTensor<float16_t, DIMS_2> &distResult,
                                 AscendTensor<uint16_t, DIMS_2> &labelResult)
{
    APP_LOG_INFO("NpuIndexIVFHSP RunL1TopKOp operation started.\n");
    AscendOperator *l1TopKOp = nullptr;
    if (l1TopKOps.find(batch) != l1TopKOps.end()) {
        l1TopKOp = l1TopKOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(l1TopKOp);
    std::vector<const aclDataBuffer *> topkOpInput;
    topkOpInput.emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(opSize.data(), opSize.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(attrs.data(), attrs.getSizeInBytes()));

    std::vector<aclDataBuffer *> topkOpOutput;
    topkOpOutput.emplace_back(aclCreateDataBuffer(distResult.data(), distResult.getSizeInBytes()));
    topkOpOutput.emplace_back(aclCreateDataBuffer(labelResult.data(), labelResult.getSizeInBytes()));

    l1TopKOp->exec(topkOpInput, topkOpOutput, aiCpuStream);
    for (auto &item : topkOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpInput.clear();

    for (auto &item : topkOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpOutput.clear();
}

void NpuIndexIVFHSP::RunL1DistOp(int batch, AscendTensor<float, DIMS_2> &query,
                                 AscendTensor<float16_t, DIMS_4> &,
                                 AscendTensor<float16_t, DIMS_2> &queryCode, AscendTensor<float16_t, DIMS_2> &dists,
                                 AscendTensor<uint16_t, DIMS_2> &flag)
{
#ifdef USE_ACL_NN_INTERFACE
    auto t0 = std::chrono::high_resolution_clock::now();
    aclOpExecutor *l1DistOp = nullptr;
    size_t workspaceSize = 0;
    std::vector<int64_t> queryShape({ batch, this->dim });
    std::vector<int64_t> coarseCentroidsShape({ utils::divUp(this->subSpaceDimL1 * this->nList, CUBE_ALIGN),
                                                utils::divUp(this->dim, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> queryCodeShape({ batch, this->nList * this->subSpaceDimL1 });
    std::vector<int64_t> distResultShape({ batch, this->nList });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

    aclTensor *queryTensor = aclCreateTensor(queryShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             queryShape.data(), 2, query.data());
    aclTensor *dataShapedTensor = aclCreateTensor(coarseCentroidsShape.data(), 4, ACL_FLOAT16, nullptr, 0,
                                                  ACL_FORMAT_ND, coarseCentroidsShape.data(), 4,
                                                  codeBooksShapedL1Npu->data());
    aclTensor *queryCodeTensor = aclCreateTensor(queryCodeShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                                 queryCodeShape.data(), 2, queryCode.data());
    aclTensor *distsTensor = aclCreateTensor(distResultShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             distResultShape.data(), 2, dists.data());
    aclTensor *flagTensor = aclCreateTensor(flagShape.data(), 2, ACL_UINT16, nullptr, 0, ACL_FORMAT_ND,
                                            flagShape.data(), 2, flag.data());
    auto ret = aclnnVstarComputeL1GetWorkspaceSize(queryTensor, dataShapedTensor, subSpaceDimL1, queryCodeTensor,
                                                   distsTensor, flagTensor, &workspaceSize, &l1DistOp);
    ACL_REQUIRE_OK(ret);
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        ACL_REQUIRE_OK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto delta_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    printf("======aclnnVstarComputeL1GetWorkspaceSize Time Spend: %f ms\n", delta_t1 / 1000.0);
    ASCEND_THROW_IF_NOT(l1DistOp);
    ret = aclnnVstarComputeL1(nullptr, 0, l1DistOp, defaultStream);
    ACL_REQUIRE_OK(ret);
    (void)aclDestroyTensor(queryTensor);
    (void)aclDestroyTensor(dataShapedTensor);
    (void)aclDestroyTensor(queryCodeTensor);
    (void)aclDestroyTensor(distsTensor);
    (void)aclDestroyTensor(flagTensor);
#else
    AscendOperator *l1DistOp = nullptr;
    if (l1DistOps.find(batch) != l1DistOps.end()) {
        l1DistOp = l1DistOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(l1DistOp);
    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(query.data(), query.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(codeBooksShapedL1Npu->data(), codeBooksShapedL1Npu->getSizeInBytes()));

    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(queryCode.data(), queryCode.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    l1DistOp->exec(distOpInput, distOpOutput, defaultStream);

    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();

    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();

#endif
}

void NpuIndexIVFHSP::RunL2TopKWithMaskOp(AscendTensor<uint8_t, DIMS_1> &maskBitNpu,
                                         AscendTensor<float16_t, DIMS_2> &dists,
                                         AscendTensor<uint16_t, DIMS_2> &l1Indices,
                                         AscendTensor<uint16_t, DIMS_2> &opFlag, AscendTensor<int64_t, DIMS_1> &attr,
                                         AscendTensor<float16_t, DIMS_2> &distsRes,
                                         AscendTensor<uint64_t, DIMS_2> &addressOffset,
                                         AscendTensor<uint64_t, DIMS_2> &idAddress)
{
    APP_LOG_INFO("NpuIndexIVFHSP RunL2TopKWithMaskOp operation started.\n");
    AscendOperator *op = nullptr;
    int batch = dists.getSize(0);
    if (l2TopKWithMaskOps.find(batch) != l2TopKWithMaskOps.end()) {
        op = l2TopKWithMaskOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    // prepare for input data's buffer
    std::vector<const aclDataBuffer *> topkOpInput;

    topkOpInput.emplace_back(aclCreateDataBuffer(maskBitNpu.data(), maskBitNpu.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(l1Indices.data(), l1Indices.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    topkOpInput.emplace_back(
        aclCreateDataBuffer(addressOffsetOfBucket->data(), addressOffsetOfBucket->getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(attr.data(), attr.getSizeInBytes()));

    // prepare for output data's buffer
    std::vector<aclDataBuffer *> topkOpOutput;
    topkOpOutput.emplace_back(aclCreateDataBuffer(distsRes.data(), distsRes.getSizeInBytes()));
    topkOpOutput.emplace_back(aclCreateDataBuffer(addressOffset.data(), addressOffset.getSizeInBytes()));
    topkOpOutput.emplace_back(aclCreateDataBuffer(idAddress.data(), idAddress.getSizeInBytes()));
    topkOpOutput.emplace_back(aclCreateDataBuffer(maskByteNpu->data(), BASE_SEG_SIZE * sizeof(uint8_t)));
    topkOpOutput.emplace_back(aclCreateDataBuffer(isMaskOffset->data(), nList * nListL2 * 2 * sizeof(uint64_t)));

    // async executing operator
    op->exec(topkOpInput, topkOpOutput, aiCpuStream);

    for (auto &item : topkOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpInput.clear();

    for (auto &item : topkOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpOutput.clear();
    APP_LOG_INFO("NpuIndexIVFHSP RunL2TopKWithMaskOp operation end.\n");
}

void NpuIndexIVFHSP::RunL2TopKOp(AscendTensor<float16_t, DIMS_2> &dists, AscendTensor<uint16_t, DIMS_2> &l1Indices,
                                 AscendTensor<uint16_t, DIMS_2> &opFlag, AscendTensor<int64_t, DIMS_1> &attr,
                                 AscendTensor<float16_t, DIMS_2> &distsRes,
                                 AscendTensor<uint64_t, DIMS_2> &addressOffset,
                                 AscendTensor<uint64_t, DIMS_2> &idAddress)
{
    APP_LOG_INFO("NpuIndexIVFHSP RunL2TopKOp operation started.\n");
    AscendOperator *op = nullptr;
    int batch = dists.getSize(0);
    if (l2TopKOps.find(batch) != l2TopKOps.end()) {
        op = l2TopKOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    // prepare for input data's buffer
    std::vector<const aclDataBuffer *> topkOpInput;

    topkOpInput.emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(l1Indices.data(), l1Indices.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    topkOpInput.emplace_back(
        aclCreateDataBuffer(addressOffsetOfBucket->data(), addressOffsetOfBucket->getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(attr.data(), attr.getSizeInBytes()));

    // prepare for output data's buffer
    std::vector<aclDataBuffer *> topkOpOutput;
    topkOpOutput.emplace_back(aclCreateDataBuffer(distsRes.data(), distsRes.getSizeInBytes()));
    topkOpOutput.emplace_back(aclCreateDataBuffer(addressOffset.data(), addressOffset.getSizeInBytes()));
    topkOpOutput.emplace_back(aclCreateDataBuffer(idAddress.data(), idAddress.getSizeInBytes()));

    // async executing operator
    op->exec(topkOpInput, topkOpOutput, aiCpuStream);

    for (auto &item : topkOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpInput.clear();

    for (auto &item : topkOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpOutput.clear();
    APP_LOG_INFO("NpuIndexIVFHSP RunL2TopKOp operation end.\n");
}

void NpuIndexIVFHSP::RunMultiL2TopKOp(AscendTensor<float16_t, DIMS_2> &dists, AscendTensor<uint16_t, DIMS_2> &l1Indices,
                                      AscendTensor<uint16_t, DIMS_2> &opFlag, AscendTensor<int64_t, DIMS_1> &attr,
                                      AscendTensor<float16_t, DIMS_2> &distsRes,
                                      AscendTensor<uint64_t, DIMS_2> &labelRes)
{
    APP_LOG_INFO("NpuMultiIndexIVFHSP RunL2TopKOp operation started.\n");
    AscendOperator *op = nullptr;
    int batch = dists.getSize(0);
    if (l2MultiTopKOps.find(batch) != l2MultiTopKOps.end()) {
        op = l2MultiTopKOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    // prepare for input data's buffer
    std::vector<const aclDataBuffer *> topkOpInput;

    topkOpInput.emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(l1Indices.data(), l1Indices.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(attr.data(), attr.getSizeInBytes()));

    // prepare for output data's buffer
    std::vector<aclDataBuffer *> topkOpOutput;
    topkOpOutput.emplace_back(aclCreateDataBuffer(distsRes.data(), distsRes.getSizeInBytes()));
    topkOpOutput.emplace_back(aclCreateDataBuffer(labelRes.data(), labelRes.getSizeInBytes()));

    // async executing operator
    op->exec(topkOpInput, topkOpOutput, aiCpuStream);

    for (auto &item : topkOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpInput.clear();

    for (auto &item : topkOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpOutput.clear();
    APP_LOG_INFO("NpuIndexIVFHSP RunMultiL2TopKOp operation end.\n");
}

void NpuIndexIVFHSP::RunL2DistOp(AscendTensor<float16_t, DIMS_2> &queryCodes,
                                 AscendTensor<float16_t, DIMS_4> &codebookL2, AscendTensor<uint16_t, DIMS_2> &l1Indices,
                                 AscendTensor<float16_t, DIMS_2> &outDists, AscendTensor<uint16_t, DIMS_2> &flag)
{
    APP_LOG_INFO("NpuIndexIVFHSP RunL2DistOp operation started.\n");
#ifdef USE_ACL_NN_INTERFACE
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t workspaceSize = 0;
    aclOpExecutor *handle = nullptr;
    std::vector<int64_t> queryShape({ queryCodes.getSize(0), queryCodes.getSize(1) });
    std::vector<int64_t> codebookL2Shape({ codebookL2.getSize(0), codebookL2.getSize(1), CUBE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> l1IndicesShape({ l1Indices.getSize(0), this->searchParam->nProbeL1 });
    std::vector<int64_t> distResultShape({ outDists.getSize(0), this->nListL2 * this->searchParam->nProbeL1 });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

    aclTensor *queryTensor = aclCreateTensor(queryShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             queryShape.data(), 2, queryCodes.data());
    aclTensor *dataShapedTensor = aclCreateTensor(codebookL2Shape.data(), 4, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                                  codebookL2Shape.data(), 4, codebookL2.data());
    aclTensor *l1IndicesTensor = aclCreateTensor(l1IndicesShape.data(), 2, ACL_UINT16, nullptr, 0, ACL_FORMAT_ND,
                                                 l1IndicesShape.data(), 2, l1Indices.data());
    aclTensor *distsTensor = aclCreateTensor(distResultShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             distResultShape.data(), 2, outDists.data());
    aclTensor *flagTensor = aclCreateTensor(flagShape.data(), 2, ACL_UINT16, nullptr, 0, ACL_FORMAT_ND,
                                            flagShape.data(), 2, flag.data());
    auto ret = aclnnVstarComputeL2GetWorkspaceSize(queryTensor, dataShapedTensor, l1IndicesTensor, subSpaceDimL1,
                                                   subSpaceDimL2, distsTensor, flagTensor, &workspaceSize, &handle);
    ACL_REQUIRE_OK(ret);
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        ACL_REQUIRE_OK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto delta_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    printf("======aclnnVstarComputeL2GetWorkspaceSize Time Spend: %f ms\n", delta_t1 / 1000.0);
    ret = aclnnVstarComputeL2(workspace, workspaceSize, handle, defaultStream);

    (void)aclDestroyTensor(queryTensor);
    (void)aclDestroyTensor(dataShapedTensor);
    (void)aclDestroyTensor(l1IndicesTensor);
    (void)aclDestroyTensor(distsTensor);
    (void)aclDestroyTensor(flagTensor);

    ACL_REQUIRE_OK(ret);
#else
    int batch = queryCodes.getSize(0);
    AscendOperator *l2DistOp = nullptr;
    if (l2DistOps.find(batch) != l2DistOps.end()) {
        l2DistOp = l2DistOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(l2DistOp);
    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(queryCodes.data(), queryCodes.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(codebookL2.data(), codebookL2.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(l1Indices.data(), l1Indices.getSizeInBytes()));

    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(outDists.data(), outDists.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    l2DistOp->exec(distOpInput, distOpOutput, defaultStream);
    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();

    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();
    APP_LOG_INFO("NpuIndexIVFHSP RunL2DistOp operation end.\n");
#endif
}

void NpuIndexIVFHSP::RunL3DistOp(AscendTensor<float16_t, DIMS_2> &queryCodes,
                                 AscendTensor<uint64_t, DIMS_2> &addressOffsetOfBucketL3,
                                 AscendTensor<float16_t, DIMS_2> &dists, AscendTensor<float16_t, DIMS_2> &distsVcMin,
                                 AscendTensor<uint16_t, DIMS_2> &opFlag, AscendTensor<int32_t, DIMS_2> &attr_nlistL1,
                                 AscendTensor<int32_t, DIMS_2> &attr_nlistL2,
                                 AscendTensor<int32_t, DIMS_2> &attr_segmentL3)
{
    APP_LOG_INFO("NpuIndexIVFHSP RunL3DistOp operation started.\n");
    int batch = queryCodes.getSize(0);
#ifdef USE_ACL_NN_INTERFACE
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t workspaceSize = 0;
    aclOpExecutor *handle = nullptr;
    std::vector<int64_t> queryCodeShape({ batch, this->nList * this->subSpaceDimL1 });
    std::vector<int64_t> codeWordsShape({ utils::divUp(static_cast<int>(this->ntotal), CUBE_ALIGN),
                                          utils::divUp(this->subSpaceDimL2, CUBE_ALIGN), CUBE_ALIGN,
                                          CUBE_ALIGN });

    std::vector<int64_t> addressOffset({ batch, this->searchParam->nProbeL2 * 6 });
    std::vector<int64_t> diff1Shape({ 1, this->subSpaceDimL2 });
    std::vector<int64_t> diff2Shape({ 1, this->subSpaceDimL2 });
    std::vector<int64_t> normL2Shape({ 1, static_cast<int>(this->ntotal) });

    std::vector<int64_t> outDistShape({ batch, this->searchParam->l3SegmentNum * BASE_SEG_SIZE });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });
    std::vector<int64_t> vcMinShape({ batch, 2 * this->searchParam->l3SegmentNum * BASE_SEG_SIZE / VCMIN_SEG_SIZE });

    aclTensor *queryCodeTensor = aclCreateTensor(queryCodeShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                                 queryCodeShape.data(), 2, queryCodes.data());
    aclTensor *codeWordsTensor = aclCreateTensor(codeWordsShape.data(), 4, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND,
                                                 codeWordsShape.data(), 4, minAddressOfBaseNpu);
    aclTensor *addressOffsetTensor = aclCreateTensor(addressOffset.data(), 2, ACL_UINT64, nullptr, 0, ACL_FORMAT_ND,
                                                     addressOffset.data(), 2, addressOffsetOfBucketL3.data());
    aclTensor *diff1Tensor = aclCreateTensor(diff1Shape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             diff1Shape.data(), 2, vDiffNpu->data());
    aclTensor *diff2Tensor = aclCreateTensor(diff2Shape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             diff2Shape.data(), 2, vDiff2Npu->data());
    aclTensor *normL2Tensor = aclCreateTensor(normL2Shape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                              normL2Shape.data(), 2, precomputeNormL2Npu->data());

    aclTensor *outDistsTensor = aclCreateTensor(outDistShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                                outDistShape.data(), 2, dists.data());
    aclTensor *flagTensor = aclCreateTensor(flagShape.data(), 2, ACL_UINT16, nullptr, 0, ACL_FORMAT_ND,
                                            flagShape.data(), 2, opFlag.data());
    aclTensor *distsVcMinTensor = aclCreateTensor(vcMinShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                                  vcMinShape.data(), 2, distsVcMin.data());
    auto ret = aclnnVSC3GetWorkspaceSize(queryCodeTensor, codeWordsTensor, addressOffsetTensor,
                                         diff1Tensor, diff2Tensor, normL2Tensor, this->nList,
                                         this->nListL2, this->searchParam->l3SegmentNum, outDistsTensor,
                                         flagTensor, distsVcMinTensor, &workspaceSize, &handle);
    ACL_REQUIRE_OK(ret);
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        ACL_REQUIRE_OK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto delta_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    printf("======aclnnVSC3GetWorkspaceSize Time Spend: %f ms\n", delta_t1 / 1000.0);
    ret = aclnnVSC3(workspace, workspaceSize, handle, defaultStream);

    (void)aclDestroyTensor(queryCodeTensor);
    (void)aclDestroyTensor(codeWordsTensor);
    (void)aclDestroyTensor(addressOffsetTensor);
    (void)aclDestroyTensor(diff1Tensor);
    (void)aclDestroyTensor(diff2Tensor);
    (void)aclDestroyTensor(normL2Tensor);
    (void)aclDestroyTensor(outDistsTensor);
    (void)aclDestroyTensor(flagTensor);
    (void)aclDestroyTensor(distsVcMinTensor);
    ACL_REQUIRE_OK(ret);

#else
    AscendOperator *l3DistOp = nullptr;
    if (l3DistOps.find(batch) != l3DistOps.end()) {
        l3DistOp = l3DistOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(l3DistOp);
    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(queryCodes.data(), queryCodes.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(minAddressOfBaseNpu, BASE_SEG_SIZE * this->dimStored));
    distOpInput.emplace_back(
        aclCreateDataBuffer(addressOffsetOfBucketL3.data(), addressOffsetOfBucketL3.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(vDiffNpu->data(), vDiffNpu->getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(vDiff2Npu->data(), vDiff2Npu->getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(precomputeNormL2Npu->data(), BASE_SEG_SIZE * sizeof(float16_t)));
    distOpInput.emplace_back(aclCreateDataBuffer(attr_nlistL1.data(), attr_nlistL1.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(attr_nlistL2.data(), attr_nlistL2.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(attr_segmentL3.data(), attr_segmentL3.getSizeInBytes()));

    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(distsVcMin.data(), distsVcMin.getSizeInBytes()));

    l3DistOp->exec(distOpInput, distOpOutput, defaultStream);
    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();

    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();
#endif
    APP_LOG_INFO("NpuIndexIVFHSP RunL3DistOp operation end.\n");
}

void NpuIndexIVFHSP::RunMultiL3DistOp(const std::vector<NpuIndexIVFHSP *> &indexes, int i,
                                      AscendTensor<float16_t, DIMS_2> &queryCodes,
                                      AscendTensor<uint64_t, DIMS_3> &addressOffsetOfBucketL3,
                                      AscendTensor<float16_t, DIMS_3> &dists,
                                      AscendTensor<float16_t, DIMS_3> &distsVcMin,
                                      AscendTensor<uint16_t, DIMS_3> &opFlag,
                                      AscendTensor<int32_t, DIMS_2> &attr_nlistL1,
                                      AscendTensor<int32_t, DIMS_2> &attr_nlistL2,
                                      AscendTensor<int32_t, DIMS_2> &attr_segmentL3)
{
    APP_LOG_INFO("NpuIndexIVFHSP RunL3DistOp operation started.\n");
    int batch = queryCodes.getSize(0);
    size_t indexSize = indexes.size();
#ifdef USE_ACL_NN_INTERFACE
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t workspaceSize = 0;
    aclOpExecutor *handle = nullptr;
    std::vector<int64_t> queryCodeShape({ batch, this->nList * this->subSpaceDimL1 });
    std::vector<int64_t> codeWordsShape({ utils::divUp(static_cast<int>(this->ntotal), CUBE_ALIGN),
                                          utils::divUp(this->subSpaceDimL2, CUBE_ALIGN), CUBE_ALIGN,
                                          CUBE_ALIGN });

    std::vector<int64_t> addressOffset({ batch, this->searchParam->nProbeL2 * 6 });
    std::vector<int64_t> diff1Shape({ 1, this->subSpaceDimL2 });
    std::vector<int64_t> diff2Shape({ 1, this->subSpaceDimL2 });
    std::vector<int64_t> normL2Shape({ 1, static_cast<int>(this->ntotal) });

    std::vector<int64_t> outDistShape({ batch, this->searchParam->l3SegmentNum * BASE_SEG_SIZE });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });
    std::vector<int64_t> vcMinShape({ batch, 2 * this->searchParam->l3SegmentNum * BASE_SEG_SIZE / VCMIN_SEG_SIZE });

    aclTensor *queryCodeTensor = aclCreateTensor(queryCodeShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                                 queryCodeShape.data(), 2, queryCodes.data());
    aclTensor *codeWordsTensor = aclCreateTensor(codeWordsShape.data(), 4, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND,
                                                 codeWordsShape.data(), 4, minAddressOfBaseNpu);
    aclTensor *addressOffsetTensor = aclCreateTensor(addressOffset.data(), 2, ACL_UINT64, nullptr, 0, ACL_FORMAT_ND,
                                                     addressOffset.data(), 2, addressOffsetOfBucketL3.data());
    aclTensor *diff1Tensor = aclCreateTensor(diff1Shape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             diff1Shape.data(), 2, vDiffNpu->data());
    aclTensor *diff2Tensor = aclCreateTensor(diff2Shape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             diff2Shape.data(), 2, vDiff2Npu->data());
    aclTensor *normL2Tensor = aclCreateTensor(normL2Shape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                              normL2Shape.data(), 2, precomputeNormL2Npu->data());

    aclTensor *outDistsTensor = aclCreateTensor(outDistShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                                outDistShape.data(), 2, dists.data());
    aclTensor *flagTensor = aclCreateTensor(flagShape.data(), 2, ACL_UINT16, nullptr, 0, ACL_FORMAT_ND,
                                            flagShape.data(), 2, opFlag.data());
    aclTensor *distsVcMinTensor = aclCreateTensor(vcMinShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                                  vcMinShape.data(), 2, distsVcMin.data());
    auto ret = aclnnVSC3GetWorkspaceSize(queryCodeTensor, codeWordsTensor, addressOffsetTensor,
                                         diff1Tensor, diff2Tensor, normL2Tensor, this->nList,
                                         this->nListL2, this->searchParam->l3SegmentNum, outDistsTensor,
                                         flagTensor, distsVcMinTensor, &workspaceSize, &handle);
    ACL_REQUIRE_OK(ret);
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        ACL_REQUIRE_OK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto delta_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    printf("======aclnnVSC3GetWorkspaceSize Time Spend: %f ms\n", delta_t1 / 1000.0);
    ret = aclnnVSC3(workspace, workspaceSize, handle, defaultStream);

    (void)aclDestroyTensor(queryCodeTensor);
    (void)aclDestroyTensor(codeWordsTensor);
    (void)aclDestroyTensor(addressOffsetTensor);
    (void)aclDestroyTensor(diff1Tensor);
    (void)aclDestroyTensor(diff2Tensor);
    (void)aclDestroyTensor(normL2Tensor);
    (void)aclDestroyTensor(outDistsTensor);
    (void)aclDestroyTensor(flagTensor);
    (void)aclDestroyTensor(distsVcMinTensor);
    ACL_REQUIRE_OK(ret);

#else
    AscendOperator *l3DistOp = nullptr;
    if (l3DistOps.find(batch) != l3DistOps.end()) {
        l3DistOp = l3DistOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(l3DistOp);
    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(queryCodes.data(), queryCodes.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(indexes[i]->minAddressOfBaseNpu, BASE_SEG_SIZE * this->dimStored));
    distOpInput.emplace_back(
        aclCreateDataBuffer(addressOffsetOfBucketL3.data() + i * (addressOffsetOfBucketL3.numElements() / indexSize),
                            addressOffsetOfBucketL3.getSizeInBytes() / indexSize));
    distOpInput.emplace_back(aclCreateDataBuffer(indexes[i]->vDiffNpu->data(), indexes[i]->vDiffNpu->getSizeInBytes()));
    distOpInput.emplace_back(
        aclCreateDataBuffer(indexes[i]->vDiff2Npu->data(), indexes[i]->vDiff2Npu->getSizeInBytes()));
    distOpInput.emplace_back(
        aclCreateDataBuffer(indexes[i]->precomputeNormL2Npu->data(), BASE_SEG_SIZE * sizeof(float16_t)));
    distOpInput.emplace_back(aclCreateDataBuffer(attr_nlistL1.data(), attr_nlistL1.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(attr_nlistL2.data(), attr_nlistL2.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(attr_segmentL3.data(), attr_segmentL3.getSizeInBytes()));

    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(
        aclCreateDataBuffer(dists.data() + i * (dists.numElements() / indexSize), dists.getSizeInBytes() / indexSize));
    distOpOutput.emplace_back(aclCreateDataBuffer(opFlag.data() + i * (opFlag.numElements() / indexSize),
                                                  opFlag.getSizeInBytes() / indexSize));
    distOpOutput.emplace_back(aclCreateDataBuffer(distsVcMin.data() + i * (distsVcMin.numElements() / indexSize),
                                                  distsVcMin.getSizeInBytes() / indexSize));

    l3DistOp->exec(distOpInput, distOpOutput, defaultStream);
    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();
    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();
#endif
    APP_LOG_INFO("NpuIndexIVFHSP RunMultiL3DistOp operation end.\n");
}

void NpuIndexIVFHSP::RunMultiL3DistWithMaskOp(const std::vector<NpuIndexIVFHSP *> &indexes, int i,
                                              AscendTensor<float16_t, DIMS_2> &queryCodes,
                                              AscendTensor<uint64_t, DIMS_3> &addressOffsetOfBucketL3,
                                              AscendTensor<float16_t, DIMS_3> &dists,
                                              AscendTensor<float16_t, DIMS_3> &distsVcMin,
                                              AscendTensor<uint16_t, DIMS_3> &opFlag,
                                              AscendTensor<int32_t, DIMS_2> &attr_nlistL1,
                                              AscendTensor<int32_t, DIMS_2> &attr_nlistL2,
                                              AscendTensor<int32_t, DIMS_2> &attr_segmentL3)
{
    APP_LOG_INFO("NpuIndexIVFHSP RunMultiL3DistWithMaskOp operation started.\n");
    int batch = queryCodes.getSize(0);
    size_t indexSize = indexes.size();
#ifdef USE_ACL_NN_INTERFACE
    auto t0 = std::chrono::high_resolution_clock::now();
size_t workspaceSize = 0;
aclOpExecutor *handle = nullptr;
std::vector<int64_t> queryCodeShape({ batch, this->nList * this->subSpaceDimL1 });
std::vector<int64_t> codeWordsShape({ utils::divUp(static_cast<int>(this->ntotal), CUBE_ALIGN),
                                      utils::divUp(this->subSpaceDimL2, CUBE_ALIGN), CUBE_ALIGN,
                                      CUBE_ALIGN });  // todo
 
std::vector<int64_t> addressOffset({ batch, this->searchParam->nProbeL2 * 6 });
std::vector<int64_t> diff1Shape({ 1, this->subSpaceDimL2 });
std::vector<int64_t> diff2Shape({ 1, this->subSpaceDimL2 });
std::vector<int64_t> normL2Shape({ 1, static_cast<int>(this->ntotal) });  // todo
 
std::vector<int64_t> outDistShape({ batch, this->searchParam->l3SegmentNum * BASE_SEG_SIZE });
std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });
std::vector<int64_t> vcMinShape({ batch, 2 * this->searchParam->l3SegmentNum * BASE_SEG_SIZE / VCMIN_SEG_SIZE });
 
aclTensor *queryCodeTensor = aclCreateTensor(queryCodeShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             queryCodeShape.data(), 2, queryCodes.data());
aclTensor *codeWordsTensor = aclCreateTensor(codeWordsShape.data(), 4, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND,
                                             codeWordsShape.data(), 4, minAddressOfBaseNpu);
aclTensor *addressOffsetTensor = aclCreateTensor(addressOffset.data(), 2, ACL_UINT64, nullptr, 0, ACL_FORMAT_ND,
                                                 addressOffset.data(), 2, addressOffsetOfBucketL3.data());
aclTensor *diff1Tensor = aclCreateTensor(diff1Shape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                         diff1Shape.data(), 2, vDiffNpu->data());
aclTensor *diff2Tensor = aclCreateTensor(diff2Shape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                         diff2Shape.data(), 2, vDiff2Npu->data());
aclTensor *normL2Tensor = aclCreateTensor(normL2Shape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                          normL2Shape.data(), 2, precomputeNormL2Npu->data());
 
aclTensor *outDistsTensor = aclCreateTensor(outDistShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                            outDistShape.data(), 2, dists.data());
aclTensor *flagTensor = aclCreateTensor(flagShape.data(), 2, ACL_UINT16, nullptr, 0, ACL_FORMAT_ND,
                                        flagShape.data(), 2, opFlag.data());
aclTensor *distsVcMinTensor = aclCreateTensor(vcMinShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                              vcMinShape.data(), 2, distsVcMin.data());
auto ret = aclnnVSC3GetWorkspaceSize(queryCodeTensor, codeWordsTensor, addressOffsetTensor,
                                     diff1Tensor, diff2Tensor, normL2Tensor, this->nList,
                                     this->nListL2, this->searchParam->l3SegmentNum, outDistsTensor,
                                     flagTensor, distsVcMinTensor, &workspaceSize, &handle);
ACL_REQUIRE_OK(ret);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_REQUIRE_OK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY));
}
auto t1 = std::chrono::high_resolution_clock::now();
auto delta_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
printf("======aclnnVSC3GetWorkspaceSize Time Spend: %f ms\n", delta_t1 / 1000.0);
ret = aclnnVSC3(workspace, workspaceSize, handle, defaultStream);
 
(void)aclDestroyTensor(queryCodeTensor);
(void)aclDestroyTensor(codeWordsTensor);
(void)aclDestroyTensor(addressOffsetTensor);
(void)aclDestroyTensor(diff1Tensor);
(void)aclDestroyTensor(diff2Tensor);
(void)aclDestroyTensor(normL2Tensor);
(void)aclDestroyTensor(outDistsTensor);
(void)aclDestroyTensor(flagTensor);
(void)aclDestroyTensor(distsVcMinTensor);
ACL_REQUIRE_OK(ret);
 
#else
    AscendOperator *l3DistOp = nullptr;
    if (l3DistWithMaskOps.find(batch) != l3DistWithMaskOps.end()) {
        l3DistOp = l3DistWithMaskOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(l3DistOp);
    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(queryCodes.data(), queryCodes.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(indexes[i]->minAddressOfBaseNpu, BASE_SEG_SIZE * this->dimStored));
    distOpInput.emplace_back(
        aclCreateDataBuffer(addressOffsetOfBucketL3.data() + i * (addressOffsetOfBucketL3.numElements() / indexSize),
                            addressOffsetOfBucketL3.getSizeInBytes() / indexSize));
    distOpInput.emplace_back(aclCreateDataBuffer(indexes[i]->vDiffNpu->data(), indexes[i]->vDiffNpu->getSizeInBytes()));
    distOpInput.emplace_back(
        aclCreateDataBuffer(indexes[i]->vDiff2Npu->data(), indexes[i]->vDiff2Npu->getSizeInBytes()));
    distOpInput.emplace_back(
        aclCreateDataBuffer(indexes[i]->precomputeNormL2Npu->data(), BASE_SEG_SIZE * sizeof(float16_t)));
    distOpInput.emplace_back(aclCreateDataBuffer(indexes[i]->maskByteNpu->data(), BASE_SEG_SIZE * sizeof(uint8_t)));
    distOpInput.emplace_back(aclCreateDataBuffer(attr_nlistL1.data(), attr_nlistL1.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(attr_nlistL2.data(), attr_nlistL2.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(attr_segmentL3.data(), attr_segmentL3.getSizeInBytes()));
 
    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(
        aclCreateDataBuffer(dists.data() + i * (dists.numElements() / indexSize), dists.getSizeInBytes() / indexSize));
    distOpOutput.emplace_back(aclCreateDataBuffer(opFlag.data() + i * (opFlag.numElements() / indexSize),
                                                  opFlag.getSizeInBytes() / indexSize));
    distOpOutput.emplace_back(aclCreateDataBuffer(distsVcMin.data() + i * (distsVcMin.numElements() / indexSize),
                                                  distsVcMin.getSizeInBytes() / indexSize));
 
    l3DistOp->exec(distOpInput, distOpOutput, defaultStream);
    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();
    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();
#endif
    APP_LOG_INFO("NpuIndexIVFHSP RunMultiL3DistWithMaskOp operation end.\n");
}


void NpuIndexIVFHSP::RunL3DistWithMaskOp(AscendTensor<float16_t, DIMS_2> &queryCodes,
                                         AscendTensor<uint64_t, DIMS_2> &addressOffsetOfBucketL3,
                                         AscendTensor<float16_t, DIMS_2> &dists,
                                         AscendTensor<float16_t, DIMS_2> &distsVcMin,
                                         AscendTensor<uint16_t, DIMS_2> &opFlag,
                                         AscendTensor<int32_t, DIMS_2> &attr_nlistL1,
                                         AscendTensor<int32_t, DIMS_2> &attr_nlistL2,
                                         AscendTensor<int32_t, DIMS_2> &attr_segmentL3)
{
    APP_LOG_INFO("NpuIndexIVFHSP RunL3DistWithMaskOp operation started.\n");
    int batch = queryCodes.getSize(0);
#ifdef USE_ACL_NN_INTERFACE
    auto t0 = std::chrono::high_resolution_clock::now();
    size_t workspaceSize = 0;
    aclOpExecutor *handle = nullptr;
    std::vector<int64_t> queryCodeShape({ batch, this->nList * this->subSpaceDimL1 });
    std::vector<int64_t> codeWordsShape({ utils::divUp(static_cast<int>(this->ntotal), CUBE_ALIGN),
                                          utils::divUp(this->subSpaceDimL2, CUBE_ALIGN), CUBE_ALIGN,
                                          CUBE_ALIGN });

    std::vector<int64_t> addressOffset({ batch, this->searchParam->nProbeL2 * 6 });
    std::vector<int64_t> diff1Shape({ 1, this->subSpaceDimL2 });
    std::vector<int64_t> diff2Shape({ 1, this->subSpaceDimL2 });
    std::vector<int64_t> normL2Shape({ 1, static_cast<int>(this->ntotal) });

    std::vector<int64_t> outDistShape({ batch, this->searchParam->l3SegmentNum * BASE_SEG_SIZE });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });
    std::vector<int64_t> vcMinShape({ batch, 2 * this->searchParam->l3SegmentNum * BASE_SEG_SIZE / VCMIN_SEG_SIZE });

    aclTensor *queryCodeTensor = aclCreateTensor(queryCodeShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                                 queryCodeShape.data(), 2, queryCodes.data());
    aclTensor *codeWordsTensor = aclCreateTensor(codeWordsShape.data(), 4, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND,
                                                 codeWordsShape.data(), 4, minAddressOfBaseNpu);
    aclTensor *addressOffsetTensor = aclCreateTensor(addressOffset.data(), 2, ACL_UINT64, nullptr, 0, ACL_FORMAT_ND,
                                                     addressOffset.data(), 2, addressOffsetOfBucketL3.data());
    aclTensor *diff1Tensor = aclCreateTensor(diff1Shape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             diff1Shape.data(), 2, vDiffNpu->data());
    aclTensor *diff2Tensor = aclCreateTensor(diff2Shape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                             diff2Shape.data(), 2, vDiff2Npu->data());
    aclTensor *normL2Tensor = aclCreateTensor(normL2Shape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                              normL2Shape.data(), 2, precomputeNormL2Npu->data());

    aclTensor *outDistsTensor = aclCreateTensor(outDistShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                                outDistShape.data(), 2, dists.data());
    aclTensor *flagTensor = aclCreateTensor(flagShape.data(), 2, ACL_UINT16, nullptr, 0, ACL_FORMAT_ND,
                                            flagShape.data(), 2, opFlag.data());
    aclTensor *distsVcMinTensor = aclCreateTensor(vcMinShape.data(), 2, ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                                                  vcMinShape.data(), 2, distsVcMin.data());
    auto ret = aclnnVSC3GetWorkspaceSize(queryCodeTensor, codeWordsTensor, addressOffsetTensor,
                                         diff1Tensor, diff2Tensor, normL2Tensor, this->nList,
                                         this->nListL2, this->searchParam->l3SegmentNum, outDistsTensor,
                                         flagTensor, distsVcMinTensor, &workspaceSize, &handle);
    ACL_REQUIRE_OK(ret);
    void *workspace = nullptr;
    if (workspaceSize != 0) {
        ACL_REQUIRE_OK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    auto delta_t1 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    printf("======aclnnVSC3GetWorkspaceSize Time Spend: %f ms\n", delta_t1 / 1000.0);
    ret = aclnnVSC3(workspace, workspaceSize, handle, defaultStream);

    (void)aclDestroyTensor(queryCodeTensor);
    (void)aclDestroyTensor(codeWordsTensor);
    (void)aclDestroyTensor(addressOffsetTensor);
    (void)aclDestroyTensor(diff1Tensor);
    (void)aclDestroyTensor(diff2Tensor);
    (void)aclDestroyTensor(normL2Tensor);
    (void)aclDestroyTensor(outDistsTensor);
    (void)aclDestroyTensor(flagTensor);
    (void)aclDestroyTensor(distsVcMinTensor);
    ACL_REQUIRE_OK(ret);

#else
    AscendOperator *l3DistOp = nullptr;
    if (l3DistWithMaskOps.find(batch) != l3DistWithMaskOps.end()) {
        l3DistOp = l3DistWithMaskOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(l3DistOp);
    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(queryCodes.data(), queryCodes.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(minAddressOfBaseNpu, BASE_SEG_SIZE * this->dimStored));
    distOpInput.emplace_back(
        aclCreateDataBuffer(addressOffsetOfBucketL3.data(), addressOffsetOfBucketL3.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(vDiffNpu->data(), vDiffNpu->getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(vDiff2Npu->data(), vDiff2Npu->getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(precomputeNormL2Npu->data(), BASE_SEG_SIZE * sizeof(float16_t)));
    distOpInput.emplace_back(aclCreateDataBuffer(maskByteNpu->data(), BASE_SEG_SIZE * sizeof(uint8_t)));
    distOpInput.emplace_back(aclCreateDataBuffer(attr_nlistL1.data(), attr_nlistL1.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(attr_nlistL2.data(), attr_nlistL2.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(attr_segmentL3.data(), attr_segmentL3.getSizeInBytes()));

    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(distsVcMin.data(), distsVcMin.getSizeInBytes()));

    l3DistOp->exec(distOpInput, distOpOutput, defaultStream);
    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();

    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();
#endif
    APP_LOG_INFO("NpuIndexIVFHSP RunL3DistWithMaskOp operation end.\n");
}

void NpuIndexIVFHSP::RunL3TopKOp(AscendTensor<float16_t, DIMS_2> &dists, AscendTensor<float16_t, DIMS_2> &distsVcMin,
                                 AscendTensor<uint16_t, DIMS_2> &opFlag,
                                 AscendTensor<uint64_t, DIMS_2> &idAddressOffsetOfBucketL3,
                                 AscendTensor<int64_t, DIMS_1> &attr,
                                 //                                 AscendTensor<uint64_t, DIMS_2>
                                 //                                 &addressOffsetOfBucketL3,
                                 AscendTensor<float16_t, DIMS_2> &distsRes, AscendTensor<int64_t, DIMS_2> &labelsRes)
{
    APP_LOG_INFO("NpuIndexIVFHSP RunL3TopKOp operation started.\n");
    AscendOperator *op = nullptr;
    int batch = dists.getSize(0);
    if (l3TopKOps.find(batch) != l3TopKOps.end()) {
        op = l3TopKOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    // prepare for input data's buffer
    std::vector<const aclDataBuffer *> topkOpInput;

    topkOpInput.emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(distsVcMin.data(), distsVcMin.getSizeInBytes()));
    topkOpInput.emplace_back(
        aclCreateDataBuffer(idAddressOffsetOfBucketL3.data(), idAddressOffsetOfBucketL3.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(attr.data(), attr.getSizeInBytes()));

    // prepare for output data's buffer
    std::vector<aclDataBuffer *> topkOpOutput;
    topkOpOutput.emplace_back(aclCreateDataBuffer(distsRes.data(), distsRes.getSizeInBytes()));
    topkOpOutput.emplace_back(aclCreateDataBuffer(labelsRes.data(), labelsRes.getSizeInBytes()));

    // async executing operator
    op->exec(topkOpInput, topkOpOutput, aiCpuStream);

    for (auto &item : topkOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpInput.clear();

    for (auto &item : topkOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpOutput.clear();
    APP_LOG_INFO("NpuIndexIVFHSP RunL3TopKOp operation end.\n");
}

void NpuIndexIVFHSP::RunL3TopKOp(const std::vector<NpuIndexIVFHSP *> &indexes, int i,
                                 AscendTensor<float16_t, DIMS_3> &dists, AscendTensor<float16_t, DIMS_3> &distsVcMin,
                                 AscendTensor<uint16_t, DIMS_3> &opFlag,
                                 AscendTensor<uint64_t, DIMS_3> &idAddressOffsetOfBucketL3,
                                 AscendTensor<int64_t, DIMS_1> &attr,
                                 //                                 AscendTensor<uint64_t, DIMS_3>
                                 //                                 &addressOffsetOfBucketL3,
                                 AscendTensor<float16_t, DIMS_3> &distsRes, AscendTensor<int64_t, DIMS_3> &labelsRes)
{
    APP_LOG_INFO("NpuIndexIVFHSP RunL3TopKOp operation started.\n");
    AscendOperator *op = nullptr;
    size_t indexSize = indexes.size();
    int batch = dists.getSize(1);
    if (l3TopKOps.find(batch) != l3TopKOps.end()) {
        op = l3TopKOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    // prepare for input data's buffer
    std::vector<const aclDataBuffer *> topkOpInput;

    topkOpInput.emplace_back(
        aclCreateDataBuffer(dists.data() + i * (dists.numElements() / indexSize), dists.getSizeInBytes() / indexSize));
    topkOpInput.emplace_back(aclCreateDataBuffer(distsVcMin.data() + i * (distsVcMin.numElements() / indexSize),
                                                 distsVcMin.getSizeInBytes() / indexSize));
    topkOpInput.emplace_back(aclCreateDataBuffer(idAddressOffsetOfBucketL3.data() +
                                                     i * (idAddressOffsetOfBucketL3.numElements() / indexSize),
                                                 idAddressOffsetOfBucketL3.getSizeInBytes() / indexSize));
    topkOpInput.emplace_back(aclCreateDataBuffer(opFlag.data() + i * (opFlag.numElements() / indexSize),
                                                 opFlag.getSizeInBytes() / indexSize));
    topkOpInput.emplace_back(aclCreateDataBuffer(attr.data(), attr.getSizeInBytes()));

    // prepare for output data's buffer
    std::vector<aclDataBuffer *> topkOpOutput;
    topkOpOutput.emplace_back(aclCreateDataBuffer(distsRes.data() + i * (distsRes.numElements() / indexSize),
                                                  distsRes.getSizeInBytes() / indexSize));
    topkOpOutput.emplace_back(aclCreateDataBuffer(labelsRes.data() + i * (labelsRes.numElements() / indexSize),
                                                  labelsRes.getSizeInBytes() / indexSize));

    // async executing operator
    op->exec(topkOpInput, topkOpOutput, aiCpuStream);

    for (auto &item : topkOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpInput.clear();

    for (auto &item : topkOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpOutput.clear();
    APP_LOG_INFO("NpuIndexIVFHSP RunL3TopKOp operation end.\n");
}

void NpuIndexIVFHSP::RunMultiL3TopKOp(
    AscendTensor<float16_t, DIMS_3> &dists, AscendTensor<float16_t, DIMS_3> &distsVcMin,
    AscendTensor<uint16_t, DIMS_3> &opFlag, AscendTensor<uint64_t, DIMS_3> &idAddressOffsetOfBucketL3,
    AscendTensor<int64_t, DIMS_1> &attr, AscendTensor<uint64_t, DIMS_3> &addressOffsetOfBucketL3,
    AscendTensor<float16_t, DIMS_3> &distsRes, AscendTensor<int64_t, DIMS_3> &labelsRes)
{
    APP_LOG_INFO("NpuMultiIndexIVFHSP RunL3TopKOp operation started.\n");
    AscendOperator *op = nullptr;
    int batch = dists.getSize(1);
    if (l3MultiTopKOps.find(batch) != l3MultiTopKOps.end()) {
        op = l3MultiTopKOps[batch].get();
    }
    ASCEND_THROW_IF_NOT(op);

    // prepare for input data's buffer
    std::vector<const aclDataBuffer *> topkOpInput;

    topkOpInput.emplace_back(aclCreateDataBuffer(dists.data(), dists.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(distsVcMin.data(), distsVcMin.getSizeInBytes()));
    topkOpInput.emplace_back(
        aclCreateDataBuffer(idAddressOffsetOfBucketL3.data(), idAddressOffsetOfBucketL3.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(opFlag.data(), opFlag.getSizeInBytes()));
    topkOpInput.emplace_back(aclCreateDataBuffer(attr.data(), attr.getSizeInBytes()));
    topkOpInput.emplace_back(
        aclCreateDataBuffer(addressOffsetOfBucketL3.data(), addressOffsetOfBucketL3.getSizeInBytes()));

    // prepare for output data's buffer
    std::vector<aclDataBuffer *> topkOpOutput;
    topkOpOutput.emplace_back(aclCreateDataBuffer(distsRes.data(), distsRes.getSizeInBytes()));
    topkOpOutput.emplace_back(aclCreateDataBuffer(labelsRes.data(), labelsRes.getSizeInBytes()));

    // async executing operator
    op->exec(topkOpInput, topkOpOutput, aiCpuStream);

    for (auto &item : topkOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpInput.clear();

    for (auto &item : topkOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    topkOpOutput.clear();
    APP_LOG_INFO("NpuIndexIVFHSP RunMultiL3TopKOp operation end.\n");
}

void NpuIndexIVFHSP::RunFpToFp16(ascendSearchacc::AscendOperator *op, AscendTensor<float, DIMS_2> &floatDataNpu,
                                 AscendTensor<float16_t, DIMS_2> &fp16DataNpu, AscendTensor<uint16_t, DIMS_2> &flag)
{
    ASCEND_THROW_IF_NOT(op);
    auto stream = resources->getDefaultStream();
    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(floatDataNpu.data(), floatDataNpu.getSizeInBytes()));
    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(fp16DataNpu.data(), fp16DataNpu.getSizeInBytes()));
    distOpOutput.emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));
    op->exec(distOpInput, distOpOutput, stream);
    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();

    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();
}

void NpuIndexIVFHSP::RunMatMul(AscendTensor<float, DIMS_2> &LeftMatNpu, AscendTensor<float, DIMS_3> &RightMatNpu,
                               AscendTensor<float, DIMS_3> &OutMatNpu)
{
    AscendOperator *MatMulop = nullptr;
    if (MatMulFp32L1Ops.find(PROGRESSBATCH) != MatMulFp32L1Ops.end()) {
        MatMulop = MatMulFp32L1Ops[PROGRESSBATCH].get();
    }
    ASCEND_THROW_IF_NOT(MatMulop);

    std::vector<const aclDataBuffer *> distOpInput;
    distOpInput.emplace_back(aclCreateDataBuffer(LeftMatNpu.data(), LeftMatNpu.getSizeInBytes()));
    distOpInput.emplace_back(aclCreateDataBuffer(RightMatNpu.data(), RightMatNpu.getSizeInBytes()));
    std::vector<aclDataBuffer *> distOpOutput;
    distOpOutput.emplace_back(aclCreateDataBuffer(OutMatNpu.data(), OutMatNpu.getSizeInBytes()));

    MatMulop->exec(distOpInput, distOpOutput, defaultStream);

    for (auto &item : distOpInput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpInput.clear();

    for (auto &item : distOpOutput) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    distOpOutput.clear();
}

APP_ERROR NpuIndexIVFHSP::ResetL1DistOp()
{
    APP_LOG_INFO("NpuIndexIVFHSP ResetL1DistOp Operation Start.\n");
#ifndef USE_ACL_NN_INTERFACE

    auto l1DistOpReset = [this](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("VstarComputeL1");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> coarseCentroidsShape({ utils::divUp(this->subSpaceDimL1 * this->nList, CUBE_ALIGN),
                                                    utils::divUp(this->dim, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> queryCodeShape({ batch, this->nList * this->subSpaceDimL1 });
        std::vector<int64_t> distResultShape({ batch, this->nList });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });  // 同步标记

        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, coarseCentroidsShape.size(), coarseCentroidsShape.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, queryCodeShape.size(), queryCodeShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);
        auto ret = aclopSetAttrInt(desc.opAttr, "subSpaceDim", this->subSpaceDimL1);
        ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "set attr[subSpaceDim] error.");

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : opAccessBatchList) {
        l1DistOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l1DistOpReset(l1DistOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "op init failed");
    }
#endif
    APP_LOG_INFO("NpuIndexIVFHSP ResetL1DistOp Operation End.\n");
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::ResetL2DistOp()
{
    APP_LOG_INFO("NpuIndexIVFHSP ResetL2DistOp Operation Start.\n");
#ifdef USE_ACL_NN_INTERFACE
#else
    auto l2DistOpReset = [this](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("VstarComputeL2");
        std::vector<int64_t> queryCodeShape({ batch, nList * this->subSpaceDimL1 });
        std::vector<int64_t> codebookL2Shape({ utils::divUp(nList * nListL2 * subSpaceDimL2, CUBE_ALIGN),
                                               utils::divUp(subSpaceDimL1, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> l1IndicesShape({ batch, this->searchParam->nProbeL1 });
        std::vector<int64_t> distResultShape({ batch, this->searchParam->nProbeL1 * this->nListL2 });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT16, queryCodeShape.size(), queryCodeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, codebookL2Shape.size(), codebookL2Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, l1IndicesShape.size(), l1IndicesShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);
        auto ret = aclopSetAttrInt(desc.opAttr, "subSpaceDim1", this->subSpaceDimL1);
        ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "set attr[subSpaceDim1] error.");
        ret = aclopSetAttrInt(desc.opAttr, "subSpaceDim2", this->subSpaceDimL2);
        ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "set attr[subSpaceDim2] error.");

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : opAccessBatchList) {
        l2DistOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l2DistOpReset(l2DistOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "L2DistOp op init failed");
    }
    APP_LOG_INFO("NpuIndexIVFHSP ResetL2DistOp Operation End.\n");

    // special init for Adding with PROGRESSBATCH batchsize
    auto l2DistOpResetAdd = [this](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("VstarComputeL2");
        std::vector<int64_t> queryCodeShape({ batch, nList * this->subSpaceDimL1 });
        std::vector<int64_t> codebookL2Shape({ utils::divUp(nList * nListL2 * subSpaceDimL2, CUBE_ALIGN),
                                               utils::divUp(subSpaceDimL1, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> l1IndicesShape({ batch, 1 });
        std::vector<int64_t> distResultShape({ batch, 1 * this->nListL2 });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT16, queryCodeShape.size(), queryCodeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, codebookL2Shape.size(), codebookL2Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, l1IndicesShape.size(), l1IndicesShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, distResultShape.size(), distResultShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);
        auto ret = aclopSetAttrInt(desc.opAttr, "subSpaceDim1", this->subSpaceDimL1);
        ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "set attr[subSpaceDim1] error.");
        ret = aclopSetAttrInt(desc.opAttr, "subSpaceDim2", this->subSpaceDimL2);
        ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "set attr[subSpaceDim2] error.");

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    l2DistOps[PROGRESSBATCH] = std::unique_ptr<AscendOperator>(nullptr);
    APPERR_RETURN_IF_NOT_LOG(l2DistOpResetAdd(l2DistOps[PROGRESSBATCH], PROGRESSBATCH),
                             APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "L2DistOp_Add op init failed");
    APP_LOG_INFO("NpuIndexIVFHSP ResetL2DistOp Operation End.\n");

#endif
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::ResetL3DistOp()
{
    APP_LOG_INFO("NpuIndexIVFHSP ResetL3DistOp Operation Start.\n");
#ifndef USE_ACL_NN_INTERFACE
    auto l3DistOpReset = [this](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("VSC3");
        std::vector<int64_t> queryCodeShape({ batch, this->nList * this->subSpaceDimL1 });
        std::vector<int64_t> codeWordsShape({ utils::divUp(BASE_SEG_SIZE, CUBE_ALIGN),
                                              utils::divUp(this->dimStored, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> addressOffsetShape({ batch, this->searchParam->nProbeL2 * 6 });
        std::vector<int64_t> diff1Shape({ 1, this->dimStored });  // each maximum contains 2 values
        std::vector<int64_t> diff2Shape({ 1, this->dimStored });  // each maximum contains 2 values
        std::vector<int64_t> normL2Shape({ 1, BASE_SEG_SIZE });   // each maximum contains 2 values
        std::vector<int64_t> attr_nlistl1Shape({ 1, this->nList });
        std::vector<int64_t> attr_nlistl2Shape({ 1, this->nListL2 });
        std::vector<int64_t> attr_l3SegmentNumShape({ 1, this->searchParam->l3SegmentNum });

        std::vector<int64_t> outDistShape({ batch, this->searchParam->l3SegmentNum * BASE_SEG_SIZE });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });
        std::vector<int64_t> vcMinShape(
            { batch, 2 * this->searchParam->l3SegmentNum * BASE_SEG_SIZE / VCMIN_SEG_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT16, queryCodeShape.size(), queryCodeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, codeWordsShape.size(), codeWordsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, addressOffsetShape.size(), addressOffsetShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, diff1Shape.size(), diff1Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, diff2Shape.size(), diff2Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, normL2Shape.size(), normL2Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, attr_nlistl1Shape.size(), attr_nlistl1Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, attr_nlistl2Shape.size(), attr_nlistl2Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, attr_l3SegmentNumShape.size(), attr_l3SegmentNumShape.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, outDistShape.size(), outDistShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, vcMinShape.size(), vcMinShape.data(), ACL_FORMAT_ND);

        auto ret = aclopSetAttrInt(desc.opAttr, "nlist1", this->nList);
        ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "set attr[nlist1] error.");
        ret = aclopSetAttrInt(desc.opAttr, "nlist2", this->nListL2);
        ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "set attr[nlist2] error.");
        ret = aclopSetAttrInt(desc.opAttr, "segmentNum", this->searchParam->l3SegmentNum);
        ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "set attr[segmentNum] error.");

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    for (auto batch : opAccessBatchList) {
        l3DistOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l3DistOpReset(l3DistOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "VSC3 op init failed");
    }
#endif
    APP_LOG_INFO("NpuIndexIVFHSP ResetL3DistOp Operation End.\n");
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::ResetL3DistWithMaskOp()
{
    APP_LOG_INFO("NpuIndexIVFHSP ResetL3DistWithMaskOp Operation Start.\n");
#ifndef USE_ACL_NN_INTERFACE
    auto l3DistOpReset = [this](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("VSM3");

        std::vector<int64_t> queryCodeShape({ batch, this->nList * this->subSpaceDimL1 });
        std::vector<int64_t> codeWordsShape({ utils::divUp(BASE_SEG_SIZE, CUBE_ALIGN),
                                              utils::divUp(this->dimStored, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN });
        std::vector<int64_t> addressOffsetShape({ batch, this->searchParam->nProbeL2 * 6 });
        std::vector<int64_t> diff1Shape({ 1, this->dimStored });  // each maximum contains 2 values
        std::vector<int64_t> diff2Shape({ 1, this->dimStored });  // each maximum contains 2 values
        std::vector<int64_t> normL2Shape({ 1, BASE_SEG_SIZE });   // each maximum contains 2 values
        std::vector<int64_t> maskShape({ 1, BASE_SEG_SIZE });     // each maximum contains 2 values
        std::vector<int64_t> attr_nlistl1Shape({ 1, this->nList });
        std::vector<int64_t> attr_nlistl2Shape({ 1, this->nListL2 });
        std::vector<int64_t> attr_l3SegmentNumShape({ 1, this->searchParam->l3SegmentNum });

        std::vector<int64_t> outDistShape({ batch, this->searchParam->l3SegmentNum * BASE_SEG_SIZE });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });
        std::vector<int64_t> vcMinShape(
            { batch, 2 * this->searchParam->l3SegmentNum * BASE_SEG_SIZE / VCMIN_SEG_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT16, queryCodeShape.size(), queryCodeShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, codeWordsShape.size(), codeWordsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, addressOffsetShape.size(), addressOffsetShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, diff1Shape.size(), diff1Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, diff2Shape.size(), diff2Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, normL2Shape.size(), normL2Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT8, maskShape.size(), maskShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, attr_nlistl1Shape.size(), attr_nlistl1Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, attr_nlistl2Shape.size(), attr_nlistl2Shape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT32, attr_l3SegmentNumShape.size(), attr_l3SegmentNumShape.data(), ACL_FORMAT_ND);
        
        desc.addOutputTensorDesc(ACL_FLOAT16, outDistShape.size(), outDistShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, vcMinShape.size(), vcMinShape.data(), ACL_FORMAT_ND);

        auto ret = aclopSetAttrInt(desc.opAttr, "nlist1", this->nList);
        ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "set attr[nlist1] error.");
        ret = aclopSetAttrInt(desc.opAttr, "nlist2", this->nListL2);
        ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "set attr[nlist2] error.");
        ret = aclopSetAttrInt(desc.opAttr, "segmentNum", this->searchParam->l3SegmentNum);
        ASCEND_THROW_IF_NOT_MSG(ret == ACL_SUCCESS, "set attr[segmentNum] error.");
        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };
    for (auto batch : opAccessBatchList) {
        l3DistWithMaskOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(l3DistOpReset(l3DistWithMaskOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "VSC3 op init failed");
    }
#endif
    APP_LOG_INFO("NpuIndexIVFHSP ResetL3DistWithMaskOp Operation End.\n");
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::ResetFpToFp16()
{
#ifndef USE_ACL_NN_INTERFACE
    auto opReset = [this](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("FpToFp16");
        std::vector<int64_t> queryShape({ batch, this->dim });
        std::vector<int64_t> queryFp16Shape({ batch, this->dim });
        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

        desc.addInputTensorDesc(ACL_FLOAT, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT16, queryFp16Shape.size(), queryFp16Shape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : opAccessBatchList) {
        fpToFp16Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(opReset(fpToFp16Ops[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "op init failed");
    }
#endif
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::ResetMatMul()
{
#ifndef USE_ACL_NN_INTERFACE
    auto opReset = [this](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("VstarBaseAddMatMul");
        std::vector<int64_t> LeftShape({ batch, this->dim });
        std::vector<int64_t> RightShape({ this->nList, this->subSpaceDimL1, this->dim });
        std::vector<int64_t> OutShape({ batch, this->nList, this->subSpaceDimL1 });

        desc.addInputTensorDesc(ACL_FLOAT, LeftShape.size(), LeftShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT, RightShape.size(), RightShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_FLOAT, OutShape.size(), OutShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : opAccessAddBatchList) {
        MatMulFp32L1Ops[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(opReset(MatMulFp32L1Ops[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "MatMul op init failed");
    }
#endif
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::ResetL1TopKOp()
{
    APP_LOG_INFO("NpuIndexIVFHSP ResetL1TopKOp operation started.\n");
    auto opReset = [this](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("IvfSpTopkL1");
        std::vector<int64_t> shape0{ batch, this->nList };
        std::vector<int64_t> shape1{ CORE_NUM, SIZE_ALIGN };
        std::vector<int64_t> shape2{ CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> shape3{ static_cast<int>(TopkIvfSpL1AttrIdx::TOPK_L1_ATTR_IDX_COUNT) };
        std::vector<int64_t> shape4{ batch, 0 };

        desc.addInputTensorDesc(ACL_FLOAT16, shape0.size(), shape0.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT32, shape1.size(), shape1.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, shape2.size(), shape2.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, shape3.size(), shape3.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, shape4.size(), shape4.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT16, shape4.size(), shape4.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : opAccessBatchList) {
        l1TopKOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(opReset(l1TopKOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");
    }
    APP_LOG_INFO("NpuIndexIVFHSP ResetL1TopKOp operation end.\n");
    return APP_ERR_OK;
}

/**
 * 需要在add全量数据之后reset或者重置searchparam时reset
 * @return
 */
APP_ERROR NpuIndexIVFHSP::ResetL2TopKOp()
{
    APP_LOG_INFO("NpuIndexIVFHSP ResetL2TopKOp operation started.\n");
    auto opReset = [this](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("IvfSpTopkL2");
        std::vector<int64_t> distsShape{ batch, this->searchParam->nProbeL1 * nListL2 };
        std::vector<int64_t> l1IndicesShape{ batch, this->searchParam->nProbeL1 };
        std::vector<int64_t> opFlagShape{ CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> addrOffsetOfBucketShape{ this->nList * this->nListL2 * 6 };
        std::vector<int64_t> attrShape{ static_cast<int64_t>(TopkIvfSpL2AttrIdx::TOPK_L2_ATTR_IDX_COUNT) };

        std::vector<int64_t> distsResShape{ batch, this->searchParam->nProbeL2 };
        std::vector<int64_t> addressOffsetL3Shape{ batch, this->searchParam->nProbeL2 * 6 };
        std::vector<int64_t> idAdressShape{ batch, this->searchParam->nProbeL2 * 2 };

        desc.addInputTensorDesc(ACL_FLOAT16, distsShape.size(), distsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, l1IndicesShape.size(), l1IndicesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, opFlagShape.size(), opFlagShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, addrOffsetOfBucketShape.size(), addrOffsetOfBucketShape.data(),
                                ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, attrShape.size(), attrShape.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, distsResShape.size(), distsResShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, addressOffsetL3Shape.size(), addressOffsetL3Shape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, idAdressShape.size(), idAdressShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : opAccessBatchList) {
        l2TopKOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(opReset(l2TopKOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");
    }
    APP_LOG_INFO("NpuIndexIVFHSP ResetL2TopKOp operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::ResetMultiL2TopKOp()
{
    APP_LOG_INFO("NpuIndexIVFHSP ResetMultiL2TopKOp operation started.\n");
    auto opReset = [this](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("IvfMultiSpTopkL2");
        std::vector<int64_t> distsShape{ batch, this->searchParam->nProbeL1 * nListL2 };
        std::vector<int64_t> l1IndicesShape{ batch, this->searchParam->nProbeL1 };
        std::vector<int64_t> opFlagShape{ CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> attrShape{ static_cast<int64_t>(IvfMultiSpTopkL2AttrIdx::TOPK_L2_ATTR_IDX_COUNT) };

        std::vector<int64_t> distsResShape{ batch, this->searchParam->nProbeL2 };
        std::vector<int64_t> idAdressShape{ batch, this->searchParam->nProbeL2 * 2 };

        desc.addInputTensorDesc(ACL_FLOAT16, distsShape.size(), distsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, l1IndicesShape.size(), l1IndicesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, opFlagShape.size(), opFlagShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, attrShape.size(), attrShape.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, distsResShape.size(), distsResShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, idAdressShape.size(), idAdressShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : opAccessBatchList) {
        l2MultiTopKOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(opReset(l2MultiTopKOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "op init failed");
    }
    APP_LOG_INFO("NpuIndexIVFHSP ResetMultiL2TopKOp operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::ResetL2TopKWithMaskOp()
{
    APP_LOG_INFO("NpuIndexIVFHSP ResetL2TopKWithMaskOp operation started.\n");
    auto opReset = [this](std::unique_ptr<AscendOperator> &op, int batch) {
        AscendOpDesc desc("IvfSpTopkL2WithMask");
        std::vector<int64_t> maskBitShape{ static_cast<int64_t>((ntotal + 7) / 8) };
        std::vector<int64_t> distsShape{ batch, this->searchParam->nProbeL1 * nListL2 };
        std::vector<int64_t> l1IndicesShape{ batch, this->searchParam->nProbeL1 };
        std::vector<int64_t> opFlagShape{ CORE_NUM, FLAG_SIZE };
        std::vector<int64_t> addrOffsetOfBucketShape{ this->nList * this->nListL2 * 6 };
        std::vector<int64_t> attrShape{ static_cast<int64_t>(TopkIvfSpL2AttrIdx::TOPK_L2_ATTR_IDX_COUNT) };

        std::vector<int64_t> distsResShape{ batch, this->searchParam->nProbeL2 };
        std::vector<int64_t> addressOffsetL3Shape{ batch, this->searchParam->nProbeL2 * 6 };
        std::vector<int64_t> idAdressShape{ batch, this->searchParam->nProbeL2 * 2 };
        std::vector<int64_t> maskByteShape{ 1, BASE_SEG_SIZE };
        std::vector<int64_t> isMaskOffsetShape{ 1, nList * nListL2 * 2 };

        desc.addInputTensorDesc(ACL_UINT8, maskBitShape.size(), maskBitShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, distsShape.size(), distsShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, l1IndicesShape.size(), l1IndicesShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, opFlagShape.size(), opFlagShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, addrOffsetOfBucketShape.size(), addrOffsetOfBucketShape.data(),
                                ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, attrShape.size(), attrShape.data(), ACL_FORMAT_ND);

        desc.addOutputTensorDesc(ACL_FLOAT16, distsResShape.size(), distsResShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, addressOffsetL3Shape.size(), addressOffsetL3Shape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, idAdressShape.size(), idAdressShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT8, maskByteShape.size(), maskByteShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_UINT64, isMaskOffsetShape.size(), isMaskOffsetShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : opAccessBatchList) {
        l2TopKWithMaskOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(opReset(l2TopKWithMaskOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "op init failed");
    }
    APP_LOG_INFO("NpuIndexIVFHSP ResetL2TopKWithMaskOp operation end.\n");
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::ResetL3TopKOp()
{
    auto opReset = [this](std::unique_ptr<AscendOperator> &op, int64_t batch) {
        AscendOpDesc desc("IvfSpTopkL3");
        std::vector<int64_t> distShape({ batch, this->searchParam->l3SegmentNum * BASE_SEG_SIZE });
        // vmDist 存了label和distance，需要2倍空间，故乘上2；使atomicAdd不会产生越界错误申请2倍应有内存，再乘上2；
        std::vector<int64_t> vmDistShape(
            { batch, 2 * 2 * this->searchParam->l3SegmentNum * BASE_SEG_SIZE / VCMIN_SEG_SIZE });
        std::vector<int64_t> idAddrShape({ batch, this->searchParam->nProbeL2 * 2 });

        std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });
        std::vector<int64_t> attrShape({ static_cast<int64_t>(TopkIvfSpL3AttrIdx::TOPK_L3_ATTR_IDX_COUNT) });
        desc.addInputTensorDesc(ACL_FLOAT16, distShape.size(), distShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, vmDistShape.size(), vmDistShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, idAddrShape.size(), idAddrShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, attrShape.size(), attrShape.data(), ACL_FORMAT_ND);

        std::vector<int64_t> outDistsShape({ batch, 0 });
        std::vector<int64_t> outLabelsShape({ batch, 0 });

        desc.addOutputTensorDesc(ACL_FLOAT16, outDistsShape.size(), outDistsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, outLabelsShape.size(), outLabelsShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : opAccessBatchList) {
        l3TopKOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(opReset(l3TopKOps[batch], batch), APP_ERR_ACL_OP_LOAD_MODEL_FAILED, "op init failed");
    }
    return APP_ERR_OK;
}

APP_ERROR NpuIndexIVFHSP::ResetMultiL3TopKOp(int indexSize)
{
    auto opReset = [this](std::unique_ptr<AscendOperator> &op, int64_t batch, int indexSize) {
        AscendOpDesc desc("IvfMultiSpTopkL3");
        std::vector<int64_t> distShape({ indexSize, batch, this->searchParam->l3SegmentNum * BASE_SEG_SIZE });
        // vmDist 存了label和distance，需要2倍空间，故乘上2；使atomicAdd不会产生越界错误申请2倍应有内存，再乘上2；
        std::vector<int64_t> vmDistShape(
            { indexSize, batch, 2 * 2 * this->searchParam->l3SegmentNum * BASE_SEG_SIZE / VCMIN_SEG_SIZE });
        std::vector<int64_t> idAddrShape({ indexSize, batch, this->searchParam->nProbeL2 * 2 });
        std::vector<int64_t> addressOffsetShape({ indexSize, batch, this->searchParam->nProbeL2 * 6 });

        std::vector<int64_t> flagShape({ indexSize, CORE_NUM, FLAG_SIZE });
        std::vector<int64_t> attrShape({ static_cast<int64_t>(TopkIvfSpL3AttrIdx::TOPK_L3_ATTR_IDX_COUNT) });
        desc.addInputTensorDesc(ACL_FLOAT16, distShape.size(), distShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_FLOAT16, vmDistShape.size(), vmDistShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, idAddrShape.size(), idAddrShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_INT64, attrShape.size(), attrShape.data(), ACL_FORMAT_ND);
        desc.addInputTensorDesc(ACL_UINT64, addressOffsetShape.size(), addressOffsetShape.data(), ACL_FORMAT_ND);

        std::vector<int64_t> outDistsShape({ indexSize, batch, 0 });
        std::vector<int64_t> outLabelsShape({ indexSize, batch, 0 });

        desc.addOutputTensorDesc(ACL_FLOAT16, outDistsShape.size(), outDistsShape.data(), ACL_FORMAT_ND);
        desc.addOutputTensorDesc(ACL_INT64, outLabelsShape.size(), outLabelsShape.data(), ACL_FORMAT_ND);

        op.reset();
        op = CREATE_UNIQUE_PTR(AscendOperator, desc);
        return op->init();
    };

    for (auto batch : opAccessBatchList) {
        l3MultiTopKOps[batch] = std::unique_ptr<AscendOperator>(nullptr);
        APPERR_RETURN_IF_NOT_LOG(opReset(l3MultiTopKOps[batch], batch, indexSize), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
                                 "op init failed");
    }
    return APP_ERR_OK;
}

void NpuIndexIVFHSP::forwardIdsMapReverse(std::map<int64_t, int64_t> idMapReverseForwarded)
{
    this->idMapReverse = idMapReverseForwarded;
}

void NpuIndexIVFHSP::SetAddWithIds(bool isAddWithIds)
{
    this->isAddWithIds = isAddWithIds;
}

bool NpuIndexIVFHSP::GetAddWithIds() const
{
    return this->isAddWithIds;
}

void NpuIndexIVFHSP::SetIdMap(const std::map<int64_t, int64_t> &idMap)
{
    this->idMap = idMap;
}

const std::map<int64_t, int64_t>& NpuIndexIVFHSP::GetIdMap() const
{
    return this->idMap;
}

const std::map<int64_t, int64_t>& NpuIndexIVFHSP::GetIdsMapReverse() const
{
    return this->idMapReverse;
}

APP_ERROR NpuIndexIVFHSP::DeleteVectorsImpl(const std::vector<int64_t> &ids)
{
    std::vector<std::vector<int64_t> > deleteByBucket(nList * nListL2, std::vector<int64_t>());
    uint64_t ntotalUpdated = ntotal;

    std::vector<int64_t> realTBD = ids;
    
    size_t origSize = realTBD.size();
    std::sort(realTBD.begin(), realTBD.end()); // sort realTBD(To-Be-Delete) vectors
    auto last = std::unique(realTBD.begin(), realTBD.end());
    realTBD.erase(last, realTBD.end());

    if (origSize > realTBD.size()) {
        printf("Warning: There are duplicates in your to-be-deleted list; these duplicates have been removed.\n");
    }

    // for each base vector to delete, we record its position within the current xxByBuckets
    for (int k = 0; k < static_cast<int>(realTBD.size()); ++k) {
        bool found = false;
        for (int i = 0; i < static_cast<int>(idxByBucket.size()); ++i) {
            if (found) {
                continue;
            }  // once the id is found, we want to stop searching (note this only works for nonFuzzy Add)
            for (int j = 0; j < static_cast<int>(idxByBucket[i].size()); ++j) {
                if (found) {
                    continue;
                }
                if (idxByBucket[i][j] == realTBD[k]) {
                    found = true;
                    deleteByBucket[i].emplace_back(j);
                    ntotalUpdated -= 1;
                }
            }
        }
        printf("\r-----Deleting Vectors Progress:[%d/%zu]", k + 1, realTBD.size());
        fflush(stdout);
    }
    printf("\n");

// start erasing norm, indices, and codeWords from buckets
    for (int i = 0; i < static_cast<int>(deleteByBucket.size()); ++i) {
        if (deleteByBucket[i].size() > 0) {
            for (int j = static_cast<int>(deleteByBucket[i].size()) - 1; j >= 0; j--) {
                auto posToRemove = deleteByBucket[i][j];
                idxByBucket[i].erase(idxByBucket[i].begin() + posToRemove);
                normL2ByBucket[i].erase(normL2ByBucket[i].begin() + posToRemove);
                codeWordsByBucket[i].erase(codeWordsByBucket[i].begin() + posToRemove * dimStored,
                                           codeWordsByBucket[i].begin() + (posToRemove + 1) * dimStored);
            }
        }
    }
    auto ret = AddIntoNpuDataStore(ntotalUpdated, codeWordsByBucket, idxByBucket, normL2ByBucket);
    ntotal = ntotalUpdated;

    return ret;
}

APP_ERROR NpuIndexIVFHSP::DeleteVectors(const std::vector<int64_t> &ids)
{
    std::vector<int64_t> realTBD;
    for (const auto &idTBD : ids) {
        auto it = this->idMapReverse.find(idTBD);
        if (it != idMapReverse.end()) {
            realTBD.emplace_back(this->idMapReverse[idTBD]);
            idMap.erase(idMapReverse[idTBD]); // 也将对应的 (realId : virtualId) 组合删除
            // 如果idTBD是一个能映射到realId的virtualId，我们确信该realId一定会被删除，因此我们将对应的 (virtualId : realId) 组合删除
            idMapReverse.erase(idTBD);
        } else {
            // 如果idTBD没有在idMapReverse中找到，需要检查该实例是否真的由AddVectorsWithIds添加；若否，则仍将该id放入待删除的id列表内
            if (!isAddWithIds) {
                realTBD.emplace_back(idTBD);
            }
        }
    }
    auto ret = DeleteVectorsImpl(realTBD);
    return ret;
}

APP_ERROR NpuIndexIVFHSP::DeleteVectors(const int64_t &id)
{
    std::vector<int64_t> vectorId(1, id);
    auto ret = DeleteVectors(vectorId);
    return ret;
}

APP_ERROR NpuIndexIVFHSP::DeleteVectors(const int64_t &startId, const int64_t &endId)
{
    std::vector<int64_t> realTBD;
    std::vector<int64_t> virtualIdRemove; // 此向量存储已经被找到realId的组合的virtualId部分，对map迭代结束后移除
    if (isAddWithIds) {
        for (const auto &virtualRealMap : idMapReverse) {
            int64_t virtualId = virtualRealMap.first;
            int64_t realId = virtualRealMap.second;
            if (virtualId >= startId && virtualId <= endId) {
                realTBD.emplace_back(realId);
                virtualIdRemove.emplace_back(virtualId);
            }
        }
    } else {
        // combine all bucket's ids together
        std::vector<int64_t> allIdx;
        for (const auto &bucket: idxByBucket) {
            allIdx.insert(allIdx.end(), bucket.begin(), bucket.end());
        }
        for (const auto &idx : allIdx) {
            if (idx >= startId && idx <= endId) {
                realTBD.emplace_back(idx);
            }
        }
    }
    auto ret = DeleteVectorsImpl(realTBD);
    if (virtualIdRemove.size() > 0) {
        // 将virtualId: realId的组合从idMap和idMapReverse中移除
        for (const auto &virtualId: virtualIdRemove) {
            idMap.erase(idMapReverse[virtualId]);
            idMapReverse.erase(virtualId);
        }
    }
    return ret;
}

APP_ERROR NpuIndexIVFHSP::Reset()
{
    ASCEND_THROW_MSG("Not Implement!");
    return APP_ERR_OK;
}

uint64_t NpuIndexIVFHSP::GetNTotal() const
{
    return ntotal;
}

uint64_t NpuIndexIVFHSP::GetUniqueBaseVecCounter() const
{
    return uniqueBaseVecCounter;
}

int NpuIndexIVFHSP::GetDim() const
{
    return dim;
}

int NpuIndexIVFHSP::GetNlistL1() const
{
    return nList;
}

int NpuIndexIVFHSP::GetNlistL2() const
{
    return nListL2;
}

int NpuIndexIVFHSP::GetSubDimL1() const
{
    return subSpaceDimL1;
}

int NpuIndexIVFHSP::GetSubDimL2() const
{
    return subSpaceDimL2;
}

/***********************************************************************
 * Reader Utility Function
 ***********************************************************************/

template <typename T>
void NpuIndexIVFHSP::ReadAndCheck(T &existingParam, const ReadAndCheckType &checkParam, const VstarIOReader &fin)
{
    T existingParamCopy = existingParam;
    fin.ReadAndCheck((&existingParam), sizeof(T));

    switch (checkParam) {
        case ReadAndCheckType::dim:
            if (existingParamCopy == 0) {  // existing Param is 0: verify for correctness of the pass-in parameter
                ASCEND_THROW_IF_NOT(std::find(validDim.begin(), validDim.end(), existingParam) != validDim.end());
            } else {
                ASCEND_THROW_IF_NOT(existingParamCopy == existingParam);
            }
            break;
        case ReadAndCheckType::subdiml1:
            if (existingParamCopy == 0) {
                ASCEND_THROW_IF_NOT(std::find(validSubSpaceDimL1.begin(), validSubSpaceDimL1.end(),
                    existingParam) != validSubSpaceDimL1.end());
            } else {
                ASCEND_THROW_IF_NOT(existingParamCopy == existingParam);
            }
            break;
        case ReadAndCheckType::subdiml2:
            if (existingParamCopy == 0) {
                ASCEND_THROW_IF_NOT(existingParam == static_cast<T>(subSpaceDimL1 / 2)); // 1/2 of subdiml1
            } else {
                ASCEND_THROW_IF_NOT(existingParamCopy == existingParam);
            }
            break;
        case ReadAndCheckType::nlist1:
            if (existingParamCopy == 0) {
                ASCEND_THROW_IF_NOT(std::find(validNlistL1.begin(), validNlistL1.end(),
                    existingParam) != validNlistL1.end());
            } else {
                ASCEND_THROW_IF_NOT(existingParamCopy == existingParam);
            }
            break;
        case ReadAndCheckType::nlist2:
            if (existingParamCopy == 0) {
                ASCEND_THROW_IF_NOT(std::find(validNlistL2.begin(), validNlistL2.end(),
                    existingParam) != validNlistL2.end());
            } else {
                ASCEND_THROW_IF_NOT(existingParamCopy == existingParam);
            }
            break;
        case ReadAndCheckType::validRange:
            ASCEND_THROW_IF_NOT(existingParam > 0 && static_cast<int>(existingParam) <= N_TOTAL_UB);
            break;
        case ReadAndCheckType::False:
            ASCEND_THROW_IF_NOT(!existingParam);
            break;
        default:
            break;
    }
}

}  // namespace ascendSearchacc
