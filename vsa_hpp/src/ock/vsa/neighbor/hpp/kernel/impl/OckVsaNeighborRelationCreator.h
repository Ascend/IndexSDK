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


#ifndef VSA_OCKVSANEIGHBORRELATIONCREATOR_H
#define VSA_OCKVSANEIGHBORRELATIONCREATOR_H

#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaNeighborRelation.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaSampleSelector.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelSystem.h"
namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
template <typename DataTemp, uint64_t DimSizeTemp>
void BuildQueryFeatureByRowIds(std::shared_ptr<std::vector<uint32_t>> primaryCells,
    std::vector<DataTemp> &batchQueryFeature, DataTemp *pSrcData)
{
    if (primaryCells.get() == nullptr) {
        return;
    }
    for (uint32_t i = 0; i < primaryCells->size(); ++i) {
        uint32_t rowId = primaryCells->at(i);
        auto ret = memcpy_s(batchQueryFeature.data() + i * DimSizeTemp, batchQueryFeature.size() * sizeof(DataTemp),
            pSrcData + rowId * DimSizeTemp, DimSizeTemp * sizeof(DataTemp));
        if (ret != 0) {
            OCK_VSA_HPP_LOG_ERROR("memcpy_s failed, the srcCount is " << DimSizeTemp * sizeof(DataTemp) <<
                ", the dist size is " << batchQueryFeature.size() * sizeof(DataTemp) << ", the ret is " << ret);
            return;
        }
    }
}

template <uint64_t DimSizeTemp>
std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup> DoSearchNeighbor(
    std::shared_ptr<hcps::handler::OckHeteroHandler> handler, std::shared_ptr<npu::OckVsaAnnNpuBlockGroup> blockInfo,
    std::shared_ptr<OckVsaAnnCreateParam> param, std::shared_ptr<hmm::OckHmmHMObject> queryHmo,
    OckVsaErrorCode &errorCode, std::shared_ptr<hmm::OckHmmHMObject> topkDistHmo,
    std::shared_ptr<hmm::OckHmmHMObject> topkLabelsHmo)
{
    if (blockInfo.get() == nullptr || queryHmo.get() == nullptr || topkDistHmo.get() == nullptr ||
        topkLabelsHmo.get() == nullptr) {
        return std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup>();
    }
    auto streamBase = hcps::handler::helper::MakeStream(*handler, errorCode, hcps::OckDevStreamType::AI_CORE);

    auto hmoQueryNormGroup = hcps::nop::OckL2NormOpRun::BuildNormHmoBlock(queryHmo, *handler, DimSizeTemp,
        relation::PRIMARY_KEY_BATCH_SIZE, errorCode);
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup>();
    }
    hcps::nop::OckL2NormOpRun::ComputeNormSync(hmoQueryNormGroup, *handler, streamBase);
    errorCode = streamBase->WaitExecComplete();
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_VSA_HPP_LOG_ERROR("ComputeNormSync failed, errorCode is " << errorCode);
        return std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup>();
    }
    auto queryNormHmo = hmm::OckHmmHMObject::CreateSubHmo(hmoQueryNormGroup->normResult, 0,
        sizeof(OckFloat16) * utils::SafeRoundUp(relation::PRIMARY_KEY_BATCH_SIZE, hcps::nop::FP16_ALIGN));
    if (queryNormHmo == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("Create queryNormHmo failed");
        errorCode = hmm::HMM_ERROR_HMO_OBJECT_INVALID;
        return std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup>();
    }

    std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup> compOpGroup =
        std::make_shared<hcps::hcop::OckTopkDistCompOpHmoGroup>(false, relation::PRIMARY_KEY_BATCH_SIZE, DimSizeTemp,
        relation::NEIGHBOR_RELATION_COUNT_PER_CELL, param->BlockRowCount(), param->GroupBlockCount(),
        param->GroupRowCount(), 0U);
    compOpGroup->normsHmo = blockInfo->norms;
    compOpGroup->featuresHmo = blockInfo->features;
    compOpGroup->SetQueryHmos(queryHmo, queryNormHmo, nullptr);
    compOpGroup->SetOutputHmos(topkDistHmo, topkLabelsHmo);
    errorCode = hcps::hcop::OckTopkDistCompOpRun::RunMultiGroupsSync({ compOpGroup }, handler);
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_VSA_HPP_LOG_ERROR("RunMultiGroupsSync failed, errorCode is " << errorCode);
        return std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup>();
    }
    errorCode = streamBase->WaitExecComplete();
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_VSA_HPP_LOG_ERROR("WaitExecComplete failed, errorCode is " << errorCode);
        return std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup>();
    }
    return compOpGroup;
}

template <typename DataTemp, uint64_t DimSizeT>
void AddIsolateCells(relation::OckVsaSampleSelector<DataTemp, DimSizeT> &selector,
    relation::OckVsaNeighborRelationHmoGroup &outRelationTable)
{
    for (uint32_t i = 0; i < selector.relatedBitSet.Size(); ++i) {
        if (!selector.relatedBitSet.At(i) && !selector.primaryBitSet.At(i)) {
            outRelationTable.AddIsolateData(i);
            selector.primaryBitSet.Set(i);
        }
    }
}

template <typename DataTemp, uint64_t DimSizeT>
OckVsaErrorCode DealWithTopKData(std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup> compOpGroup,
    std::shared_ptr<acladapter::OckSyncUtils> ockSyncUtils,
    relation::OckVsaSampleSelector<DataTemp, DimSizeT> &selector, std::shared_ptr<std::vector<uint32_t>> primaryCells,
    relation::OckVsaNeighborRelationHmoGroup &outRelationTable, std::shared_ptr<OckVsaAnnCreateParam> param)
{
    if (compOpGroup == nullptr || primaryCells == nullptr) {
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    OckVsaErrorCode errorCode = VSA_SUCCESS;

    auto labelBuffer = compOpGroup->topkLabelsHmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0,
        compOpGroup->topkLabelsHmo->GetByteSize());
    if (labelBuffer == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("compOpGroup topkLabelsHmo get labelBuffer failed");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    auto distBuffer = compOpGroup->topkDistsHmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0,
        compOpGroup->topkDistsHmo->GetByteSize());
    if (distBuffer == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("compOpGroup topkDistsHmo get distBuffer failed");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    uint64_t *pLabels = reinterpret_cast<uint64_t *>(labelBuffer->Address());
    OckFloat16 *pDists = reinterpret_cast<OckFloat16 *>(distBuffer->Address());
    for (uint32_t i = 0; i < primaryCells->size(); ++i) {
        auto primaryId = primaryCells->at(i);
        std::vector<uint32_t> rowIds;
        double distanceMinThreshold = param->SecondClassNeighborCellThreshold();  // 通常cos最大距离不会超过1
        double firstDistanceThreshold = param->FirstClassNeighborCellThreshold(); // 通常cos最大距离不会超过1
        for (uint32_t j = 0; j < compOpGroup->k; ++j) {
            uint32_t labelIndex = i * compOpGroup->k + j;
            float dist = acladapter::OckAscendFp16::Fp16ToFloat(pDists[labelIndex]);
            if (pLabels[labelIndex] == primaryId) {
                continue;
            }
            if (dist < distanceMinThreshold) {
                break;
            }
            if (dist > firstDistanceThreshold && rowIds.size() < relation::SAMPLE_INTERVAL_OF_NEIGHBOR_CELL) {
                selector.SetUsed(static_cast<uint32_t>(pLabels[labelIndex]));
            }
            rowIds.push_back(static_cast<uint32_t>(pLabels[labelIndex]));
        }
        selector.SetUsed(primaryId);
        outRelationTable.AddData(primaryId, rowIds);
    }
    return errorCode;
}

void CalcDistanceThreshold(const std::vector<OckFloat16> &topDistance, std::shared_ptr<OckVsaAnnCreateParam> param)
{
    float firstClassDistSum = 0;
    float secondClassDistSum = 0;
    for (uint32_t i = 0; i < relation::PRIMARY_KEY_BATCH_SIZE * relation::THRESHOLD_AUTO_ADAPTER_BATCH_SEARCH_TIMES;
        ++i) {
        uint32_t firstLabelIndex = i * relation::NEIGHBOR_RELATION_COUNT_PER_CELL + relation::FIRST_CLASS_SAMPLE_INDEX;
        uint32_t secondLabelIndex =
            i * relation::NEIGHBOR_RELATION_COUNT_PER_CELL + relation::SECOND_CLASS_SAMPLE_INDEX;
        float firstDist = acladapter::OckAscendFp16::Fp16ToFloat(topDistance[firstLabelIndex]);
        float secondDist = acladapter::OckAscendFp16::Fp16ToFloat(topDistance[secondLabelIndex]);
        firstClassDistSum += firstDist;
        secondClassDistSum += secondDist;
    }
    double firstThreshold =
        firstClassDistSum / relation::PRIMARY_KEY_BATCH_SIZE / relation::THRESHOLD_AUTO_ADAPTER_BATCH_SEARCH_TIMES;
    double secondThreshold =
        secondClassDistSum / relation::PRIMARY_KEY_BATCH_SIZE / relation::THRESHOLD_AUTO_ADAPTER_BATCH_SEARCH_TIMES;
    param->SetFirstClassNeighborCellThreshold(firstThreshold);
    param->SetSecondClassNeighborCellThreshold(secondThreshold);
}

void PrintDistanceThreshold(const std::vector<OckFloat16> &topDistance)
{
    std::vector<float> distanceSum(relation::NEIGHBOR_RELATION_COUNT_PER_CELL, 0);
    for (uint32_t i = 0; i < relation::PRIMARY_KEY_BATCH_SIZE * relation::THRESHOLD_AUTO_ADAPTER_BATCH_SEARCH_TIMES;
        ++i) {
        for (uint32_t j = 0; j < relation::NEIGHBOR_RELATION_COUNT_PER_CELL; ++j) {
            uint32_t labelIndex = i * relation::NEIGHBOR_RELATION_COUNT_PER_CELL + j;
            distanceSum[j] += acladapter::OckAscendFp16::Fp16ToFloat(topDistance[labelIndex]);
        }
    }
    uint32_t distanceDiviFactor =
        relation::PRIMARY_KEY_BATCH_SIZE * relation::THRESHOLD_AUTO_ADAPTER_BATCH_SEARCH_TIMES;
    std::ostringstream os;
    os << "Neighbor Cell Threshold: ";
    for (size_t i = 0; i < distanceSum.size(); i++) {
        os << distanceSum[i] / static_cast<float>(distanceDiviFactor) << ",";
    }
    OCK_VSA_HPP_LOG_DEBUG(os.str());
}

OckVsaErrorCode GetSampleWithEqualDist(uint32_t interval, uint32_t *pData, uint32_t needCount, uint32_t startPos)
{
    for (uint32_t i = 0; i < needCount; ++i) {
        pData[i] = startPos + interval * i;
    }
    return VSA_SUCCESS;
}

template <typename DataTemp, uint64_t DimSizeTemp>
OckVsaErrorCode InitTopkThreshold(std::shared_ptr<npu::OckVsaAnnNpuBlockGroup> blockInNpu,
    std::shared_ptr<hcps::handler::OckHeteroHandler> handler,
    std::shared_ptr<npu::OckVsaAnnRawBlockInfo> unShapedBlockDatas, std::shared_ptr<OckVsaAnnCreateParam> param)
{
    OckVsaErrorCode errorCode = VSA_SUCCESS;
    if (blockInNpu == nullptr) {
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    auto ockSyncUtils = std::make_shared<acladapter::OckSyncUtils>(*handler->Service());
    auto primaryCells = std::make_shared<std::vector<uint32_t>>(relation::PRIMARY_KEY_BATCH_SIZE, 0UL);
    std::vector<DataTemp> batchQueryFeature(relation::PRIMARY_KEY_BATCH_SIZE * DimSizeTemp);

    std::vector<OckFloat16> topDistance(relation::PRIMARY_KEY_BATCH_SIZE * relation::NEIGHBOR_RELATION_COUNT_PER_CELL *
        relation::THRESHOLD_AUTO_ADAPTER_BATCH_SEARCH_TIMES);

    // copy query data to query hmo
    auto queryHmo = hcps::handler::helper::MakeDeviceHmo(*handler,
        sizeof(DataTemp) * DimSizeTemp * primaryCells->size(), errorCode);
    uint32_t interval = (param->GroupRowCount() / relation::THRESHOLD_AUTO_ADAPTER_BATCH_SEARCH_TIMES /
        relation::PRIMARY_KEY_BATCH_SIZE / 2U) *
        2U -
        1U;
    interval = std::max(interval, 1U);

    auto topkDistHmo = hcps::handler::helper::MakeDeviceHmo(*handler,
        sizeof(OckFloat16) * relation::NEIGHBOR_RELATION_COUNT_PER_CELL * relation::PRIMARY_KEY_BATCH_SIZE, errorCode);
    auto topkLabelsHmo = hcps::handler::helper::MakeDeviceHmo(*handler,
        sizeof(uint64_t) * relation::NEIGHBOR_RELATION_COUNT_PER_CELL * relation::PRIMARY_KEY_BATCH_SIZE, errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    for (uint32_t runTimes = 0; runTimes < relation::THRESHOLD_AUTO_ADAPTER_BATCH_SEARCH_TIMES; ++runTimes) {
        errorCode = GetSampleWithEqualDist(interval, primaryCells->data(), relation::PRIMARY_KEY_BATCH_SIZE,
            runTimes * relation::PRIMARY_KEY_BATCH_SIZE);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);

        BuildQueryFeatureByRowIds<DataTemp, DimSizeTemp>(primaryCells, batchQueryFeature,
            reinterpret_cast<DataTemp *>(unShapedBlockDatas->feature->Addr()));

        errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(queryHmo->Addr()),
            DimSizeTemp * sizeof(DataTemp) * primaryCells->size(),
            reinterpret_cast<const DataTemp *>(batchQueryFeature.data()),
            DimSizeTemp * sizeof(DataTemp) * primaryCells->size(), acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        auto compOpGroup =
            DoSearchNeighbor<DimSizeTemp>(handler, blockInNpu, param, queryHmo, errorCode, topkDistHmo, topkLabelsHmo);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);

        errorCode = ockSyncUtils->Copy(topDistance.data() +
            runTimes * relation::PRIMARY_KEY_BATCH_SIZE * relation::NEIGHBOR_RELATION_COUNT_PER_CELL,
            relation::PRIMARY_KEY_BATCH_SIZE * relation::NEIGHBOR_RELATION_COUNT_PER_CELL * sizeof(OckFloat16),
            reinterpret_cast<OckFloat16 *>(compOpGroup->topkDistsHmo->Addr()),
            relation::PRIMARY_KEY_BATCH_SIZE * relation::NEIGHBOR_RELATION_COUNT_PER_CELL * sizeof(OckFloat16),
            acladapter::OckMemoryCopyKind::DEVICE_TO_HOST);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
    }
    CalcDistanceThreshold(topDistance, param);
    PrintDistanceThreshold(topDistance);
    return errorCode;
}

template <typename DataTemp, uint64_t DimSizeTemp>
OckVsaErrorCode ParallBuildRelationTable(relation::OckVsaNeighborRelationHmoGroup &outRelationTable,
    std::shared_ptr<hcps::handler::OckHeteroHandler> handler, std::shared_ptr<npu::OckVsaAnnNpuBlockGroup> blockInNpu,
    std::shared_ptr<OckVsaAnnCreateParam> param, std::shared_ptr<npu::OckVsaAnnRawBlockInfo> unShapedBlockDatas,
    bool &isThresholdInitialised)
{
    if (!isThresholdInitialised) {
        InitTopkThreshold<DataTemp, DimSizeTemp>(blockInNpu, handler, unShapedBlockDatas, param);
        isThresholdInitialised = true;
    }

    OckVsaErrorCode errorCode = VSA_SUCCESS;

    auto ockSyncUtils = std::make_shared<acladapter::OckSyncUtils>(*handler->Service());
    auto primaryCells = std::make_shared<std::vector<uint32_t>>(relation::PRIMARY_KEY_BATCH_SIZE, 0UL);
    std::vector<DataTemp> batchQueryFeature(relation::PRIMARY_KEY_BATCH_SIZE * DimSizeTemp);

    relation::OckVsaSampleSelector<DataTemp, DimSizeTemp> selector(param->GroupRowCount());
    selector.SortHammingIndexVector(handler, unShapedBlockDatas);

    uint32_t runTimes = 0;
    auto queryHmo = hcps::handler::helper::MakeDeviceHmo(*handler,
        sizeof(DataTemp) * DimSizeTemp * primaryCells->size(), errorCode);
    auto topkDistHmo = hcps::handler::helper::MakeDeviceHmo(*handler,
        sizeof(OckFloat16) * relation::NEIGHBOR_RELATION_COUNT_PER_CELL * relation::PRIMARY_KEY_BATCH_SIZE, errorCode);
    auto topkLabelsHmo = hcps::handler::helper::MakeDeviceHmo(*handler,
        sizeof(uint64_t) * relation::NEIGHBOR_RELATION_COUNT_PER_CELL * relation::PRIMARY_KEY_BATCH_SIZE, errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    while (selector.SelectUnusedRow(relation::PRIMARY_KEY_BATCH_SIZE, primaryCells->data(),
        relation::PRIMARY_KEY_SELECTION_INTERVAL)) {
        BuildQueryFeatureByRowIds<DataTemp, DimSizeTemp>(primaryCells, batchQueryFeature,
            reinterpret_cast<DataTemp *>(unShapedBlockDatas->feature->Addr()));
        runTimes++;

        errorCode = ockSyncUtils->Copy(reinterpret_cast<void *>(queryHmo->Addr()),
            DimSizeTemp * sizeof(DataTemp) * primaryCells->size(),
            reinterpret_cast<const DataTemp *>(batchQueryFeature.data()),
            DimSizeTemp * sizeof(DataTemp) * primaryCells->size(), acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);

        auto compOpGroup =
            DoSearchNeighbor<DimSizeTemp>(handler, blockInNpu, param, queryHmo, errorCode, topkDistHmo, topkLabelsHmo);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        errorCode = DealWithTopKData<DataTemp, DimSizeTemp>(compOpGroup, ockSyncUtils, selector, primaryCells,
            outRelationTable, param);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);

        if (runTimes % 100UL == 0UL) {
            OCK_VSA_HPP_LOG_INFO((runTimes * relation::PRIMARY_KEY_BATCH_SIZE) << " data selected! " <<
                selector.relatedBitSet.Count() << "/" << param->GroupRowCount() << " curPos:" << selector.curPos <<
                " processed!");

            OCK_VSA_HPP_LOG_INFO("Check memInfo, usedInfo is " <<
                *(handler->HmmMgr().GetUsedInfo(64ULL * 1024ULL * 1024ULL)));
        }
        if (selector.relatedBitSet.Count() >
            static_cast<uint64_t>(param->GroupRowCount() * relation::NEIGHBOR_SELECT_THRESHOLD)) {
            break;
        }
    }
    AddIsolateCells<DataTemp, DimSizeTemp>(selector, outRelationTable);
    return errorCode;
}
}
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif // VSA_OCKVSANEIGHBORRELATIONCREATOR_H