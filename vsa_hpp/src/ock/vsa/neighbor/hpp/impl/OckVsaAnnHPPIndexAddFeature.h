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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_ADD_FEATURE_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_ANN_INDEX_ADD_FEATURE_H
#include "ock/vsa/neighbor/hpp/impl/OckVsaAnnHPPIndexSystem.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaNeighborRelation.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaNeighborRelationCreator.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT, typename KeyTrait>
OckVsaErrorCode OckVsaAnnHPPIndex<DataT, DimSizeT, NormTypeByteSizeT, KeyTrait>::AddFeature(
    const OckVsaAnnAddFeatureParam<DataT, KeyTrait> &inputFeatureParam)
{
    if (inputFeatureParam.features == nullptr || inputFeatureParam.attributes == nullptr ||
        inputFeatureParam.labels == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("Input param cannot be nullptr");
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    auto addFeatureStartTime = std::chrono::steady_clock::now();
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    OckVsaAnnAddFeatureParam<DataT, KeyTrait> featureParam = inputFeatureParam;
    // 判断本次新增是否会超出底库最大数量, 是则返回报错，提醒用户先删除老数据清出空间
    if (this->GetFeatureNum() + featureParam.count > this->MaxFeatureRowCount()) {
        OCK_VSA_HPP_LOG_ERROR("Add features exceed maxFeatureCount, delete the old data first.");
        return VSA_ERROR_EXCEED_HPP_INDEX_MAX_FEATURE_NUMBER;
    }

    while (featureParam.count > 0) {
        uint64_t npuAddCount = std::min(featureParam.count, npuIndex->MaxFeatureRowCount() - npuIndex->GetFeatureNum());
        npuAddCount = std::min(npuAddCount, (uint64_t)param->GroupRowCount());
        if (npuAddCount == 0) {
            OCK_VSA_HPP_LOG_ERROR("Add features exceed npu maxFeatureCount[" << npuIndex->MaxFeatureRowCount() <<
                "], which is " << npuIndex->MaxFeatureRowCount() + featureParam.count);
            return VSA_ERROR_EXCEED_NPU_INDEX_MAX_FEATURE_NUMBER;
        }
        OckVsaAnnAddFeatureParam<DataT, KeyTrait> tempParam = featureParam;
        tempParam.count = npuAddCount;
        errorCode = npuIndex->AddFeature(tempParam);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);

        featureParam.count -= npuAddCount;
        featureParam.features += DimSizeT * npuAddCount;
        featureParam.attributes += npuAddCount;
        featureParam.customAttr += npuAddCount * param->ExtKeyAttrByteSize();
        featureParam.labels += npuAddCount;

        OCK_VSA_HPP_LOG_INFO("memoryInfo: " << *(handler->HmmMgr().GetUsedInfo(FRAGMENT_SIZE_THRESHOLD)));

        while (this->GetFeatureNum() >= npuIndex->MaxFeatureRowCount() &&
            npuIndex->GetFeatureNum() > MIN_FEATURE_GROUP_IN_DEVICE * param->GroupRowCount()) {
            // 创建一个AnnAddFeatureParam对象，用于记录oldestGroup删除数据空洞后剩下的有效数据
            AddFeatureParamMeta<int8_t, attr::OckTimeSpaceAttrTrait> paramStruct;
            uint64_t validCount = 0ULL;
            // 添加触发硬删除的条件：1.host预留的一个group空间已占满; 2.本次新增会发生pop
            if (hppKernel->GroupCount() > hppKernel->MaxGroupCount()) {
                validCount = hppKernel->ValidRowCountInOldestGroup();
                paramStruct.validateFeatures.resize(validCount * DimSizeT);
                paramStruct.attributes.resize(validCount);
                paramStruct.customAttr.resize(validCount * param->ExtKeyAttrByteSize());
                OCK_CHECK_RETURN_ERRORCODE(hppKernel->DeleteInvalidFeature(paramStruct));
            }
            std::deque<std::shared_ptr<hcps::hfo::OckTokenIdxMap>> tokenToRowIdsMap;
            std::vector<uint64_t> labels;
            OCK_CHECK_RETURN_ERRORCODE(errorCode);

            std::shared_ptr<npu::OckVsaAnnNpuBlockGroup> outBlockGroup =
                std::make_shared<npu::OckVsaAnnNpuBlockGroup>();
            auto rawBlockInfo = npuIndex->PopFrontBlockGroup(outBlockGroup, labels, tokenToRowIdsMap, errorCode);
            OCK_VSA_HPP_LOG_DEBUG("After PopFrontBlockGroup memoryInfo: " <<
                *(handler->HmmMgr().GetUsedInfo(FRAGMENT_SIZE_THRESHOLD)) << " errorCode=" << errorCode);
            OCK_CHECK_RETURN_ERRORCODE(errorCode);

            auto neighborRelationGroup = std::make_shared<relation::OckVsaNeighborRelationHmoGroup>(0UL);
            errorCode = ParallBuildRelationTable<DataT, DimSizeT>(*neighborRelationGroup, handler, outBlockGroup, param,
                rawBlockInfo, isThresholdInitialised);

            OCK_CHECK_RETURN_ERRORCODE(errorCode);
            errorCode = hppKernel->AddFeature(*rawBlockInfo, labels, tokenToRowIdsMap, neighborRelationGroup);

            OCK_VSA_HPP_LOG_DEBUG("After AddFeature memoryInfo: " <<
                *(handler->HmmMgr().GetUsedInfo(FRAGMENT_SIZE_THRESHOLD)));
            OCK_CHECK_RETURN_ERRORCODE(errorCode);
            rawBlockInfo.reset(); // 内存太大，及时清理
            outBlockGroup.reset();

            // 如果执行了删除数据空洞操作，将删除空洞后的有效数据重新添加回来
            if (validCount > 0ULL) {
                auto addFeatureParam = OckVsaAnnAddFeatureParam<int8_t, attr::OckTimeSpaceAttrTrait>(validCount,
                    paramStruct.validateFeatures.data(), paramStruct.attributes.data(), paramStruct.validLabels.data(),
                    paramStruct.customAttr.data());
                errorCode = AddFeature(addFeatureParam);
                OCK_CHECK_RETURN_ERRORCODE(errorCode);
            }

            auto memUsedInfo = handler->HmmMgr().GetUsedInfo(FRAGMENT_SIZE_THRESHOLD);
            OCK_VSA_HPP_LOG_INFO("After hppKernel.AddFeature npu.groupCount:" <<
                utils::SafeDivUp(npuIndex->GetFeatureNum(), param->GroupRowCount()) << " hppKernel.groupCount:" <<
                hppKernel->GroupCount() << " npu.FeatureNum:" << npuIndex->GetFeatureNum() <<
                " hppKernel.ValidRowCount:" << hppKernel->ValidRowCount() << " memoryInfo: " << *memUsedInfo);
        }
    }
    auto addFeatureTime =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - addFeatureStartTime)
            .count();
    OCK_VSA_HPP_LOG_INFO("AddFeature count=" << inputFeatureParam.count << " cost time: " << addFeatureTime << "ms"
                                             << " npu.groupCount:" <<
        utils::SafeDivUp(npuIndex->GetFeatureNum(), param->GroupRowCount()) << " kernel.GroupCount:" <<
        hppKernel->GroupCount());
    return errorCode;
}
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif