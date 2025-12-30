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


#ifndef OCK_VSA_ANN_COLLECT_RESULT_NPU_PROCESSOR_H
#define OCK_VSA_ANN_COLLECT_RESULT_NPU_PROCESSOR_H
#include <thread>
#include <atomic>
#include <mutex>
#include <list>
#include <chrono>
#include <condition_variable>
#include "acl/acl.h"
#include "ock/acladapter/utils/OckAscendFp16.h"
#include "ock/acladapter/utils/OckSyncUtils.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/hcps/handler/OckHandlerHmmHelper.h"
#include "ock/hcps/nop/l2_norm_op/OckL2NormOpRun.h"
#include "ock/hcps/hcop/topk_dist_comp_op/OckTopkDistCompOpRun.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCollectResultProcessor.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace npu {
struct NotifyTopKTaskNode {
    NotifyTopKTaskNode(std::shared_ptr<hcps::hfo::OckOneSideIdxMap> map,
        std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup> hmoSet) : idxMap(map), hmoGroup(hmoSet) {}

    std::shared_ptr<hcps::hfo::OckOneSideIdxMap> idxMap;
    std::shared_ptr<hcps::hcop::OckTopkDistCompOpHmoGroup> hmoGroup;
};
template <typename DataTemp, uint64_t DimSizeTemp>
class OckVsaAnnCollectResultNpuProcessor : public adapter::OckVsaAnnCollectResultProcessor<DataTemp, DimSizeTemp> {
public:
    using DescTopNQueue = hcps::algo::OckTopNQueue<float, uint64_t, hcps::algo::OckCompareDescAdapter<float, uint64_t>>;

    virtual ~OckVsaAnnCollectResultNpuProcessor() noexcept
    {
        if (notifyThd.get() != nullptr) {
            canceled.store(true);
            {
                std::lock_guard<std::mutex> lock(thrMutex);
                condVar.notify_all();
            }
            notifyThd->join();
            notifyThd.reset();
        }
    }
    OckVsaAnnCollectResultNpuProcessor(std::shared_ptr<hcps::handler::OckHeteroHandler> heteroHandler,
        const std::vector<DataTemp> &queryCondition, uint32_t topK)
        : handler(heteroHandler),
          queryCond(queryCondition),
          topN(topK),
          canceled(true),
          notifyEnd(false),
          lastTaskErrorCode(hmm::HMM_SUCCESS)
    {}
    OckVsaErrorCode Init(void) override
    {
        // 创建queryHmo并将queryCond数据复制到queryHmo（利用OckSyncUtils::Copy进行复制)
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        queryHmo = hcps::handler::helper::MakeDeviceHmo(*handler, queryCond.size() * sizeof(DataTemp), errorCode);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("make device hmo(queryHmo) failed, errorCode is " << errorCode);
            return errorCode;
        }
        syncUtils = std::make_shared<acladapter::OckSyncUtils>(*(handler->Service()));
        errorCode = syncUtils->Copy(reinterpret_cast<void *>(queryHmo->Addr()),
            queryHmo->GetByteSize(),
            queryCond.data(),
            queryCond.size() * sizeof(DataTemp),
            acladapter::OckMemoryCopyKind::HOST_TO_DEVICE);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("copy " << queryCond.size() * sizeof(DataTemp) << " bytes data to queryHmo(" <<
                               *queryHmo << ") failed, the errorCode is " << errorCode);
            return errorCode;
        }
        queryNorm =
            hcps::handler::helper::MakeDeviceHmo(*handler, hcps::nop::FP16_ALIGN * sizeof(OckFloat16), errorCode);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("make device hmo(queryNorm) failed, errorCode is " << errorCode);
            return errorCode;
        }
        streamBase = hcps::handler::helper::MakeStream(*handler, errorCode, hcps::OckDevStreamType::AI_CPU);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("make stream failed, errorCode is " << errorCode);
            return errorCode;
        }
        ComputeQueryNorm(errorCode);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("compute query norm failed, errorCode is " << errorCode);
            return errorCode;
        }
        topNDists = hcps::handler::helper::MakeDeviceHmo(*handler, topN * sizeof(OckFloat16), errorCode);
        topNLabels = hcps::handler::helper::MakeDeviceHmo(*handler, topN * sizeof(uint64_t), errorCode);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("make device hmo(topNDists/topNLabels) failed, errorCode is " << errorCode);
            return errorCode;
        }
        errorCode = StartRun();
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("start run thread failed, errorCode is " << errorCode);
            return errorCode;
        }
        topNQueue = DescTopNQueue::Create(topN);
        return errorCode;
    }
    void NotifyResult(std::shared_ptr<adapter::OckVsaAnnFeature> feature,
        std::shared_ptr<hcps::hfo::OckOneSideIdxMap> idxMap, bool usingMask, OckVsaErrorCode &errorCode) override
    {
        std::lock_guard<std::mutex> lock(thrMutex);
        auto hmoGroup = std::make_shared<hcps::hcop::OckTopkDistCompOpHmoGroup>(
            usingMask, 1U, DimSizeTemp, topN, feature->maxRowCount, 1U, feature->validateRowCount, 0U);
        hmoGroup->PushDataBase(feature->feature, feature->norm);
        hmoGroup->SetQueryHmos(queryHmo, queryNorm, feature->mask);
        hmoGroup->SetOutputHmos(topNDists, topNLabels);
        auto topKTaskNode = std::make_shared<NotifyTopKTaskNode>(idxMap, hmoGroup);
        taskQueue.push_back(topKTaskNode);
        canceled.store(false);
        condVar.notify_all();
        errorCode = lastTaskErrorCode;
    }
    std::vector<hcps::algo::FloatNode> NotifyResultEnd(OckVsaErrorCode &errorCode) override
    {
        notifyEnd.store(true);
        {
            std::lock_guard<std::mutex> lock(thrMutex);
            condVar.notify_all();
        }
        notifyThd->join();
        notifyThd.reset();
        errorCode = lastTaskErrorCode;
        return *topNQueue->PopAll();
    }
    std::shared_ptr<std::vector<std::vector<hcps::algo::FloatNode>>> GetTopNResults(
        std::shared_ptr<adapter::OckVsaAnnFeatureSet> featureSet, uint32_t topK, OckVsaErrorCode &errorCode) override
    {
        if (errorCode != 0) {
            return std::shared_ptr<std::vector<std::vector<hcps::algo::FloatNode>>>();
        }
        std::vector<std::vector<hcps::algo::FloatNode>> res;
        auto hmoGroup = std::make_shared<hcps::hcop::OckTopkDistCompOpHmoGroup>(featureSet->UsingMask(),
            1U, DimSizeTemp, topK, featureSet->RowCountPerFeature(), 1U, featureSet->RowCountPerFeature(), 0U);
        hmoGroup->SetQueryHmos(queryHmo, queryNorm, nullptr);
        auto topkDistsHmo = hcps::handler::helper::MakeDeviceHmo(*handler, topK * sizeof(OckFloat16), errorCode);
        auto topkLabelsHmo = hcps::handler::helper::MakeDeviceHmo(*handler, topK * sizeof(uint64_t), errorCode);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("make device hmo while compute topK failed, errorCode is " << errorCode);
            return std::shared_ptr<std::vector<std::vector<hcps::algo::FloatNode>>>();
        }
        hmoGroup->SetOutputHmos(topkDistsHmo, topkLabelsHmo);
        std::vector<OckFloat16> distances(topK);
        std::vector<uint64_t> indices(topK);
        for (uint32_t i = 0; i < featureSet->FeatureCount(); ++i) {
            auto vsaFeature = featureSet->GetFeature(i);
            hmoGroup->PushDataBase(vsaFeature.feature, vsaFeature.norm);
            hmoGroup->maskHMO = vsaFeature.mask;
            hmoGroup->blockSize = vsaFeature.maxRowCount;
            hmoGroup->ntotal = vsaFeature.validateRowCount;
            hcps::hcop::OckTopkDistCompOpRun::RunOneGroupSync(hmoGroup, streamBase, handler);
            errorCode = streamBase->WaitExecComplete();
            if (errorCode != hmm::HMM_SUCCESS) {
                OCK_HCPS_LOG_ERROR("In GetTopNResults, WaitExecComplete failed, errorCode is " << errorCode);
            }
            errorCode = syncUtils->Copy(distances.data(),
                distances.size() * sizeof(OckFloat16),
                reinterpret_cast<void *>(hmoGroup->topkDistsHmo->Addr()),
                hmoGroup->topkDistsHmo->GetByteSize(),
                acladapter::OckMemoryCopyKind::DEVICE_TO_HOST);
            if (errorCode != hmm::HMM_SUCCESS) {
                OCK_HCPS_LOG_ERROR("copy data from topKDistsHmo(" << *topkDistsHmo << ") failed, the errorCode is " <<
                                   errorCode);
                return std::shared_ptr<std::vector<std::vector<hcps::algo::FloatNode>>>();
            }
            errorCode = syncUtils->Copy(indices.data(),
                indices.size() * sizeof(uint64_t),
                reinterpret_cast<void *>(hmoGroup->topkLabelsHmo->Addr()),
                hmoGroup->topkLabelsHmo->GetByteSize(),
                acladapter::OckMemoryCopyKind::DEVICE_TO_HOST);
            if (errorCode != hmm::HMM_SUCCESS) {
                OCK_HCPS_LOG_ERROR("copy data from topkLabelsHmo(" << *topkLabelsHmo << ") failed, the errorCode is " <<
                                   errorCode);
                return std::shared_ptr<std::vector<std::vector<hcps::algo::FloatNode>>>();
            }
            std::vector<hcps::algo::FloatNode> floatNodes(std::min(topK, vsaFeature.validateRowCount));
            for (uint32_t j = 0; j < floatNodes.size(); ++j) {
                floatNodes[j].distance = acladapter::OckAscendFp16::Fp16ToFloat(distances[j]);
                floatNodes[j].idx = indices[j];
            }
            res.emplace_back(floatNodes);
            hmoGroup->featuresHmo.clear();
            hmoGroup->normsHmo.clear();
        }
        return std::make_shared<std::vector<std::vector<hcps::algo::FloatNode>>>(res);
    }

    std::shared_ptr<relation::OckVsaNeighborRelationTopNResult> GetSampleCellTopNResult(
        const adapter::OckVsaNeighborSampleInfo& sampleInfo, uint32_t topK,
        OckVsaErrorCode &errorCode) override
    {
        if (errorCode != 0) {
            return std::shared_ptr<relation::OckVsaNeighborRelationTopNResult>();
        }
        uint32_t ntotal =
            static_cast<uint32_t>(sampleInfo.blockRowCount * sampleInfo.shapedFeatureBlockListInNpu.size() +
            sampleInfo.lastBlockRowCount - sampleInfo.blockRowCount);
        uint32_t validateTopN = std::min(ntotal, topK);
        std::vector<OckFloat16> distances(validateTopN);
        std::vector<uint64_t> labels(validateTopN);
        auto topkDistsHmo =
            hcps::handler::helper::MakeDeviceHmo(*handler, validateTopN * sizeof(OckFloat16), errorCode);
        auto topkLabelsHmo = hcps::handler::helper::MakeDeviceHmo(*handler, validateTopN * sizeof(uint64_t), errorCode);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("make device hmo while GetSampleCellTopNResult failed, errorCode is " << errorCode);
            return std::shared_ptr<relation::OckVsaNeighborRelationTopNResult>();
        }
        auto hmoGroup = std::make_shared<hcps::hcop::OckTopkDistCompOpHmoGroup>(false, 1U, DimSizeTemp, validateTopN,
            sampleInfo.blockRowCount, sampleInfo.shapedFeatureBlockListInNpu.size(), ntotal, 0U);
        hmoGroup->SetQueryHmos(queryHmo, queryNorm, nullptr);
        hmoGroup->SetOutputHmos(topkDistsHmo, topkLabelsHmo);

        for (size_t i = 0; i < sampleInfo.shapedFeatureBlockListInNpu.size(); ++i) {
            hmoGroup->PushDataBase(sampleInfo.shapedFeatureBlockListInNpu[i],
                                   sampleInfo.normBlockListInNpu[i]);
        }
        hcps::hcop::OckTopkDistCompOpRun::RunOneGroupSync(hmoGroup, streamBase, handler);
        errorCode = streamBase->WaitExecComplete();
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("In GetTopNResults, WaitExecComplete failed, errorCode is " << errorCode);
        }

        errorCode = syncUtils->Copy(distances.data(), validateTopN * sizeof(OckFloat16),
                                    reinterpret_cast<void *>(hmoGroup->topkDistsHmo->Addr()),
                                    hmoGroup->topkDistsHmo->GetByteSize(),
                                    acladapter::OckMemoryCopyKind::DEVICE_TO_HOST);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("copy data from topKDistsHmo(" << *topkDistsHmo << ") failed, the errorCode is " <<
                                                              errorCode);
            return std::shared_ptr<relation::OckVsaNeighborRelationTopNResult>();
        }
        errorCode = syncUtils->Copy(labels.data(), validateTopN * sizeof(uint64_t),
                                    reinterpret_cast<void *>(hmoGroup->topkLabelsHmo->Addr()),
                                    hmoGroup->topkLabelsHmo->GetByteSize(),
                                    acladapter::OckMemoryCopyKind::DEVICE_TO_HOST);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("copy data from topkLabelsHmo(" << *topkLabelsHmo << ") failed, the errorCode is " <<
                                                               errorCode);
            return std::shared_ptr<relation::OckVsaNeighborRelationTopNResult>();
        }

        std::vector<uint32_t> outLabels(validateTopN);
        TransUint64ToUint32(labels.data(), outLabels.data(), labels.size());

        auto topNResult = relation::MakeOckVsaNeighborRelationTopNResult(outLabels.data(), distances.data(),
            validateTopN, sampleInfo.groupRowCountInfo);
        return topNResult;
    }

private:
    void TransUint64ToUint32(uint64_t* srcData, uint32_t* outData, size_t length)
    {
        if (srcData == nullptr || outData == nullptr) {
            return;
        }
        for (size_t i = 0; i <length; ++i) {
            outData[i] = static_cast<uint32_t>(srcData[i]);
        }
    }
    void ComputeQueryNorm(hmm::OckHmmErrorCode& errorCode)
    {
        auto hmoBlock = hcps::nop::OckL2NormOpRun::BuildNormHmoBlock(queryHmo, *handler, DimSizeTemp, 1U, errorCode);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("build norm hmo block failed, errorCode is " << errorCode);
            return;
        }
        errorCode = hcps::nop::OckL2NormOpRun::ComputeNormSync(hmoBlock, *handler, streamBase);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("In ComputeQueryNorm, WaitExecComplete failed, errorCode is " << errorCode);
            return;
        }
        errorCode = handler->HmmMgrPtr()->CopyHMO(*queryNorm, 0U, *hmoBlock->normResult, 0U, sizeof(OckFloat16));
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("copy data from norm result failed, errorCode is " << errorCode);
        }
    }

    OckVsaErrorCode SummaryLastResult(std::shared_ptr<NotifyTopKTaskNode> taskNode)
    {
        OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
        uint32_t validateTopN = std::min(topN, taskNode->hmoGroup->ntotal);
        std::vector<OckFloat16> distances(validateTopN);
        std::vector<uint64_t> indices(validateTopN);
        errorCode = syncUtils->Copy(distances.data(),
            distances.size() * sizeof(OckFloat16),
            reinterpret_cast<void *>(topNDists->Addr()),
            distances.size() * sizeof(OckFloat16),
            acladapter::OckMemoryCopyKind::DEVICE_TO_HOST);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("copy data from topNDists(" << *taskNode->hmoGroup->topkDistsHmo <<
                ") failed, the errorCode is " << errorCode);
            return errorCode;
        }
        errorCode = syncUtils->Copy(indices.data(),
            indices.size() * sizeof(uint64_t),
            reinterpret_cast<void *>(topNLabels->Addr()),
            indices.size() * sizeof(uint64_t),
            acladapter::OckMemoryCopyKind::DEVICE_TO_HOST);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_HCPS_LOG_ERROR("copy data from topNLabels(" << *taskNode->hmoGroup->topkLabelsHmo <<
                ") failed, the errorCode is " << errorCode);
            return errorCode;
        }
        for (uint32_t i = 0; i < validateTopN; ++i) {
            float dis = acladapter::OckAscendFp16::Fp16ToFloat(distances[i]);
            if (taskNode->idxMap->GetIdx(indices[i]) == hcps::hfo::INVALID_IDX_VALUE || dis > 1.001f) {
                OCK_HCPS_LOG_ERROR("there is not a label in idxMap at position " << indices[i] <<
                                   ", which is top " << i << ", dis is " << dis << ".");
                continue;
            }
            topNQueue->AddData(taskNode->idxMap->GetIdx(indices[i]), dis);
        }
        return errorCode;
    }

    OckVsaErrorCode StartRun(void)
    {
        OCK_HCPS_LOG_DEBUG("start run");
        std::lock_guard<std::mutex> lock(thrMutex);
        if (notifyThd.get() != nullptr) {
            return hmm::HMM_ERROR_TASK_ALREADY_RUNNING;
        }
        canceled.store(false);
        notifyThd = std::make_shared<std::thread>(
            &OckVsaAnnCollectResultNpuProcessor<DataTemp, DimSizeTemp>::DoTopKCalcFun, this);
        return hmm::HMM_SUCCESS;
    }

    void DoTopKCalcFun(void)
    {
        auto ret = aclrtSetDevice(handler->GetDeviceId());
        if (ret != ACL_SUCCESS) {
            OCK_HCPS_LOG_ERROR("acl set device(" << handler->GetDeviceId() << ") failed");
            return;
        }
        while (!canceled) {
            if (taskQueue.empty()) {
                if (notifyEnd.load()) {
                    break;
                }
                std::unique_lock<std::mutex> lock(thrMutex);
                condVar.wait_for(lock, std::chrono::milliseconds(1ULL));
                continue;
            }
            auto data = taskQueue.front();
            hcps::hcop::OckTopkDistCompOpRun::RunOneGroupSync(data->hmoGroup,
                                                              streamBase,
                                                              handler);
            lastTaskErrorCode = streamBase->WaitExecComplete();
            if (lastTaskErrorCode != hmm::HMM_SUCCESS) {
                OCK_HCPS_LOG_ERROR("wait task exec failed, the errorCode is " << lastTaskErrorCode);
                continue;
            }
            lastTaskErrorCode = SummaryLastResult(data);
            taskQueue.pop_front();
            if (lastTaskErrorCode != hmm::HMM_SUCCESS) {
                OCK_HCPS_LOG_ERROR("summary result failed, the errorCode is " << lastTaskErrorCode);
                continue;
            }
        }
        ret = aclrtResetDevice(handler->GetDeviceId());
        if (ret != ACL_SUCCESS) {
            OCK_HCPS_LOG_ERROR("acl reset device(" << handler->GetDeviceId() << ") failed");
        }
    }
    std::shared_ptr<hcps::handler::OckHeteroHandler> handler;
    const std::vector<DataTemp> &queryCond;
    const uint32_t topN;
    std::atomic_bool canceled;
    std::atomic_bool notifyEnd;
    OckVsaErrorCode lastTaskErrorCode;
    std::shared_ptr<hmm::OckHmmHMObject> queryHmo{ nullptr };
    std::shared_ptr<hmm::OckHmmHMObject> queryNorm{ nullptr };
    std::shared_ptr<hcps::OckHeteroStreamBase> streamBase{ nullptr };
    std::shared_ptr<acladapter::OckSyncUtils> syncUtils{ nullptr };
    std::shared_ptr<hmm::OckHmmHMObject> topNDists{ nullptr };
    std::shared_ptr<hmm::OckHmmHMObject> topNLabels{ nullptr };
    std::shared_ptr<DescTopNQueue> topNQueue{ nullptr };
    mutable std::mutex thrMutex{};
    std::condition_variable condVar{};
    std::shared_ptr<std::thread> notifyThd{ nullptr };
    std::list<std::shared_ptr<NotifyTopKTaskNode>> taskQueue{};
};
}  // namespace npu
namespace adapter {
template <typename DataTemp, uint64_t DimSizeTemp>
std::shared_ptr<OckVsaAnnCollectResultProcessor<DataTemp, DimSizeTemp>> OckVsaAnnCollectResultProcessor<DataTemp, DimSizeTemp>::CreateNPUProcessor(
    std::shared_ptr<hcps::handler::OckHeteroHandler> handler, const std::vector<DataTemp> &queryCond, uint32_t topN)
{
    return std::make_shared<npu::OckVsaAnnCollectResultNpuProcessor<DataTemp, DimSizeTemp>>(handler, queryCond, topN);
}
}  // namespace adapter
}  // namespace neighbor
}  // namespace vsa
}  // namespace ock
#endif