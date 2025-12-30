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


#ifndef OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_SLICE_ASSEMBLE_IMPL_H
#define OCK_VSA_HETERO_PIECEWISE_PROGRESSIVE_SLICE_ASSEMBLE_IMPL_H
#include <fstream>
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPKernelSystem.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSliceIdMgr.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPAssembleContext.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPFeatureWriter.h"
#include "ock/vsa/neighbor/OckVsaAnnDetailPrinter.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaHPPSearchStaticInfo.h"
#include "ock/vsa/OckVsaErrorCode.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
template <typename DataT, uint64_t DimSizeT>
void PrintOckVsaAnnFeatureDetailInfo(adapter::OckVsaAnnFeature &feature)
{
    printer::Print<DataT, DimSizeT>(feature, FORCE_PRINT);
}
template <typename DataT>
void OutputDataIntoFile(std::shared_ptr<hmm::OckHmmHMObject> hmo, const std::string &name)
{
    if (hmo == nullptr) {
        return;
    }
    std::vector<DataT> localData(hmo->GetByteSize() / sizeof(DataT));
    auto buffer = hmo->GetBuffer(hmm::OckHmmHeteroMemoryLocation::LOCAL_HOST_MEMORY, 0ULL, hmo->GetByteSize());
    if (buffer == nullptr) {
        OCK_VSA_HPP_LOG_ERROR("OutputDataIntoFile GetBuffer failed");
        return;
    }
    std::ofstream file(name.c_str(), std::ios::out | std::ios::binary);
    if (file.is_open()) {
        file.write(reinterpret_cast<char *>(buffer->Address()), localData.size() * sizeof(DataT));
        file.close();
    }
}
namespace impl {
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
OckVsaErrorCode AssembleDataByWholeSlice(hcps::handler::OckHeteroHandler &handler,
    const OckVsaHPPAssembleDataContext<DataT, DimSizeT> &context,
    adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT> &processor, OckVsaHPPSearchStaticInfo &stInfo)
{
    auto composeStartTime = std::chrono::steady_clock::now();
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    auto featurePtr = adapter::MakeOckVsaAnnFeature<DataT, DimSizeT, NormTypeByteSizeT>(handler,
        context.param.BlockRowCount(), errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    auto writer = std::make_shared<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>(
        featurePtr, handler, context.param.BlockRowCount());
    if (writer->idxMap == nullptr || writer->feature == nullptr) {
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    uint64_t needCalcDataRowCount = 0ULL;
    for (uint32_t dequeId = 0; dequeId < context.grpDatas.size(); ++dequeId) {
        auto &sliceIdSet = context.sliceIdMgr.SliceSet(dequeId);
        auto &grpData = context.grpDatas.at(dequeId);
        for (auto sliceId : sliceIdSet) {
            if (writer->feature->validateRowCount >= context.param.BlockRowCount()) {
                needCalcDataRowCount += writer->feature->validateRowCount;
                stInfo.transferRows += writer->feature->validateRowCount;
                processor.NotifyResult(writer->feature, writer->idxMap, true, errorCode);
                OCK_CHECK_RETURN_ERRORCODE(errorCode);
                writer = std::make_shared<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>(
                    adapter::MakeOckVsaAnnFeature<DataT, DimSizeT, NormTypeByteSizeT>(handler,
                    context.param.BlockRowCount(), errorCode),
                    handler, context.param.BlockRowCount());
                OCK_CHECK_RETURN_ERRORCODE(errorCode);
            }
            writer->AddWholeSlice(context, grpData, dequeId, sliceId);
        }
    }
    stInfo.composeNotifytime = ElapsedMicroSeconds(composeStartTime);
    if (writer->feature->validateRowCount > 0UL) {
        stInfo.transferRows += writer->feature->validateRowCount;
        processor.NotifyResult(writer->feature, writer->idxMap, true, errorCode);
        needCalcDataRowCount += writer->feature->validateRowCount;
    }

    OCK_VSA_HPP_LOG_DEBUG("AssembleDataByWholeSlice needCalcDataRowCount=" << needCalcDataRowCount);
    return errorCode;
}

template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>> AssembleGroupDataByRowSet(
    const OckVsaHPPAssembleDataContext<DataT, DimSizeT> &context, uint32_t grpId,
    std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>> writer,
    adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT> &processor, hcps::handler::OckHeteroHandler &handler)
{
    if (writer == nullptr || writer->idxMap == nullptr || writer->feature == nullptr) {
        return std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>> ();
    }
    auto errorCode = hmm::HMM_SUCCESS;
    auto &rowIdSet = context.sliceIdMgr.SliceSet(grpId);
    auto &grpData = context.grpDatas.at(grpId);
    uint16_t *dstNormStartAddr = reinterpret_cast<uint16_t *>(writer->feature->norm->Addr());
    uint16_t *srcNormStartAddr = reinterpret_cast<uint16_t *>(grpData.feature->norm->Addr());

    if (dstNormStartAddr == nullptr || srcNormStartAddr == nullptr) {
        return std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>> ();
    }
    for (auto rowId : rowIdSet) {
        writer->shapedFeature.AddFrom(grpData.shapedFeature, rowId);
        dstNormStartAddr[writer->feature->validateRowCount] = srcNormStartAddr[rowId];
        writer->idxMap->Add(
            context.idxMgr.GetOutterIdx(context.innerIdConvertor.ToIdx(context.groupIdDeque[grpId], rowId)));
        writer->feature->validateRowCount++;

        if (writer->feature->validateRowCount >= context.param.BlockRowCount()) {
            processor.NotifyResult(writer->feature, writer->idxMap, false, errorCode);
            if (errorCode != hmm::HMM_SUCCESS) {
                break;
            }
            writer = std::make_shared<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>(
                adapter::MakeOckVsaAnnFeature<DataT, DimSizeT, NormTypeByteSizeT>(handler,
                context.param.BlockRowCount(), errorCode),
                handler, context.param.BlockRowCount());
            if (writer == nullptr || writer->feature == nullptr || writer->feature->norm == nullptr) {
                return std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>> ();
            }
            dstNormStartAddr = reinterpret_cast<uint16_t *>(writer->feature->norm->Addr());
            if (errorCode != hmm::HMM_SUCCESS) {
                break;
            }
        }
    }
    return writer;
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
OckVsaErrorCode MultiThreadAssembleGroupDataByRowSet(hcps::handler::OckHeteroHandler &handler,
    const OckVsaHPPAssembleDataContext<DataT, DimSizeT> &context,
    adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT> &processor,
    std::vector<std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>> &writerVec)
{
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    auto stream = hcps::handler::helper::MakeStream(handler, errorCode);
    auto ops = std::make_shared<hcps::OckHeteroOperatorGroup>();
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    if (ops == nullptr || stream == nullptr) {
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    for (uint32_t grpId = 0; grpId < context.grpDatas.size(); ++grpId) {
        writerVec.push_back(std::make_shared<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>(
            adapter::MakeOckVsaAnnFeature<DataT, DimSizeT, NormTypeByteSizeT>(handler, context.param.BlockRowCount(),
            errorCode),
            handler, context.param.BlockRowCount()));
        std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>> writer = writerVec.back();
        ops->push_back(hcps::OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
            [&context, grpId, &writerVec, &processor, &handler](hcps::OckHeteroStreamContext &) {
                writerVec[grpId] = AssembleGroupDataByRowSet<DataT, DimSizeT, NormTypeByteSizeT>(context, grpId,
                    writerVec[grpId], processor, handler);
                return hmm::HMM_SUCCESS;
            }));
    }
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    stream->AddOps(*ops);
    return stream->WaitExecComplete();
}
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
OckVsaErrorCode CalcAssembleDataTopKByRowSet(hcps::handler::OckHeteroHandler &handler,
    const OckVsaHPPAssembleDataContext<DataT, DimSizeT> &context,
    adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT> &processor, OckVsaHPPSearchStaticInfo &stInfo)
{
    std::vector<std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>> leftWriterVec;
    auto ret = MultiThreadAssembleGroupDataByRowSet(handler, context, processor, leftWriterVec);
    if (ret != hmm::HMM_SUCCESS) {
        return ret;
    }
    OCK_VSA_HPP_LOG_DEBUG("MultiThreadAssembleGroupDataByRowSet complete " << context);
    return ReAssembleLeftWriterData(handler, context, processor, leftWriterVec, stInfo);
}

template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
std::shared_ptr<hcps::OckHeteroOperatorBase> CreateAssembleDataOpByFullFilter(hcps::handler::OckHeteroHandler &handler,
    const OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> &context,
    adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT> &processor,
    std::vector<std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>> &writerVec, uint32_t grpId)
{
    if (grpId >= writerVec.size()) {
        return std::shared_ptr<hcps::OckHeteroOperatorBase>();
    }
    auto writer = writerVec[grpId];
    if (writer == nullptr || writer->feature == nullptr || writer->idxMap == nullptr) {
        return std::shared_ptr<hcps::OckHeteroOperatorBase>();
    }
    return hcps::OckSimpleHeteroOperator<acladapter::OckTaskResourceType::HOST_CPU>::Create(
        [&handler, &context, &processor, &writerVec, grpId](hcps::OckHeteroStreamContext &) {
            auto &grpData = context.grpDatas.at(grpId);
            OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
            auto innerWriter = writerVec[grpId];
            for (uint32_t sliceId = 0; sliceId < context.param.GroupSliceCount(); ++sliceId) {
                uint32_t startPos = 0UL;
                while (startPos < context.param.SliceRowCount()) {
                    innerWriter->AddSliceByMaskFilter(context, grpData, grpId, sliceId, startPos);
                    if (innerWriter->feature->validateRowCount >= context.param.BlockRowCount()) {
                        PrintOckVsaAnnFeatureDetailInfo<DataT, DimSizeT>(*innerWriter->feature);
                        processor.NotifyResult(innerWriter->feature, innerWriter->idxMap, false, errorCode);
                        OCK_CHECK_RETURN_ERRORCODE(errorCode);
                        auto feature = adapter::MakeOckVsaAnnFeature<DataT, DimSizeT, NormTypeByteSizeT>(handler,
                            context.param.BlockRowCount(), errorCode);
                        OCK_CHECK_RETURN_ERRORCODE(errorCode);
                        writerVec[grpId] = std::make_shared<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>(
                            feature, handler, context.param.BlockRowCount());
                        innerWriter = writerVec[grpId];
                    }
                }
            }
            return errorCode;
        });
}

template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
OckVsaErrorCode AssembleGroupDataByFullFilter(hcps::handler::OckHeteroHandler &handler,
    const OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> &context,
    adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT> &processor,
    std::vector<std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>> &writerVec,
    std::shared_ptr<hcps::OckHeteroOperatorBase> rawSearchOp)
{
    if (rawSearchOp == nullptr) {
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    auto stream = hcps::handler::helper::MakeStream(handler, errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);
    stream->AddOp(rawSearchOp);

    hcps::OckHeteroOperatorGroup ops;
    writerVec.reserve(context.grpDatas.size());
    // 这里的grpId是 deque的下标概念， 不是innerIdx中的groupIdx概念
    for (uint32_t grpId = 0; grpId < context.grpDatas.size(); ++grpId) {
        auto featurePtr = adapter::MakeOckVsaAnnFeature<DataT, DimSizeT, NormTypeByteSizeT>(handler,
            context.param.BlockRowCount(), errorCode);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);
        auto writer = std::make_shared<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>(
            featurePtr, handler, context.param.BlockRowCount());
        writerVec.push_back(writer);
        ops.push_back(CreateAssembleDataOpByFullFilter(handler, context, processor, writerVec, grpId));
    }
    stream->AddOps(ops);
    return stream->WaitExecComplete();
}

template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
uint64_t SumOfLeftWriterDatasCount(
    const std::vector<std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>> &writerVec)
{
    uint64_t retCount = 0ULL;
    for (auto &writer : writerVec) {
        if (writer == nullptr) {
            return 0;
        }
        retCount += writer->feature->validateRowCount;
    }
    return retCount;
}

template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
OckVsaErrorCode ReAssembleLeftWriterDataUnCubeAligned(
    std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>> writer,
    hcps::handler::OckHeteroHandler &handler, const OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> &context,
    adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT> &processor,
    std::vector<std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>> &writerVec)
{
    if (writer == nullptr || writer->feature ==nullptr || writer->idxMap == nullptr) {
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    for (uint32_t grpId = 0UL; grpId < writerVec.size(); ++grpId) {
        if (writerVec[grpId] == nullptr || writerVec[grpId]->feature == nullptr) {
            return VSA_ERROR_INVALID_INPUT_PARAM;
        }
        uint32_t startPos = utils::SafeRoundDown(writerVec[grpId]->feature->validateRowCount,
            static_cast<uint32_t>(hcps::nop::CUBE_ALIGN));
        while (startPos < writerVec[grpId]->feature->validateRowCount) {
            if (writer->feature->validateRowCount >= context.param.BlockRowCount()) {
                processor.NotifyResult(writer->feature, writer->idxMap, false, errorCode);
                OCK_CHECK_RETURN_ERRORCODE(errorCode);
                writer = std::make_shared<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>(
                    adapter::MakeOckVsaAnnFeature<DataT, DimSizeT, NormTypeByteSizeT>(handler,
                    context.param.BlockRowCount(), errorCode),
                    handler, context.param.BlockRowCount());
                OCK_CHECK_RETURN_ERRORCODE(errorCode);
            }
            writer->MergeOtherUnCubeAligned(*writerVec[grpId], startPos, context.param.BlockRowCount());
        }
    }
    if (writer->feature->validateRowCount > 0UL) {
        processor.NotifyResult(writer->feature, writer->idxMap, false, errorCode);
    }
    return errorCode;
}

template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>> ReAssembleLeftWriterDataCubeAligned(
    hcps::handler::OckHeteroHandler &handler, const OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> &context,
    adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT> &processor,
    std::vector<std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>> &writerVec,
    OckVsaErrorCode &errorCode)
{
    std::shared_ptr<ock::vsa::neighbor::adapter::OckVsaAnnFeature> featurePtr =
        adapter::MakeOckVsaAnnFeature<DataT, DimSizeT, NormTypeByteSizeT>(handler,
        context.param.BlockRowCount(), errorCode);
    if (featurePtr == nullptr) {
        return std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>();
    }
    std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>> writer =
        std::make_shared<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>(
            featurePtr, handler, context.param.BlockRowCount());
    if (errorCode != hmm::HMM_SUCCESS) {
        OCK_VSA_HPP_LOG_ERROR("make OckVsaHPPFeatureWriter return error:" << errorCode);
        return std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>();
    }
    for (uint32_t grpId = 0UL; grpId < writerVec.size(); ++grpId) {
        uint32_t startPos = 0;
        if (writerVec[grpId] == nullptr || writerVec[grpId]->feature == nullptr) {
            return std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>();
        }
        while (startPos < utils::SafeRoundDown(writerVec[grpId]->feature->validateRowCount, hcps::nop::CUBE_ALIGN)) {
            NotifyResultAndCreateNewWriter(handler, context, writer, processor, errorCode);
            if (errorCode != hmm::HMM_SUCCESS) {
                OCK_VSA_HPP_LOG_ERROR("NotifyResult or make OckVsaHPPFeatureWriter return error:" << errorCode);
                return std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>();
            }
            writer->MergeOtherCubeAligned(*writerVec[grpId], startPos, context.param.BlockRowCount());
        }
    }
    return writer;
}

template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
void NotifyResultAndCreateNewWriter(hcps::handler::OckHeteroHandler &handler,
    const OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> &context,
    std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>> &writer,
    adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT> &processor, OckVsaErrorCode &errorCode)
{
    if (writer->feature == nullptr || writer->idxMap == nullptr) {
        return;
    }
    if (writer->feature->validateRowCount >= context.param.BlockRowCount()) {
        processor.NotifyResult(writer->feature, writer->idxMap, false, errorCode);
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_VSA_HPP_LOG_ERROR("NotifyResult return error:" << errorCode);
        }
        writer = std::make_shared<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>(
            adapter::MakeOckVsaAnnFeature<DataT, DimSizeT, NormTypeByteSizeT>(handler, context.param.BlockRowCount(),
            errorCode),
            handler, context.param.BlockRowCount());
        if (errorCode != hmm::HMM_SUCCESS) {
            OCK_VSA_HPP_LOG_ERROR("make OckVsaHPPFeatureWriter return error:" << errorCode);
        }
    }
}
/*
@brief 这里将多个线程组装完的结果重新组装成一片大内存
*/
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
OckVsaErrorCode ReAssembleLeftWriterData(hcps::handler::OckHeteroHandler &handler,
    const OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> &context,
    adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT> &processor,
    std::vector<std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>> &writerVec,
    OckVsaHPPSearchStaticInfo &stInfo)
{
    OckVsaErrorCode errorCode = hmm::HMM_SUCCESS;
    stInfo.transferRows += SumOfLeftWriterDatasCount<DataT, DimSizeT, NormTypeByteSizeT>(writerVec);

    auto writer = ReAssembleLeftWriterDataCubeAligned<DataT, DimSizeT, NormTypeByteSizeT>(handler, context, processor,
        writerVec, errorCode);
    OCK_CHECK_RETURN_ERRORCODE(errorCode);

    return ReAssembleLeftWriterDataUnCubeAligned<DataT, DimSizeT, NormTypeByteSizeT>(writer, handler, context,
        processor, writerVec);
}
/*
@brief 过滤条件比较严苛的场景，通常会有大量数据过滤掉，遍历、判断执行耗时占比会比较长
*/
template <typename DataT, uint64_t DimSizeT, uint64_t NormTypeByteSizeT>
OckVsaErrorCode AssembleDataByFullFilter(hcps::handler::OckHeteroHandler &handler,
    const OckVsaHPPAssembleDataContextSimple<DataT, DimSizeT> &context,
    adapter::OckVsaAnnCollectResultProcessor<DataT, DimSizeT> &processor,
    std::shared_ptr<hcps::OckHeteroOperatorBase> rawSearchOp, OckVsaHPPSearchStaticInfo &stInfo)
{
    if (rawSearchOp == nullptr) {
        return VSA_ERROR_INVALID_INPUT_PARAM;
    }
    auto filterStartTime = std::chrono::steady_clock::now();
    std::vector<std::shared_ptr<OckVsaHPPFeatureWriter<DataT, DimSizeT, NormTypeByteSizeT>>> leftWriterVec;
    auto ret = AssembleGroupDataByFullFilter(handler, context, processor, leftWriterVec, rawSearchOp);
    if (ret != hmm::HMM_SUCCESS) {
        return ret;
    }
    stInfo.filterTime = ElapsedMicroSeconds(filterStartTime);
    auto composeStartTime = std::chrono::steady_clock::now();
    OCK_VSA_HPP_LOG_DEBUG("AssembleGroupDataByFullFilter complete " << context);
    ret = ReAssembleLeftWriterData(handler, context, processor, leftWriterVec, stInfo);
    stInfo.composeNotifytime = ElapsedMicroSeconds(composeStartTime);
    return ret;
}
} // namespace impl
} // namespace hpp
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif