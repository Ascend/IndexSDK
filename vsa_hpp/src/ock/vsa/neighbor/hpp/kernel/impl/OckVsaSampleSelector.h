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

#ifndef VSA_OCKVSASAMPLESELECTOR_H
#define VSA_OCKVSASAMPLESELECTOR_H

#include <cstdint>
#include <utility>
#include "ock/log/OckVsaHppLogger.h"
#include "ock/hcps/hop/OckSplitGroupOp.h"
#include "ock/hcps/algo/OckElasticBitSet.h"
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/hop/OckExternalQuicklySortOp.h"
#include "ock/acladapter/data/OckTaskResourceType.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuBlockGroup.h"
#include "ock/vsa/neighbor/hpp/kernel/impl/OckVsaVectorHash.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace relation {
template <typename _DataT, uint64_t DimSizeT,
    uint64_t CodeSize = utils::SafeDivUp(DimSizeT, sizeof(uint64_t) * __CHAR_BIT__)>
struct OckVsaSampleSelector {
    OckVsaSampleSelector(uint64_t rowSize)
        : rowCount(rowSize),
          curPos(0UL),
          relatedBitSet(hcps::algo::OckElasticBitSet{ rowSize }),
          primaryBitSet(hcps::algo::OckElasticBitSet{ rowSize })
    {}

    /*
    @brief 选择采样行, 必须保证16对齐，本接口不具备幂等性，第二次调用时之前返回的数据不会被选上
    @param needCount 需要的最大行数 必须16对齐
    @param pData 结果存放位置
    @return 实际得到的行数 必须16对齐
    @note bitset中选过的数据标记为1，未选中的数据标记为0
    */
    bool SelectUnusedRow(uint32_t needCount, uint32_t *pData)
    {
        if (curPos == rowCount || pData == nullptr) {
            return false;
        }
        uint32_t count = 0UL;
        for (uint64_t i = curPos; i < rowCount; ++i) {
            if (relatedBitSet.At(i)) {
                continue;
            }
            relatedBitSet.Set(i);
            primaryBitSet.Set(i);
            pData[count] = i;
            count++;
            if (count == needCount) {
                curPos = i + 1;
                return true;
            }
        }
        curPos = rowCount;

        for (uint64_t j = 0; j < rowCount; ++j) {
            if (primaryBitSet.At(j)) {
                continue;
            }
            primaryBitSet.Set(j);
            pData[count] = j;
            count++;
            if (count == needCount) {
                return true;
            }
        }
        return false;
    }

    /* *
     * @brief 主键选择优化方案的准备函数： （1）符号编码；
     * unShapedBlockDatas->feature，dim*int8_t (host HMO) -> vector<pair<int64, int64>>
     */
    void GenHammingIndexVector(const std::shared_ptr<ock::vsa::neighbor::npu::OckVsaAnnRawBlockInfo> unShapedBlockDatas,
        hcps::OckHeteroStreamBase &stream)
    {
        const uint64_t wordCount = DimSizeT / (sizeof(uint64_t) * __CHAR_BIT__);
        HammingIndexVector.resize(rowCount);
        needSortedHammingIndexVector.resize(rowCount);
        auto ops = hcps::hop::MakeOckSplitGroupAtmoicOpsNoReturn<uint64_t, acladapter::OckTaskResourceType::HOST_CPU>(
            0ULL, rowCount, 512ULL, [this, unShapedBlockDatas](uint64_t k) {
                _DataT *features = reinterpret_cast<_DataT *>(unShapedBlockDatas->feature->Addr()) + k * DimSizeT;
                OckVsaVectorHash<wordCount> signEncodeVector;
                signEncodeVector.CopyFromSign(features);
                std::pair<OckVsaVectorHash<wordCount>, uint32_t> signEncodePair{ signEncodeVector, k };
                HammingIndexVector[k] = signEncodePair;
                std::pair<OckVsaVectorHash<wordCount>, uint32_t> *signEncodePairPtr = &HammingIndexVector[k];
                needSortedHammingIndexVector[k] = signEncodePairPtr;
            });
        stream.AddOps(*ops);
        stream.WaitExecComplete();
    }

    /* *
     * @brief 主键选择优化方案的准备函数： （1）符号编码；（2）排序；（3）间隔一定数量选取主键
     */
    OckVsaErrorCode SortHammingIndexVector(std::shared_ptr<hcps::handler::OckHeteroHandler> handler,
        const std::shared_ptr<ock::vsa::neighbor::npu::OckVsaAnnRawBlockInfo> unShapedBlockDatas)
    {
        hmm::OckHmmErrorCode errorCode = hmm::HMM_SUCCESS;
        auto stream = hcps::handler::helper::MakeStream(*handler, errorCode);
        OCK_CHECK_RETURN_ERRORCODE(errorCode);

        auto start = std::chrono::high_resolution_clock::now();
        GenHammingIndexVector(unShapedBlockDatas, *stream);
        uint64_t splitThreshold = 10240ULL;
        sortedHammingIndexVector.resize(rowCount);
        hcps::hop::ExternalQuicklySort(*stream, needSortedHammingIndexVector, sortedHammingIndexVector,
            utils::PtrPairCompareAdapter(), splitThreshold);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
        OCK_VSA_HPP_LOG_INFO("Vector encoding and sorting time = " << duration.count() << " seconds.");
        return errorCode;
    }

    /* *
     * @brief 主键选择优化方案的主键函数： （3）间隔一定数量选取主键
     */
    bool SelectUnusedRow(uint32_t needCount, uint32_t *pData, uint64_t primaryKeySelectionInterval)
    {
        if (curPos >= rowCount) {
            return false;
        }
        if (pData == nullptr) {
            return false;
        }
        uint32_t count = 0UL;
        for (uint64_t i = curPos; i < rowCount; i += primaryKeySelectionInterval) {
            uint32_t curIndex = sortedHammingIndexVector[i]->second;
            if (relatedBitSet.At(curIndex)) {
                continue;
            }
            relatedBitSet.Set(curIndex);
            primaryBitSet.Set(curIndex);
            pData[count] = curIndex;
            count++;
            if (count == needCount) {
                curPos = i + primaryKeySelectionInterval;
                return true;
            }
        }
        curPos = rowCount;

        for (uint64_t j = 0; j < rowCount; j += primaryKeySelectionInterval) {
            uint32_t curIndex = sortedHammingIndexVector[j]->second;
            if (primaryBitSet.At(curIndex)) {
                continue;
            }
            primaryBitSet.Set(curIndex);
            pData[count] = curIndex;
            count++;
            if (count == needCount) {
                return true;
            }
        }
        return false;
    }

    void SetUsed(uint32_t rowId)
    {
        if (rowId >= rowCount) {
            return;
        }
        relatedBitSet.Set(rowId);
    }

    uint64_t rowCount;
    uint64_t curPos;
    std::vector<std::pair<OckVsaVectorHash<CodeSize>, uint32_t>> HammingIndexVector{}; // <符号编码, 向量序号>
    std::vector<std::pair<OckVsaVectorHash<CodeSize>, uint32_t> *> needSortedHammingIndexVector{};
    std::vector<std::pair<OckVsaVectorHash<CodeSize>, uint32_t> *> sortedHammingIndexVector{};
    hcps::algo::OckElasticBitSet relatedBitSet; // 包含主小区+邻小区
    hcps::algo::OckElasticBitSet primaryBitSet; // 只包含主小区
};

template <typename _DataT, uint64_t DimSizeT>
inline std::ostream &operator << (std::ostream &os, const OckVsaSampleSelector<_DataT, DimSizeT> &data)
{
    return os << "{'rowCount': " << data.rowCount << ", 'curPos': " << data.curPos << "}";
}
}
}
}
}
}

#endif // VSA_OCKVSASAMPLESELECTOR_H
