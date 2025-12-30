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

#ifndef VSA_OCK_VSA_NEIGHBOR_RELATION_H
#define VSA_OCK_VSA_NEIGHBOR_RELATION_H
#include <cstdint>
#include <algorithm>
#include <utility>
#include "ock/hmm/mgr/OckHmmHMObject.h"
#include "ock/hcps/algo/OckShape.h"
#include "ock/hcps/algo/OckElasticBitSet.h"
#include "ock/log/OckHcpsLogger.h"
#include "ock/log/OckVsaHppLogger.h"
#include "ock/vsa/neighbor/npu/OckVsaAnnNpuBlockGroup.h"

namespace ock {
namespace vsa {
namespace neighbor {
namespace hpp {
namespace relation {
const uint32_t NEIGHBOR_RELATION_COUNT_PER_CELL = 512UL;
const uint32_t PRIMARY_KEY_BATCH_SIZE = 64UL; // 必须16对齐
const uint64_t PRIMARY_KEY_SELECTION_INTERVAL = 12ULL;
const double NEIGHBOR_SELECT_THRESHOLD = 0.9;
const uint32_t THRESHOLD_AUTO_ADAPTER_BATCH_SEARCH_TIMES = 40UL;
const uint32_t FIRST_CLASS_SAMPLE_INDEX = 255UL;
const uint32_t SECOND_CLASS_SAMPLE_INDEX = 511UL;
const uint32_t SAMPLE_INTERVAL_OF_NEIGHBOR_CELL = (FIRST_CLASS_SAMPLE_INDEX + SECOND_CLASS_SAMPLE_INDEX) / 2UL;

struct OckVsaNeighborRelation {
    ~OckVsaNeighborRelation() noexcept
    {
        if (neighbor != nullptr) {
            delete[] neighbor;
        }
    }

    explicit OckVsaNeighborRelation(uint32_t primaryRowId = 0UL)
        : primary(primaryRowId), neighbor(nullptr), validateRowCount(0)
    {}

    explicit OckVsaNeighborRelation(uint32_t primaryRowId, const std::vector<uint32_t> &rowIds)
        : primary(primaryRowId),
          neighbor(new uint32_t[rowIds.size()]),
          validateRowCount(static_cast<uint32_t>(rowIds.size()))
    {
        if (neighbor != nullptr) {
            for (uint32_t i = 0; i < rowIds.size(); ++i) {
                neighbor[i] = rowIds[i];
            }
        }
    }

    OckVsaNeighborRelation(const OckVsaNeighborRelation &other)
    {
        this->primary = other.primary;
        this->validateRowCount = other.validateRowCount;
        size_t bytes = validateRowCount * sizeof(uint32_t);
        this->neighbor = new uint32_t[validateRowCount];
        if (this->neighbor != nullptr) {
            (void)memcpy_s(neighbor, bytes, other.neighbor, bytes);
        }
    }

    OckVsaNeighborRelation &operator=(const OckVsaNeighborRelation &other)
    {
        this->primary = other.primary;
        this->validateRowCount = other.validateRowCount;
        size_t bytes = validateRowCount * sizeof(uint32_t);
        if (this->neighbor != nullptr) {
            delete[] this->neighbor;
        }
        this->neighbor = new uint32_t[validateRowCount];
        if (this->neighbor != nullptr) {
            (void)memcpy_s(neighbor, bytes, other.neighbor, bytes);
        }
        return *this;
    }

    bool Isolate(void) const
    {
        return validateRowCount == 0UL;
    }

    uint32_t primary{ 0 };   // 主数据ID(本组底库中的顺序号)
    uint32_t *neighbor{ nullptr }; // 邻近数据ID(本组底库中的顺序号)
    uint32_t validateRowCount{ 0 };
};

inline std::ostream &operator << (std::ostream &os, const OckVsaNeighborRelation &data)
{
    os << "primary[" << data.primary << "]neighbor[";
    if (data.neighbor != nullptr) {
        for (uint32_t i = 0; i < data.validateRowCount; ++i) {
            if (data.neighbor[i] != 0) {
                os << "," << data.neighbor[i];
            }
        }
    }

    return os << "]";
}

struct OckVsaNeighborRelationHmoGroup {
    OckVsaNeighborRelationHmoGroup(uint32_t validRowCount = 0UL) : validateRowCount(validRowCount) {}

    void AddIsolateData(uint32_t primaryRowId)
    {
        validateRowCount++;
        relationTable.push_back(std::make_shared<OckVsaNeighborRelation>(primaryRowId));
    }

    void AddData(uint32_t primaryRowId, const std::vector<uint32_t> &rowIds)
    {
        validateRowCount++;
        relationTable.push_back(std::make_shared<OckVsaNeighborRelation>(primaryRowId, rowIds));
    };

    OckVsaNeighborRelation &At(uint32_t pos)
    {
        return *relationTable.at(pos);
    }

    const OckVsaNeighborRelation &At(uint32_t pos) const
    {
        return *relationTable.at(pos);
    }

    void SelectRelatedRowIds(const std::vector<uint32_t> &primaryRowIds, const hcps::algo::OckRefBitSet &maskData,
        const uint32_t groupRowCount, const uint32_t topK, std::unordered_set<uint32_t> &outPosSet)
    {
        for (uint32_t rowId : primaryRowIds) {
            if (rowId >= relationTable.size()) {
                continue;
            }
            auto rel = relationTable[rowId];
            if (rel->neighbor == nullptr) {
                continue;
            }
            if (maskData.At(rel->primary)) {
                outPosSet.insert(relationTable[rowId]->primary);
            }
            for (size_t j = 0; j < rel->validateRowCount; j++) {
                if (maskData.At(rel->neighbor[j])) {
                    outPosSet.insert(rel->neighbor[j]);
                }
            }
        }
    };

    std::vector<std::shared_ptr<OckVsaNeighborRelation>> relationTable{};
    uint32_t validateRowCount;
};

inline std::ostream &operator << (std::ostream &os, const OckVsaNeighborRelationHmoGroup &data)
{
    for (uint32_t i = 0; i < data.validateRowCount; ++i) {
        os << "[" << i << "]:" << data.At(i) << std::endl;
    }
    return os;
}

template <typename DataTemp, uint64_t DimSizeTemp> std::string ToString(const DataTemp *pData)
{
    std::ostringstream os;
    for (uint64_t i = 0; i < DimSizeTemp; ++i) {
        if (i != 0) {
            os << ",";
        }
        os << int16_t(pData[i]);
    }
    return os.str();
}

template <typename DataTemp, uint64_t DimSizeTemp>
inline std::shared_ptr<hmm::OckHmmHMObject> MakeShapedSampleFeature(hcps::handler::OckHeteroHandler &handler,
    const OckVsaNeighborRelationHmoGroup &groupInfo, std::shared_ptr<hmm::OckHmmHMObject> feature,
    OckVsaErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    uint64_t sampleFeatureBytes = (uint64_t)groupInfo.validateRowCount * sizeof(DataTemp) * DimSizeTemp;
    std::shared_ptr<hmm::OckHmmHMObject> hostShapedHMO =
        hcps::handler::helper::MakeHostHmo(handler, sampleFeatureBytes, errorCode);
    if (feature == nullptr || hostShapedHMO == nullptr) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    hcps::algo::OckShape<DataTemp, DimSizeTemp> shapedFeatureBlock(hostShapedHMO->Addr(), sampleFeatureBytes, 0UL);

    DataTemp *pSrcData = reinterpret_cast<DataTemp *>(feature->Addr());
    if (pSrcData == nullptr) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    for (size_t i = 0; i < groupInfo.validateRowCount; i++) {
        if (groupInfo.relationTable[i] == nullptr) {
            return std::shared_ptr<hmm::OckHmmHMObject>();
        }
        shapedFeatureBlock.AddData(pSrcData + groupInfo.relationTable[i]->primary * DimSizeTemp);
    }
    return hostShapedHMO;
};

inline std::shared_ptr<hmm::OckHmmHMObject> MakeShapedSampleNorm(hcps::handler::OckHeteroHandler &handler,
    const OckVsaNeighborRelationHmoGroup &groupInfo, std::shared_ptr<hmm::OckHmmHMObject> norm,
    OckVsaErrorCode &errorCode)
{
    if (errorCode != hmm::HMM_SUCCESS) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    uint64_t sampleNromBytes = groupInfo.validateRowCount * sizeof(uint16_t);
    std::shared_ptr<hmm::OckHmmHMObject> hostUnshapedHMO =
        hcps::handler::helper::MakeHostHmo(handler, sampleNromBytes, errorCode);
    if (norm == nullptr || hostUnshapedHMO == nullptr) {
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    uint16_t *pSrcData = reinterpret_cast<uint16_t *>(norm->Addr());
    uint16_t *pDstData = reinterpret_cast<uint16_t *>(hostUnshapedHMO->Addr());
    if (pSrcData == nullptr || pDstData == nullptr) {
        errorCode = VSA_ERROR_INPUT_PARAM_WRONG;
        return std::shared_ptr<hmm::OckHmmHMObject>();
    }
    for (size_t i = 0; i < groupInfo.validateRowCount; i++) {
        if (groupInfo.relationTable[i] == nullptr) {
            return std::shared_ptr<hmm::OckHmmHMObject>();
        }
        pDstData[i] = pSrcData[groupInfo.relationTable[i]->primary];
    }
    return hostUnshapedHMO;
};

struct OckVsaNeighborFeatureGroup {
    std::shared_ptr<hmm::OckHmmHMObject> shapedSampleFeature; // 已经分形
    std::shared_ptr<hmm::OckHmmHMObject> sampleNorm;
    uint32_t validateRowCount;
};
}
} // namespace relation
} // namespace neighbor
} // namespace vsa
} // namespace ock
#endif // VSA_OCK_VSA_NEIGHBOR_RELATION_H
