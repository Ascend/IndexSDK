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

#include <vector>
#include "common/ErrorCode.h"
#include "common/utils/LogUtils.h"
#include "faiss/impl/FaissAssert.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "ock/vsa/neighbor/hpp/OckVsaAnnHppSetup.h"
#include "ock/vsa/neighbor/base/OckVsaAnnFactory.h"
#include "ock/vsa/neighbor/base/OckVsaAnnCreateParam.h"
#include "ock/vsa/neighbor/base/OckVsaAnnAddFeatureParam.h"
#include "ock/vsa/attr/OckTimeSpaceAttrTrait.h"
#include "ock/hcps/modelpath/OckModelPath.h"
#include "ock/hcps/modelpath/OckSetModelPath.h"
#include "ock/vsa/neighbor/hpp/OckVsaAnnHppSetup.h"
#include "ascendhost/include/impl/TSInt8FlatHPPCosFactory.h"
#include "ascenddaemon/utils/AscendUtils.h"

namespace ascend {
template <typename _DataT, uint64_t _DimSizeT, uint64_t _NormTypeByteSizeT, typename _KeyTraitT>
class TSInt8FlatHPPCos : public TSSuperBase {
public:
    TSInt8FlatHPPCos(uint32_t deviceId, uint32_t tokenNum, uint64_t maxFeatureRowCount, uint32_t extKeyAttrsByteSize,
        uint32_t extKeyAttrBlockSize)
        : deviceId(deviceId),
          tokenNum(tokenNum),
          maxFeatureRowCount(maxFeatureRowCount),
          extKeyAttrsByteSize(extKeyAttrsByteSize),
          extKeyAttrBlockSize(extKeyAttrBlockSize)
    {
        ock::vsa::neighbor::hpp::SetUpHPPTsFactory();

        const char *modelpath = std::getenv("MX_INDEX_MODELPATH");
        if (modelpath != nullptr) {
            ock::hcps::OckModelPath::Instance().SetPath(modelpath);
            ock::hcps::OckSetModelPath::Instance().NotifyDevice();
        }
        ock::vsa::OckVsaErrorCode errorCode = ock::hmm::HMM_SUCCESS;
        auto facReg = ock::vsa::neighbor::OckVsaAnnIndexFactoryRegister<_DataT, _DimSizeT, _NormTypeByteSizeT,
            _KeyTraitT>::Instance();
        auto fac = facReg.GetFactory("HPPTS");

        FAISS_THROW_IF_NOT_MSG(fac != nullptr, "failed to create HPPTS factory");
        cpu_set_t cpuSet;
        CPU_ZERO(&cpuSet);
        CPU_SET(1U, &cpuSet); // 设置1号CPU核
        CPU_SET(2U, &cpuSet);
        CPU_SET(3U, &cpuSet);
        CPU_SET(4U, &cpuSet);

        auto param = ock::vsa::neighbor::OckVsaAnnCreateParam::Create(cpuSet, deviceId, maxFeatureRowCount, tokenNum,
            extKeyAttrsByteSize, extKeyAttrBlockSize);

        ock::vsa::attr::OckTimeSpaceAttrTrait dftTrait{ tokenNum };
        indexBase = fac->Create(param, dftTrait, errorCode);
        FAISS_THROW_IF_NOT_FMT(errorCode == ock::hmm::HMM_SUCCESS, "failed to create HPPTS, errorCode is %d",
            errorCode);
    }

    int64_t getAttrTotal() const override
    {
        return indexBase->GetFeatureNum();
    };
    APP_ERROR addFeatureWithLabels(int64_t n, const void *features, const faiss::ascend::FeatureAttr *attrs,
        const int64_t *labels, const uint8_t *customAttr, const faiss::ascend::ExtraValAttr *) override
    {
        auto addFeatureParam = ock::vsa::neighbor::OckVsaAnnAddFeatureParam<_DataT, _KeyTraitT>(n,
            reinterpret_cast<const _DataT *>(features),
            reinterpret_cast<const typename _KeyTraitT::KeyTypeTuple *>(attrs), labels, customAttr);
        return indexBase->AddFeature(addFeatureParam);
    };

    APP_ERROR delFeatureWithLabels(int64_t count, const int64_t *labels) override
    {
        return indexBase->DeleteFeatureByLabel(count, labels);
    };

    APP_ERROR getFeatureByLabel(int64_t count, const int64_t *labels, void *features) const override
    {
        return indexBase->GetFeatureByLabel(count, labels, reinterpret_cast<_DataT *>(features));
    };

    APP_ERROR deleteFeatureByToken(int64_t count, const uint32_t *tokens) override
    {
        return indexBase->DeleteFeatureByToken(count, tokens);
    };

    APP_ERROR search(uint32_t count, const void *queryFeature, const faiss::ascend::AttrFilter *attrFilter,
        bool shareAttrFilter, uint32_t topk, int64_t *labels, float *distances, uint32_t *validNums,
        bool enableTimeFilter = true, const faiss::ascend::ExtraValFilter *extraValFilter = nullptr) override
    {
        VALUE_UNUSED(extraValFilter);
        std::vector<_KeyTraitT> hppAttrFilter;
        if (shareAttrFilter) {
            hppAttrFilter.emplace_back(ock::vsa::attr::OckTimeSpaceAttrTrait(tokenNum));
            hppAttrFilter[0].minTime = attrFilter[0].timesStart;
            hppAttrFilter[0].maxTime = attrFilter[0].timesEnd;
            auto ret =
                memcpy_s(hppAttrFilter[0].bitSet.dataHolder, hppAttrFilter[0].bitSet.WordCount() * sizeof(uint64_t),
                attrFilter[0].tokenBitSet, attrFilter[0].tokenBitSetLen * sizeof(uint8_t));
            APPERR_RETURN_IF_NOT_FMT(ret == EOK, ret, "memcpy_s failed, errorCode is %d.", ret);
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                hppAttrFilter.emplace_back(ock::vsa::attr::OckTimeSpaceAttrTrait{ tokenNum });
                hppAttrFilter[i].minTime = attrFilter[i].timesStart;
                hppAttrFilter[i].maxTime = attrFilter[i].timesEnd;
                auto ret =
                    memcpy_s(hppAttrFilter[i].bitSet.dataHolder, hppAttrFilter[i].bitSet.WordCount() * sizeof(uint64_t),
                    attrFilter[i].tokenBitSet, attrFilter[i].tokenBitSetLen * sizeof(uint8_t));
                APPERR_RETURN_IF_NOT_FMT(ret == EOK, ret, "memcpy_s failed, errorCode is %d.", ret);
            }
        }

        auto queryCondition = ock::vsa::neighbor::OckVsaAnnQueryCondition<_DataT, _DimSizeT, _KeyTraitT>(count,
            reinterpret_cast<const _DataT *>(queryFeature), hppAttrFilter.data(), shareAttrFilter, topk, nullptr, 0,
            false, enableTimeFilter);

        auto outResult =
            ock::vsa::neighbor::OckVsaAnnQueryResult<_DataT, _KeyTraitT>(count, topk, labels, distances, validNums);
        return indexBase->Search(queryCondition, outResult);
    };

    APP_ERROR searchWithExtraMask(uint32_t count, const void *queryFeature, const faiss::ascend::AttrFilter *attrFilter,
        bool shareAttrFilter, uint32_t topk, const uint8_t *extraMask, uint64_t extraMaskLen, bool extraMaskIsAtDevice,
        int64_t *labels, float *distances, uint32_t *validNums, bool enableTimeFilter, const uint16_t *) override
    {
        std::vector<_KeyTraitT> hppAttrFilter;
        if (shareAttrFilter) {
            hppAttrFilter.emplace_back(ock::vsa::attr::OckTimeSpaceAttrTrait(tokenNum));
            hppAttrFilter[0].minTime = attrFilter[0].timesStart;
            hppAttrFilter[0].maxTime = attrFilter[0].timesEnd;
            auto ret =
                    memcpy_s(hppAttrFilter[0].bitSet.dataHolder, hppAttrFilter[0].bitSet.WordCount() * sizeof(uint64_t),
                             attrFilter[0].tokenBitSet, attrFilter[0].tokenBitSetLen * sizeof(uint8_t));
            APPERR_RETURN_IF_NOT_FMT(ret == EOK, ret, "memcpy_s failed, errorCode is %d.", ret);
        } else {
            for (uint32_t i = 0; i < count; ++i) {
                hppAttrFilter.emplace_back(ock::vsa::attr::OckTimeSpaceAttrTrait{ tokenNum });
                hppAttrFilter[i].minTime = attrFilter[i].timesStart;
                hppAttrFilter[i].maxTime = attrFilter[i].timesEnd;
                auto ret =
                    memcpy_s(hppAttrFilter[i].bitSet.dataHolder, hppAttrFilter[i].bitSet.WordCount() * sizeof(uint64_t),
                    attrFilter[i].tokenBitSet, attrFilter[i].tokenBitSetLen * sizeof(uint8_t));
                APPERR_RETURN_IF_NOT_FMT(ret == EOK, ret, "memcpy_s failed, errorCode is %d.", ret);
            }
        }
        auto queryCondition = ock::vsa::neighbor::OckVsaAnnQueryCondition<_DataT, _DimSizeT, _KeyTraitT>(count,
            reinterpret_cast<const _DataT *>(queryFeature), hppAttrFilter.data(),
            shareAttrFilter, topk, extraMask, extraMaskLen, extraMaskIsAtDevice, enableTimeFilter);

        auto outResult =
            ock::vsa::neighbor::OckVsaAnnQueryResult<_DataT, _KeyTraitT>(count, topk, labels, distances, validNums);
        return indexBase->Search(queryCondition, outResult);
    };

    APP_ERROR getFeatureAttrsByLabel(int64_t count, const int64_t *labels,
        faiss::ascend::FeatureAttr *attrs) const override
    {
        return indexBase->GetFeatureAttrByLabel(count, labels,
            reinterpret_cast<typename _KeyTraitT::KeyTypeTuple *>(attrs));
    };

    APP_ERROR getCustomAttrByBlockId(uint32_t blockId, uint8_t *&customAttr) const override
    {
        APP_ERROR errorCode = APP_ERR_OK;
        uintptr_t customAttrAddr = indexBase->GetCustomAttrByBlockId(blockId, errorCode);
        if (errorCode != APP_ERR_OK) {
            customAttr = nullptr;
        } else {
            customAttr = reinterpret_cast<uint8_t*>(customAttrAddr);
        }
        return errorCode;
    };

    APP_ERROR AddFeatureByIndice(int64_t, const void *, const faiss::ascend::FeatureAttr *,
        const int64_t *, const uint8_t *, const faiss::ascend::ExtraValAttr *)
    {
        APP_LOG_ERROR("Not support AddFeatureByIndice now");
        return APP_ERR_ILLEGAL_OPERATION;
    }

    APP_ERROR GetFeatureByIndice(int64_t, const int64_t *, int64_t *,
        void *, faiss::ascend::FeatureAttr *, faiss::ascend::ExtraValAttr *) const
    {
        APP_LOG_ERROR("Not support GetFeatureByIndice now");
        return APP_ERR_ILLEGAL_OPERATION;
    }

    APP_ERROR FastDeleteFeatureByIndice(int64_t, const int64_t *)
    {
        APP_LOG_ERROR("Not support FastDeleteFeatureByIndice now");
        return APP_ERR_ILLEGAL_OPERATION;
    }

    APP_ERROR FastDeleteFeatureByRange(int64_t, int)
    {
        APP_LOG_ERROR("Not support FastDeleteFeatureByRange now");
        return APP_ERR_ILLEGAL_OPERATION;
    }

    std::vector<uint8_t> GetBaseMask() const
    {
        APP_LOG_ERROR("Not support GetBaseMask now");
        return std::vector<uint8_t>();
    }

private:
    uint32_t deviceId;
    uint32_t tokenNum;
    uint64_t maxFeatureRowCount;
    uint32_t extKeyAttrsByteSize;
    uint32_t extKeyAttrBlockSize;
    std::shared_ptr<ock::vsa::neighbor::OckVsaAnnIndexBase<_DataT, _DimSizeT, _NormTypeByteSizeT, _KeyTraitT>>
        indexBase;
};

std::shared_ptr<TSSuperBase> TSInt8FlatHPPCosFactory::Create(uint32_t dims, uint32_t deviceId, uint32_t tokenNum,
    uint64_t maxFeatureRowCount, uint32_t extKeyAttrsByteSize, uint32_t extKeyAttrBlockSize)
{
    APPERR_RETURN_IF_NOT_FMT(dims == HPP_DIM, std::shared_ptr<TSSuperBase>(), "HPPTS doesn't support DIM %u.", dims);
    return std::make_shared<TSInt8FlatHPPCos<int8_t, HPP_DIM, NORM_BYTE_SIZE, ock::vsa::attr::OckTimeSpaceAttrTrait>>(
        deviceId, tokenNum, maxFeatureRowCount, extKeyAttrsByteSize, extKeyAttrBlockSize);
}
}