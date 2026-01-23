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

#ifndef TS_BASE_INCLUDED
#define TS_BASE_INCLUDED

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include "ascenddaemon/AscendResourcesProxy.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/AscendTensorInl.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "ascenddaemon/utils/StaticUtils.h"
#include "ascendhost/include/index/AscendIndexTS.h"
#include "common/ErrorCode.h"
#include "common/utils/AscendException.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/SocUtils.h"
#include "common/utils/OpLauncher.h"
#include "common/utils/DataType.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"
#include "ascendhost/include/impl/TSSuperBase.h"
namespace ascend {
namespace {
// Some ops require that input feature number is divisible by 1024, while 256 is a commonly used factor
// to fit the resource size.
constexpr uint32_t DEFAULT_FEATURE_BLOCK_SIZE = 1024 * 256;
constexpr int EXTRA_VAL_ALIGN = 16;
constexpr int OPS_DATA_TYPE_ALIGN = 8;
constexpr int OPS_DATA_TYPE_TIMES = 2;
constexpr int MASK_ALIGN = 8;
constexpr uint32_t TOKEN_SET_BIT = 8;
constexpr uint8_t OPS_DATA_PADDING_VAL = 64;
constexpr int64_t UPPER_LIMIT_FOR_ADD = 1.0E6;         //  max limit for each add count
constexpr int64_t UPPER_LIMIT_FOR_GET = 1.0E6;         //  max limit for each get count
constexpr uint32_t UPPER_LIMIT_FOR_TOKENNUM = 3.0E5;   //  upper limit for token count
constexpr int64_t UPPER_LIMIT_FOR_NTOTAL = 1.0E9;      //  upper limit for ntotal
constexpr uint32_t UPPER_LIMIT_FOR_DEVICEID = 1024;    //  upper limit for device id
constexpr uint32_t DEFAULT_ATTR_MEM_BLOCK_COUNT = 1;   //  Number of blocks processed each time in the original solution
constexpr uint32_t ATTR_MEM_BLOCK_COUNT = 60;          //  Number of blocks processed each time in the new solution
constexpr uint32_t UPPER_LIMIT_FOR_CUSTOM_ATTR_LEN = 32;
constexpr uint32_t UPPER_LIMIT_FOR_CUSTOM_ATTR_BLOCK_SIZE = DEFAULT_FEATURE_BLOCK_SIZE * 64;
constexpr uint32_t UPPER_LIMIT_FOR_BASE_SIZE = 8.0E8;         //  max limit for base size
const std::vector<uint32_t> MASK_BATCH{256, 128, 64, 48, 36, 32, 30, 24, 20, 18, 16, 12, 8, 7, 6, 5, 4, 3, 2, 1};
constexpr uint64_t MAX_MEM = 0x100000000; // 4GB
const int IDX_BLOCK_OFFSET = 0;
const int IDX_EXTRA_MASK_LEN = 1;
const int IDX_USE_EXTRA_MASK = 2;
const int IDX_ACTUAL_MASK_LEN = 3;
} // namespace

class TSBase : public TSSuperBase {
    // public methods
public:
    TSBase(uint32_t tokenNum, uint32_t customAttrLen, uint32_t customAttrBlockSize);
    APP_ERROR initialize(int deviceId);
    virtual ~TSBase() noexcept = default;
    int64_t getAttrTotal() const;

    void addFeatureAttrs(int64_t n, const faiss::ascend::FeatureAttr *attrs, const uint8_t *customAttr);
    APP_ERROR AddFeatureByIndice(int64_t n, const void *features,
        const faiss::ascend::FeatureAttr *attrs, const int64_t *indices, const uint8_t *customAttr,
        const faiss::ascend::ExtraValAttr *extraVal) override;
    APP_ERROR AddFeatureAttrsByIndice(int64_t n, const std::vector<std::pair<int64_t, int64_t>> &segments,
        const int64_t *indices, const faiss::ascend::FeatureAttr *attrs, const uint8_t *customAttr,
        const faiss::ascend::ExtraValAttr *extraVal);
    APP_ERROR GetFeatureByIndice(int64_t count, const int64_t *indices, int64_t *labels,
        void *features, faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *extraVal) const override;
    APP_ERROR FastDeleteFeatureByIndice(int64_t n, const int64_t *indices) override;
    APP_ERROR FastDeleteFeatureByRange(int64_t start, int n) override;
    void SetMaskValid(int64_t n, const int64_t *indices, int64_t ntotal);
    void SetMaskInvalid(int64_t start, int64_t n, const int64_t *indices, int64_t ntotal);
    std::vector<uint8_t> GetBaseMask() const override;
    void AddWithExtraValAttrs(int64_t n, const faiss::ascend::FeatureAttr *attrs, const uint8_t *customAttr,
        const faiss::ascend::ExtraValAttr *extraVal);
    void deleteAttrByIds(const std::vector<int64_t> &ids);
    void deleteCustomAttrByIds(const std::vector<int64_t> &ids);
    void getIdsByToken(uint32_t tokenId, std::vector<int64_t> &ids);
    APP_ERROR getCustomAttrByBlockId(uint32_t blockId, uint8_t *&customAttr) const override;
    APP_ERROR getFeatureAttrsByLabel(int64_t n, const int64_t *labels,
        faiss::ascend::FeatureAttr *attrs) const override;
    void generateMask(const faiss::ascend::AttrFilter *attrFilter, uint8_t *masks,
        const faiss::ascend::ExtraValFilter *extraValFilter = nullptr);
    void generateMask(const faiss::ascend::AttrFilter *attrFilter, int batchIndex,
        std::vector<std::unique_ptr<DeviceVector<uint8_t>>> &masks);
    void generateMaskWithExtra(const faiss::ascend::AttrFilter *attrFilter, int batchIndex, const uint8_t *extraMask,
        const uint64_t extraMaskLen, const bool extraMaskIsAtDevice,
        std::vector<std::unique_ptr<DeviceVector<uint8_t>>> &masks);
    void generateMaskWithExtra(const faiss::ascend::AttrFilter *attrFilter, const uint8_t *extraMask,
        const uint64_t extraMaskLen, const bool extraMaskIsAtDevice, uint8_t *masks);
    void buildAttr(const faiss::ascend::AttrFilter *attrFilter, int batch, AscendTensor<int32_t, DIMS_2> &queryTime,
        AscendTensor<uint8_t, DIMS_2> &tokenIds);
    void generateMask(int batch, int blockOffset, int blockNum,
        AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds,
        AscendTensor<uint8_t, DIMS_3> &masks);
    void generateMaskWithExtra(int batch, int blockOffset, int blockNum,
        AscendTensor<int32_t, DIMS_2> &queryTimes, AscendTensor<uint8_t, DIMS_2> &tokenIds, const uint8_t *extraMask,
        AscendTensor<uint8_t, DIMS_3> &masks,
        const AscendTensor<uint8_t, DIMS_3> &baseMaskDev = AscendTensor<uint8_t, DIMS_3>());
    virtual APP_ERROR getBaseByRange(uint32_t offset, uint32_t num, int64_t *labels, void *features,
        faiss::ascend::FeatureAttr *attributes, faiss::ascend::ExtraValAttr *extraVal) = 0;
    void setSaveHostMemory();
public:
    int64_t attrTotal{0};
    // protected methods
protected:
    void addAttrsImpl(int64_t n, const faiss::ascend::FeatureAttr *attrs, const faiss::ascend::ExtraValAttr *extraVal);
    void addCustomAttrsImpl(int64_t n, const uint8_t *customAttr);
    void removeIndvalidAttr(uint64_t originTotal, uint64_t removeCnt);
    void resetMaskGenerateComputeOp();
    void resetValMaskGenerateComputeOp();
    void runMaskGenerateCompute(const AscendTensor<int32_t, DIMS_1> &queryTime,
                                const AscendTensor<uint8_t, DIMS_1> &tokenBitSet,
                                const AscendTensor<int32_t, DIMS_1> &attrTimes,
                                const AscendTensor<int32_t, DIMS_1> &attrTokenQs,
                                const AscendTensor<uint8_t, DIMS_1> &attrTokenRs,
                                AscendTensor<uint8_t, DIMS_1> &outMask,
                                uint32_t blockCount,
                                aclrtStream stream);

    void runValMaskGenerateCompute(const AscendTensor<int32_t, DIMS_1> &queryTime,
                                   const AscendTensor<uint8_t, DIMS_1> &tokenBitSet,
                                   const AscendTensor<int32_t, DIMS_1> &attrTimes,
                                   const AscendTensor<int32_t, DIMS_1> &attrTokenQs,
                                   const AscendTensor<uint8_t, DIMS_1> &attrTokenRs,
                                   const AscendTensor<int16_t, DIMS_1> &valFilter,
                                   const AscendTensor<int16_t, DIMS_1> &baseVals,
                                   AscendTensor<uint8_t, DIMS_1> &outMask,
                                   uint32_t blockCount,
                                   aclrtStream stream);

    APP_ERROR resetBatchMaskGenerateComputeOp() const;
    APP_ERROR resetBatchValMaskGenerateComputeOp() const;

    void runBatchMaskGenerateCompute(int batch,
                                    const std::vector<const AscendTensorBase *> &input,
                                    const std::vector<const AscendTensorBase *> &output,
                                    aclrtStream stream) const;
    void runBatchMaskValGenerateCompute(int batch,
                                        const std::vector<const AscendTensorBase *> &input,
                                        const std::vector<const AscendTensorBase *> &output,
                                        aclrtStream stream) const;
    // extra mask op
    void resetExtraMaskGenerateComputeOp();
    void runExtraMaskGenerateCompute(const AscendTensor<int32_t, DIMS_1> &queryTime,
                                     const AscendTensor<uint8_t, DIMS_1> &tokenBitSet,
                                     const AscendTensor<int32_t, DIMS_1> &attrTimes,
                                     const AscendTensor<int32_t, DIMS_1> &attrTokenQs,
                                     const AscendTensor<uint8_t, DIMS_1> &attrTokenRs,
                                     const AscendTensor<uint8_t, DIMS_1> &extraMask,
                                     AscendTensor<uint8_t, DIMS_1> &outMask,
                                     uint32_t blockCount,
                                     aclrtStream stream);
    APP_ERROR resetBatchExtraMaskGenerateComputeOp(bool withBaseMask) const;
    void runBatchExtraMaskGenerateCompute(int batch,
                                        const std::vector<const AscendTensorBase *> &input,
                                        const std::vector<const AscendTensorBase *> &output,
                                        aclrtStream stream) const;
    void runBatchExtraAndBaseMaskGenerateCompute(int batch, const std::vector<const AscendTensorBase *> &input,
        const std::vector<const AscendTensorBase *> &output, aclrtStream stream) const;
    void runVectorRemoveOp(const std::vector<uint64_t> &src,
        const std::vector<uint64_t> &dst, int dataType, int copyNum);
    
    template <typename T>
    void removeNormBase(const std::vector<int64_t> &indices, size_t blockSize, size_t ntotal,
        faiss::ascend::Type normBaseType, std::vector<std::unique_ptr<DeviceVector<T>>> &normBase)
    {
        APP_LOG_INFO("TSBase remove baseShape and normbase operation start. \n");
        // move the end normbase to the locate of delete normbase
        size_t removedCnt = indices.size();
        std::vector<uint64_t> srcNormBase(removedCnt);
        std::vector<uint64_t> dstNormBase(removedCnt);

        for (size_t i = 0; i < removedCnt; i++) {
            size_t srcIdx = ntotal - i - 1;
            size_t srcBlockNum = srcIdx / blockSize;
            size_t srcLocInCurrBlock = srcIdx % blockSize;
            size_t dstBlockNum = static_cast<size_t>(indices[i]) / blockSize;
            size_t dstLocInCurrBlock = static_cast<size_t>(indices[i]) % blockSize;

            auto srcDataPtr = normBase[srcBlockNum]->data() + srcLocInCurrBlock;
            auto dstDataPtr = normBase[dstBlockNum]->data() + dstLocInCurrBlock;

            srcNormBase[i] = reinterpret_cast<uint64_t>(srcDataPtr);
            dstNormBase[i] = reinterpret_cast<uint64_t>(dstDataPtr);
        }
        // move the normbase one at a time
        runVectorRemoveOp(srcNormBase, dstNormBase, normBaseType, 1);
        APP_LOG_INFO("TSBase remove baseShape and normbase operation finished. \n");
    }

    template<typename T>
    T* calcAttrStartAddress(const std::vector<std::unique_ptr<DeviceVector<T>>> &attrVec,
                            size_t blockId, uint32_t times = 1) const
    {
        return attrVec.at(blockId / ATTR_MEM_BLOCK_COUNT)->data() +
            (blockId % ATTR_MEM_BLOCK_COUNT) * featureAttrBlockSize * times;
    }

    void ReshapeAttrsSpace(int64_t newAddCount, const uint8_t *customAttr,
        const faiss::ascend::ExtraValAttr *extraVal);
    void CopyAndSaveCustomAttrs(int64_t n, int64_t fakeTotal, const uint8_t *customAttr);
    void CopyAndSaveTSAttrs(int64_t n, int64_t fakeTotal, const faiss::ascend::FeatureAttr *attrs,
        const faiss::ascend::ExtraValAttr *extraVal);
    static APP_ERROR CheckIndices(int64_t ntotal, int64_t n, const int64_t* indices, int64_t &replaceNum,
        std::vector<std::pair<int64_t, int64_t>> &segments);

    // protected attributes
protected:
    uint32_t tokenNum = 0;
    uint32_t featureAttrBlockSize{DEFAULT_FEATURE_BLOCK_SIZE};
    uint32_t multiFeaAttrBlkSize{featureAttrBlockSize * ATTR_MEM_BLOCK_COUNT};
    bool is_initialized = false;
    bool enableValFilter = false;
    bool enableTimeFilter = true;
    bool extraMaskIsAtDevice = false;
    uint64_t extraMaskLen = 0;
    std::unique_ptr<AscendResourcesProxy> pResources;
    bool enableSaveHostMemory = false;

    // label mapping index id
    std::unordered_map<int64_t, int64_t> label2Idx;
    // token id mapping index id
    std::unordered_map<uint32_t, std::unordered_set<int64_t>> token2Idx;
    // save feature attributes
    std::vector<faiss::ascend::FeatureAttr> featureAttrs;
    std::vector<faiss::ascend::ExtraValAttr> extraValAttrs;

    std::vector<std::unique_ptr<DeviceVector<uint8_t>>> featuresBase;
    std::vector<std::unique_ptr<DeviceVector<int32_t>>> attrTime;
    std::vector<std::unique_ptr<DeviceVector<int32_t>>> attrTokenQuotient;
    std::vector<std::unique_ptr<DeviceVector<uint8_t>>> attrTokenRemainder;
    std::vector<std::unique_ptr<DeviceVector<uint8_t>>> customAttrBase;
    std::vector<std::unique_ptr<DeviceVector<int16_t>>> attrVal;

    std::unordered_map<uint32_t, std::unique_ptr<AscendOperator>> maskGenerateComputeOpMap;
    std::unordered_map<uint32_t, std::unique_ptr<AscendOperator>> extraMaskGenerateComputeOpMap;
    std::unordered_map<uint32_t, std::unique_ptr<AscendOperator>> maskValGenerateComputeOpMap;
    std::map<int, std::unique_ptr<AscendOperator>> topkComputeOps;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeOps;

    std::vector<uint8_t> baseMask; // 针对快速删除场景记录底库是否有效
    bool useBaseMask {false};

private:
    void updataMappingAfterDel(const std::vector<int64_t> &ids);
    void removeIndvalidCustomAttr(uint64_t originTotal, uint64_t removeCnt);
    void copyLastFeatureAttrs(int64_t cpyNum, int64_t oldBlockNum, int lastBlockOffset,
        std::vector<int32_t> &tmpTimes, std::vector<int32_t> &tmpTokenQs, std::vector<uint8_t> &tmpTokenRs);
    void copyLastExtraValAttrs(int64_t cpyNum, int64_t oldBlockNum, int lastBlockOffset,
        std::vector<int16_t> &tmpVals);

    void copyBlockFeatureAttrs(int64_t copyNum, int offset, int64_t i,
        std::vector<int32_t> &tmpTimes, std::vector<int32_t> &tmpTokenQs, std::vector<uint8_t> &tmpTokenRs);
    void copyBlockExtraValAttrs(int64_t copyNum, int offset, int64_t i, std::vector<int16_t> &tmpVals);

    void runExtraValOp(std::vector<uint64_t> &extraValSrcAddr, std::vector<uint64_t> &extraValDstAddr);

    void updataAttrsAfterDel(int64_t id, int lastIdx);

    void InitTimeAndTokenId(AscendTensor<int32_t, DIMS_1>& queryTime, AscendTensor<uint8_t, DIMS_1>& tokenIds,
        const faiss::ascend::AttrFilter *attrFilter);

    void InitTimeAndTokenIdWithVal(AscendTensor<int32_t, DIMS_1>& queryTime, AscendTensor<uint8_t, DIMS_1>& tokenIds,
        const faiss::ascend::AttrFilter *attrFilter, const faiss::ascend::ExtraValFilter *extraValFilter,
        AscendTensor<int16_t, DIMS_1> &valFilter);

    uint32_t customAttrLen = 0;
    uint32_t customAttrBlockSize = 0;
};
} // namespace ascend
#endif