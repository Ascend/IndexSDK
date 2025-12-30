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


#ifndef ASCEND_INDEXFLAT_INCLUDED
#define ASCEND_INDEXFLAT_INCLUDED

#include <vector>
#include <memory>
#include <mutex>

#include "ascenddaemon/impl/Index.h"
#include "ascenddaemon/utils/TopkOp.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "ascenddaemon/utils/AscendTensor.h"

namespace ascend {
namespace {
constexpr uint8_t BURST_LEN = 64;
constexpr uint8_t BURST_LEN_LOW = 32;
constexpr uint8_t BURST_LEN_HIGH = 64;
constexpr uint8_t BURST_BLOCK_RATIO = 2;
constexpr uint8_t OPTIMIZE_BATCH_THRES = 48;
}

struct topkOpParams {
    inline topkOpParams(AscendTensor<float16_t, DIMS_3, size_t> &dists,
                        AscendTensor<float16_t, DIMS_3, size_t> &maxdists,
                        AscendTensor<uint32_t, DIMS_3> &sizes,
                        AscendTensor<uint16_t, DIMS_3> &flags,
                        AscendTensor<int64_t, DIMS_1> &attrs,
                        AscendTensor<float16_t, DIMS_2> &outdists,
                        AscendTensor<int64_t, DIMS_2> &outlabel)
        : dists(dists), maxdists(maxdists), sizes(sizes), flags(flags), attrs(attrs),
          outdists(outdists), outlabel(outlabel) {}
    AscendTensor<float16_t, DIMS_3, size_t> &dists;
    AscendTensor<float16_t, DIMS_3, size_t> &maxdists;
    AscendTensor<uint32_t, DIMS_3> &sizes;
    AscendTensor<uint16_t, DIMS_3> &flags;
    AscendTensor<int64_t, DIMS_1> &attrs;
    AscendTensor<float16_t, DIMS_2> &outdists;
    AscendTensor<int64_t, DIMS_2> &outlabel;
};

class IndexFlat : public Index {
public:
    IndexFlat(int dim, int64_t resourceSize = -1);

    ~IndexFlat();

    virtual APP_ERROR addVectors(AscendTensor<float16_t, DIMS_2> &rawData);

    void resizeBaseShaped(size_t n);

    void resizeBaseShapedInt8(size_t n);

    APP_ERROR copyAndSaveVectors(size_t startOffset, AscendTensor<float16_t, DIMS_2> &rawData);

    APP_ERROR copyAndSaveVectors(size_t startOffset, AscendTensor<int8_t, DIMS_2> &rawData);

    APP_ERROR searchImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels) override;

    APP_ERROR reset();

    inline int getSize() const
    {
        return ntotal;
    }

    APP_ERROR getVectors(uint32_t offset, uint32_t num, std::vector<float16_t> &vectors);

    APP_ERROR getInt8VectorsAiCpu(uint32_t offset, uint32_t num, std::vector<int8_t> &vectors);

    inline int getDim() const
    {
        return dims;
    }

    inline int getBlockSize() const
    {
        return blockSize;
    }

    inline const std::vector<std::unique_ptr<DeviceVector<float16_t>>> &getBaseShaped() const
    {
        return baseShaped;
    }

    inline const std::vector<std::unique_ptr<DeviceVector<float16_t>>>& getNormBase() const
    {
        return normBase;
    }

    void getBaseEnd();

protected:
    virtual APP_ERROR searchImpl(AscendTensor<float16_t, DIMS_2> &queries, int k,
        AscendTensor<float16_t, DIMS_2> &outDistance, AscendTensor<idx_t, DIMS_2> &outIndices) = 0;

    APP_ERROR searchFilterImpl(int n, const float16_t *x, int k, float16_t *distances, idx_t *labels,
        uint8_t *masks, uint32_t maskRealLen) override;

    virtual APP_ERROR searchPaged(size_t pageId, size_t pageNum, AscendTensor<float16_t, DIMS_2> &queries,
        AscendTensor<float16_t, DIMS_2> &maxDistances, AscendTensor<int64_t, DIMS_2> &maxIndices) = 0;

    size_t getVecCapacity(size_t vecNum, size_t size) const;
    size_t getInt8VecCapacity(size_t vecNum, size_t size) const;
    APP_ERROR getVectorsAiCpu(uint32_t offset, uint32_t num, std::vector<float16_t> &vectors);
    void moveVectorForward(idx_t srcIdx, idx_t dstIdx) override;
    void releaseUnusageSpace(int oldTotal, int remove) override;

    size_t calcShapedBaseSize(idx_t totalNum);

    void runMultisearchTopkCompute(int batch, const std::vector<const AscendTensorBase *> &input,
                                   const std::vector<const AscendTensorBase *> &output, aclrtStream stream);
    void resetTransdataShapeOnlineOp();
    void resetTransdataShaperawOnlineOp();

    void execTransdataShapedrawOp(std::string &opName, aclrtStream stream, AscendTensor<float16_t, DIMS_4> &src1,
                            AscendTensor<int64_t, DIMS_1> &src2, AscendTensor<float16_t, DIMS_2> &dst);
    void execTransdataShapedOp(std::string &opName, aclrtStream stream,
                               AscendTensor<float16_t, DIMS_2> &src1,
                               AscendTensor<int64_t, DIMS_1> &src2,
                               AscendTensor<float16_t, DIMS_4> &dst);

    APP_ERROR computeMultisearchTopkParam(AscendTensor<uint32_t, DIMS_1> &indexOffsetInputs,
        AscendTensor<uint32_t, DIMS_1> &labelOffsetInputs, AscendTensor<uint16_t, DIMS_1> &reorderFlagInputs,
        std::vector<idx_t> &ntotals, std::vector<idx_t> &offsetBlocks) const;
    
    APP_ERROR runTopkOnlineOp(int batch, int flagNum, topkOpParams &params, aclrtStream stream);

    APP_ERROR resetOfflineMultisearchTopk(IndexTypeIdx topkType, int flagNum);
    APP_ERROR resetOnlineMultisearchTopk();
    APP_ERROR resetTopkOnline();

    std::vector<std::unique_ptr<DeviceVector<float16_t>>> normBase;
    std::vector<std::unique_ptr<DeviceVector<float16_t>>> baseShaped;
    std::vector<std::unique_ptr<DeviceVector<int8_t>>> baseShapedInt8;
    int blockSize;
    int devVecCapacity;
    int devInt8VecCapacity;
    int burstsOfBlock;
    int pageSize;
    uint8_t *maskData;
    uint32_t maskLen;
    int blockMaskSize;

    static int GetBurstsOfBlock(int nq, int blockSize, int &burstLen)
    {
        if (faiss::ascend::SocUtils::GetInstance().IsAscend910B()) {
            burstLen = BURST_LEN_HIGH;
        } else {
            burstLen = (nq > OPTIMIZE_BATCH_THRES) ? BURST_LEN_LOW : BURST_LEN_HIGH;
        }
        return utils::divUp(blockSize, burstLen) * BURST_BLOCK_RATIO;
    }

    // 正常来说全局算子资源应该调用resetOp、runOp接口来使用，从而保证线程安全；
    // 这里没有按照正常的使用方法，因此需要单独加锁
    static std::mutex multiSearchTopkMtx;
 
    IndexTypeIdx multiSearchTopkType { IndexTypeIdx::ITI_MAX };

private:
    APP_ERROR addVectorsAiCpu(size_t startOffset, AscendTensor<float16_t, DIMS_2> &rawData);
    APP_ERROR addInt8VectorsAicpu(size_t startOffset, AscendTensor<int8_t, DIMS_2> &rawData);

    DeviceVector<float16_t, ExpandPolicySlim> dataVec;
    DeviceVector<int8_t, ExpandPolicySlim> dataInt8Vec;
    DeviceVector<int64_t, ExpandPolicySlim> attrsVec;
};
} // namespace ascend

#endif // ASCEND_INDEXFLAT_INCLUDED
