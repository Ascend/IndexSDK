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


#ifndef ASCEND_INDEX_BINARY_FLAT_IMPL_INCLUDED
#define ASCEND_INDEX_BINARY_FLAT_IMPL_INCLUDED

#include "ascendhost/include/index/AscendIndexBinaryFlat.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <shared_mutex>

#include <faiss/Index.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include "ascend/utils/fp16.h"
#include "ascenddaemon/AscendResourcesProxy.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/AscendTensorInl.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "ascenddaemon/utils/DeviceVector.h"
#include "ascenddaemon/utils/StaticUtils.h"
#include "common/AscendFp16.h"
#include "common/ErrorCode.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/OpLauncher.h"
#include "ops/cpukernel/impl/utils/kernel_shared_def.h"

using namespace ascend;

namespace faiss {
namespace ascend {
using idx_t = int64_t;
constexpr idx_t MAX_N = 1e9;
constexpr int MAX_TOPK = 1e5;
constexpr int MAX_SEARCH = 10240; //  限定count最大值为10240
constexpr int CUBE_ALIGN = 32;      // 16 * 16 for fp 16; 16 * 32 for uint8
constexpr int Z_REGION_HEIGHT = 32; // hyper parameters H of small z
constexpr int HAMMING_CUBE_ALIGN = 4;
constexpr int LARGE_DIM = 1024;
constexpr int BLOCK_SIZE = 1024 * 256; // hyper parameters to add/search
constexpr int BURST_LEN = 128;
constexpr int ACTUAL_NUM_SIZE = 8;
constexpr int FLAG_SIZE = 16;
constexpr int PAGE_BLOCKS = 7; // every page consists of x blocks
constexpr idx_t SINGLE_ADD_MAX = 1e7;
const std::vector<uint32_t> DIMS{256, 512, 1024}; // supported DIMS
const std::vector<uint32_t> BATCH_SIZES{256, 128, 64, 32, 16, 8, 7, 6, 5, 4, 3, 2, 1};
constexpr int64_t BINARY_FLAT_MAX_MEM = 0x800000000; // 32GB
constexpr int BINARY_BYTE_SIZE = 8;
constexpr int BIG_TOPK_START = 1024; // 大topK值定义的起始值，即大于等于此值后即认为是大topK场景。
constexpr int BURST_LEN_LOW = 32;
constexpr int BURST_LEN_HIGH = 64;
constexpr int BURST_BLOCK_RATIO = 2;
constexpr int OPTIMIZE_BATCH_THRES = 48;
constexpr int SIZE_ALIGN = 8;
constexpr int CORE_NUM = 8;
class AscendIndexBinaryFlatImpl {
public:
    /* Initialize acl and resource */
    void Initialize();

    static void setRemoveFast(bool useRemoveFast);

    /* Construct from a pre-existing faiss::IndexBinaryFlat instance */
    AscendIndexBinaryFlatImpl(const faiss::IndexBinaryFlat *index, AscendIndexBinaryFlatConfig config, bool usedFloat);
    AscendIndexBinaryFlatImpl(const faiss::IndexBinaryIDMap *index, AscendIndexBinaryFlatConfig config, bool usedFloat);

    /* Construct an empty instance that can be added to */
    AscendIndexBinaryFlatImpl(int dims, AscendIndexBinaryFlatConfig config, bool usedFloat);

    void add_with_ids(idx_t n, const uint8_t *x, const idx_t *xids);

    size_t remove_ids(const faiss::IDSelector &sel);

    void search(idx_t n, const uint8_t *x, idx_t k, int32_t *distances, idx_t *labels);

    void search(idx_t n, const float *x, idx_t k, float *distances, idx_t *labels);

    void reset();

    /* Initialize ourselves from the given CPU index; will overwrite all data in ourselves */
    void copyFrom(const faiss::IndexBinaryFlat *index);
    void copyFrom(const faiss::IndexBinaryIDMap *index);

    /* Copy ourselves to the given CPU index; will overwrite all data in the index instance */
    void copyTo(faiss::IndexBinaryFlat *index) const;
    void copyTo(faiss::IndexBinaryIDMap *index) const;

    /* AscendIndexBinaryFlat object is NON-copyable */
    AscendIndexBinaryFlatImpl(const AscendIndexBinaryFlatImpl &) = delete;
    AscendIndexBinaryFlatImpl &operator=(const AscendIndexBinaryFlatImpl &) = delete;

    virtual ~AscendIndexBinaryFlatImpl() = default;

protected:
    void getVectors(uint32_t offset, uint32_t num, std::vector<uint8_t> &xb) const;

    inline void getIds(std::vector<idx_t> &idx) const
    {
        idx.assign(this->ids.begin(), this->ids.end());
    }

    void copyCode(const faiss::IndexBinaryFlat *index, const idx_t *ids = nullptr);

    void removeIdx(const std::vector<idx_t> &removeIds);

    void removeBatch(const IDSelectorBatch *rangeSel, std::vector<idx_t> &removeIds);

    void removeRange(const IDSelectorRange *rangeSel, std::vector<idx_t> &removeIds);

    void removeSingle(std::vector<idx_t> &removes, idx_t delLabel);

    void removeInvalidData(int oldTotal, int remove);

    void moveShapedForward(idx_t srcIdx, idx_t dstIdx);

    size_t removeIdsImpl(const std::vector<idx_t> &indices);

    void addVectors(int n, const uint8_t *x);

    void addWithIdsImpl(idx_t n, const uint8_t *x);

    void runTopkCompute(AscendTensor<float16_t, DIMS_3> &dists, AscendTensor<float16_t, DIMS_3> &maxDists,
                        AscendTensor<uint32_t, DIMS_3> &sizes, AscendTensor<uint16_t, DIMS_3> &flags,
                        AscendTensor<int64_t, DIMS_1> &attrs, AscendTensor<float16_t, DIMS_2> &outDists,
                        AscendTensor<int64_t, DIMS_2> &outLabel, aclrtStream stream);

    void runTopkFloatCompute(AscendTensor<float16_t, DIMS_3> &dists, AscendTensor<float16_t, DIMS_3> &maxDists,
                             AscendTensor<uint32_t, DIMS_3> &sizes, AscendTensor<uint16_t, DIMS_3> &flags,
                             AscendTensor<int64_t, DIMS_1> &attrs, AscendTensor<float16_t, DIMS_2> &outDists,
                             AscendTensor<int64_t, DIMS_2> &outLabel, aclrtStream stream);

    void runDistCompute(AscendTensor<uint8_t, DIMS_2> &queryVecs,
                        AscendTensor<uint8_t, DIMS_4> &shapedData,
                        AscendTensor<uint32_t, DIMS_2> &size,
                        AscendTensor<float16_t, DIMS_2> &outDistances,
                        AscendTensor<float16_t, DIMS_2> &outMaxDistances,
                        AscendTensor<uint16_t, DIMS_2> &flag,
                        aclrtStream stream);

    void runDistFloatCompute(AscendTensor<float16_t, DIMS_2> &queryVecs,
                             AscendTensor<uint8_t, DIMS_4> &shapedData,
                             AscendTensor<uint32_t, DIMS_2> &size,
                             AscendTensor<float16_t, DIMS_2> &outDistances,
                             AscendTensor<float16_t, DIMS_2> &outMaxDistances,
                             AscendTensor<uint16_t, DIMS_2> &flag,
                             aclrtStream stream);

    void resetDistCompOp();

    void resetDistFloatCompOp();

    void resetTopkCompOp();

    void resetTopkFloatCompOp();

    void postProcess(idx_t batch, int topK, float16_t *outDistances, int32_t *distances, idx_t *labels);

    void postProcess(int batch, int topK, float16_t *outDistances, float *distances, idx_t *labels);

    void searchPaged(int pageIdx,
                     int batch,
                     const uint8_t *x,
                     int topK,
                     AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
                     AscendTensor<idx_t, DIMS_2> &outIndicesOnDevice);

    void searchPaged(int pageIdx,
                     int batch,
                     AscendTensor<float16_t, DIMS_2> &queries,
                     int topK,
                     AscendTensor<float16_t, DIMS_2> &outDistanceOnDevice,
                     AscendTensor<idx_t, DIMS_2> &outIndicesOnDevice);

    void searchBatch(int batch, const uint8_t *x, int topK, int32_t *distances, idx_t *labels);

    void searchBatch(int batch,  const float *x, int topK, float *distances, idx_t *labels);

    void setRemoveAttr(AscendTensor<int64_t, DIMS_1> &attrsInput, int dimAlignSize, int align1, int align2) const;

    inline void largeDimSetting()
    {
        zRegionHeight = 16; // z region height shrink to 16
        burstLen = 64;      // burst length shrink to 64
    }

    static int GetBurstsOfBlock(int nq, int blockSize, int &burstLen)
    {
        burstLen = (nq > OPTIMIZE_BATCH_THRES) ? BURST_LEN_LOW : BURST_LEN_HIGH;

        return utils::divUp(blockSize, burstLen) * BURST_BLOCK_RATIO;
    }

    inline void SetShapeDim(int &reshapeDim1, int &reshapeDim2, int &align1, int &align2) const
    {
        reshapeDim1 = utils::divUp(BLOCK_SIZE, CUBE_ALIGN);
        reshapeDim2 = this->code_size / CUBE_ALIGN;
        align1 = CUBE_ALIGN;
        align2 = CUBE_ALIGN;
    }

protected:
    int deviceId{0};
    idx_t resourceSize{BINARY_FLAT_DEFAULT_MEM};
    int zRegionHeight{Z_REGION_HEIGHT};
    int burstLen{BURST_LEN};
    int d;         // < vector dimension
    int code_size; // < number of bytes per vector ( = d / 8 )
    idx_t ntotal;  // < total nb of indexed vectors
    bool verbose;  // < verbosity level
    bool isUsedFloat{false};
    static bool isRemoveFast;

    // set if the Index does not require training, or if training is done
    // already
    bool is_trained;
    // type of metric this index uses for search
    MetricType metric_type;

    std::vector<idx_t> ids;
    std::unordered_map<idx_t, idx_t> label2IdxMap;

    std::unique_ptr<AscendResourcesProxy> pResources;
    std::vector<std::unique_ptr<DeviceVector<uint8_t>>> baseShaped;

    std::map<int, std::unique_ptr<AscendOperator>> topkComputeOps;
    std::map<int, std::unique_ptr<AscendOperator>> topkComputeUint8Ops;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeOps;
    std::map<int, std::unique_ptr<AscendOperator>> distComputeFloatOps;

    mutable std::shared_mutex mtx;

private:
    void resetInner();

    void add_with_ids_inner(idx_t n, const uint8_t *x, const idx_t *xids);

    void SeparateAdd(idx_t n, const uint8_t *x);

    void CheckParam(AscendIndexBinaryFlatConfig config);
};
} // namespace ascend
} // namespace faiss

#endif /* ASCEND_INDEX_BINARY_FLAT_IMPL_INCLUDED */