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


#ifndef ASCEND_INDEX_CLUSTERING_IMPL_HOST
#define ASCEND_INDEX_CLUSTERING_IMPL_HOST

#include <map>

#include "ascend/custom/AscendClustering.h"
#include "ascenddaemon/AscendResourcesProxy.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "common/ErrorCode.h"
#include "common/AscendFp16.h"

#include "index_custom/IndexFlatATSubAicpu.h"
#include "ascenddaemon/impl/IndexIVFFlat.h"

class AscendThreadPool;

namespace ascend {
class IndexFlatATAicpu;
}

namespace faiss {
namespace ascend {
using namespace ::ascend;
class AscendClusteringImpl {
public:
    AscendClusteringImpl(int d, int k, MetricType metric_type, AscendClusteringConfig config, AscendClustering *intf);

    virtual ~AscendClusteringImpl();

    void add(idx_t n, const float *x);
    void addFp32(idx_t n, const float *x);

    void train(int niter, float *centroids, bool clearData);

    void trainFp32(int niter, float *centroids, bool clearData);

    void randomCentrodataFp32(AscendTensor<float, DIMS_2> &centrodata);

    void distributedTrain(int niter, float *centroids, const std::vector<int> &deviceList, bool clearData);

    void computeCorr(float *corr, bool clearData);

    size_t getNTotal();

    void subClusAddInt8(idx_t n, const uint8_t *x, int seq, int k);

    void subClusExecInt8(int niter, uint16_t *labels, float *centroids, int batchSubNlist, size_t batchNTotal);

    void updateVDM(uint16_t *vmin, uint16_t *vdiff);

    int getSubBucketNum(int seq);

    void normalizeVecByHost(float *centrodataHost, AscendTensor<float, DIMS_2> &centrodataDev);

    AscendClusteringImpl(const AscendClusteringImpl&) = delete;
    AscendClusteringImpl& operator=(const AscendClusteringImpl&) = delete;

protected:
    void init();

    void resetKmUpdateCentroidsOp();

    void runkmUpdateCentroidsCompute(AscendTensor<float16_t, DIMS_2> &codes,
                                     AscendTensor<float16_t, DIMS_2> &centrodata,
                                     AscendTensor<uint64_t, DIMS_1> &assign,
                                     aclrtStream stream);

    void randomCentrodata(AscendTensor<float16_t, DIMS_2> &centrodata);

    void trainPostProcess(AscendTensor<float16_t, DIMS_2> &centrodata, float *centroids);

    void runCorrComputeOp(const AscendTensor<float16_t, DIMS_4>& codesShaped,
                        const AscendTensor<uint64_t, DIMS_1>& actualNum, AscendTensor<float16_t, DIMS_2>& corrResult,
                        AscendTensor<uint16_t, DIMS_2>& flag, aclrtStream stream);

    APP_ERROR resetCorrComputeOp();

    void randomCentrodataInt8(const uint8_t *data, int n, int nlist, std::vector<float16_t> &centrodata);

    size_t getNumCodes() const;

    size_t getNumCentroids() const;

    void checkParaInt8(int niter, uint16_t *labels, float *centroids, int batchSubNlist, size_t batchNTotal);

    void checkParaFp32(int niter, float *centroids);

    void execClusteringInt8(int niter, uint16_t *labels, float *centroids);

    void trainInt8(std::vector<float16_t> &centrodata, int ncentroids, int n, int niter, int reqCentroids,
        const uint8_t *data, uint16_t *labels, float *subCentroids);

    APP_ERROR resetSubcentsAccumOp();

    void runSubCntCompute(AscendTensor<float16_t, DIMS_4> &subcentsCores,
                            AscendTensor<int16_t, DIMS_3> &hassignCores,
                            AscendTensor<uint16_t, DIMS_1> &actualSize,
                            AscendTensor<float16_t, DIMS_2> &subcents,
                            AscendTensor<int16_t, DIMS_1> &hassign,
                            AscendTensor<uint16_t, DIMS_2> &flag,
                            aclrtStream stream);

private:
    std::unique_ptr<AscendResourcesProxy> resources;

    // aicpu op for topk computation
    std::unique_ptr<AscendOperator> kmUpdateCentroidsComputeOp;
    std::unique_ptr<AscendOperator> kmUpdateCentroidsComputeOpFp32;

    AscendClustering *intf_;

    MetricType metricType;
    AscendClusteringConfig config;
    size_t ntotal;

    // thread pool for multithread processing
    std::shared_ptr<AscendThreadPool> pool;

    std::vector<float16_t> codes;

    std::vector<float> codesFp32;

    void computeCentroids(size_t d, // dim
                          size_t k, // nlist
                          size_t n, // ntotal
                          const float* x,
                          const int64_t* assign, // ntotal条向量所属桶
                          float* centroids);

    std::unique_ptr<IndexFlatATAicpu> npuFlatAt;
    std::unique_ptr<IndexIVFFlat> npuFlatAtFp32; // 复用ivfflat一阶段检索

    std::unique_ptr<IndexFlatATSubAicpu> npuFlatAicpu;
    std::unique_ptr<AscendOperator> corrComputeOp;
    void corrPreProcess(float *corr);

    std::unique_ptr<AscendOperator> subCentsOp;

    AscendTensor<float16_t, DIMS_2> vDM;
    std::vector<float> vMin;
    std::vector<float> vDiff;

    bool isUpdateVDM;
    size_t updateNtotal;
    std::vector<uint8_t> codesInt8;
    std::vector<uint16_t> numCodesEachBuck;
    std::vector<uint16_t> numCentroidsEachBuck;
};
}
}

#endif