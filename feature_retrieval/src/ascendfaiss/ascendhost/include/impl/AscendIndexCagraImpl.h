/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

#ifndef ASCEND_INDEX_CAGRA_IMPL_HOST
#define ASCEND_INDEX_CAGRA_IMPL_HOST

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ascenddaemon/AscendResourcesProxy.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "common/AscendFp16.h"
#include "common/ErrorCode.h"

using ascend::APP_ERROR;
using ascend::AscendOperator;
using ascend::AscendResourcesProxy;
using ascend::AscendTensor;
using ascend::DIMS_1;
using ascend::DIMS_2;
using ascend::DIMS_3;
using ascend::DIMS_4;

namespace faiss
{
namespace ascend
{

class AscendIndexCagraImpl
{
   public:
    AscendIndexCagraImpl(int dim, int topK, const std::vector<int>& deviceList);
    virtual ~AscendIndexCagraImpl();

    APP_ERROR Init(int graphDegree, int dataNum);

    APP_ERROR Add(const uint32_t* graph, const uint32_t* hash, const float* data);

    APP_ERROR QuantizeData(int n, const float* queryData, int ntotal, const float* baseData);

    APP_ERROR Search(int n, const float* queryData, int topK, float* dists, uint32_t* labels);

    AscendIndexCagraImpl(const AscendIndexCagraImpl&) = delete;
    AscendIndexCagraImpl& operator=(const AscendIndexCagraImpl&) = delete;

   protected:
    int degree{0};
    int dim{0};
    int topK{0};
    size_t dataNum{0};
    int codeSize{0};
    int rotatedSize{0};
    std::vector<int> deviceList;
    int64_t ascendResourceSize{128 * 1024 * 1024};
    std::map<int, std::unique_ptr<AscendOperator>> cagraSearchOp;
    std::vector<int> searchBatchSizes;
    AscendTensor<uint8_t, DIMS_1> g_code;
    std::vector<float> g_precompute;
    std::vector<uint8_t> g_rotated;

    AscendTensor<uint32_t, DIMS_2> g_graphDevice;
    AscendTensor<uint32_t, DIMS_2> g_hashDevice;
    AscendTensor<float, DIMS_2> g_dataDevice;

   private:
    APP_ERROR SearchImpl(int n, const float* queryData, int topK, float* precompute, uint8_t* rotated, uint32_t* labels,
                         float* dists);

    APP_ERROR SearchPaged(AscendTensor<float, DIMS_2>& queries, AscendTensor<float, DIMS_1>& preCompute,
                          AscendTensor<uint8_t, DIMS_1>& rotated, AscendTensor<uint32_t, DIMS_2>& outIndices,
                          AscendTensor<float, DIMS_2>& outDistances);

    void ComputeBlockDist(AscendTensor<float, DIMS_2>& queryTensor, AscendTensor<uint32_t, DIMS_2>& graphDevice,
                          AscendTensor<uint32_t, DIMS_2>& hashDevice, AscendTensor<float, DIMS_2>& data,
                          AscendTensor<float, DIMS_1>& preCompute, AscendTensor<uint8_t, DIMS_1>& preCode,
                          AscendTensor<uint8_t, DIMS_1>& rotated, AscendTensor<uint32_t, DIMS_2>& outIndices,
                          AscendTensor<float, DIMS_2>& outDistances, aclrtStream stream);

    APP_ERROR ResetCagraSearchOp();
    std::unique_ptr<AscendResourcesProxy> pResources;

    // Device information
    int deviceId;

    inline std::vector<float> MatTranspose(const std::vector<float>& A, int rows, int cols)
    {
        std::vector<float> AT(cols * rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) AT[j * rows + i] = A[i * cols + j];
        return AT;
    }
    inline std::vector<float> MatMul(const std::vector<float>& A, const std::vector<float>& B, int ARows, int ACols,
                                     int BCols)
    {
        std::vector<float> C(ARows * BCols, 0.0f);
        for (int i = 0; i < ARows; ++i)
        {
            for (int k = 0; k < ACols; ++k)
            {
                float a_ik = A[i * ACols + k];
                for (int j = 0; j < BCols; ++j) C[i * BCols + j] += a_ik * B[k * BCols + j];
            }
        }
        return C;
    }

    std::vector<float> GenerateOrthogonalMatrix(int D, unsigned int seed);
};

}  // namespace ascend
}  // namespace faiss

#endif  // ASCEND_INDEX_CAGRA_IMPL_HOST
