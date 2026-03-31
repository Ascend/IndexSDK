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

#ifndef ASCEND_INDEX_CAGRA_IMPL_HOST
#define ASCEND_INDEX_CAGRA_IMPL_HOST

#include <vector>
#include <string>
#include <memory>
#include <map>
#include "ascenddaemon/AscendResourcesProxy.h"
#include "ascenddaemon/utils/AscendOperator.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "common/ErrorCode.h"
#include "common/AscendFp16.h"

#include "index/AscendIndexCagra.h"
using ascend::APP_ERROR;
using ascend::AscendOperator;
using ascend::AscendResourcesProxy;
using ascend::AscendTensor;
using ascend::DIMS_1;
using ascend::DIMS_2;
using ascend::DIMS_3;
using ascend::DIMS_4;

class AscendThreadPool;

namespace faiss {
namespace ascend {

class AscendIndexCagraImpl {
public:
    AscendIndexCagraImpl(const IndexCagraInitParams& params);
    virtual ~AscendIndexCagraImpl();

    APP_ERROR Init(const IndexCagraInitParams& params, const IndexCagraSearchParams& searchParams);

    APP_ERROR AddGraph(const std::vector<uint32_t>& graphData, const std::string& saveBinPath);

    APP_ERROR Search(int n, const float* queryData, int topK, const uint32_t* graph, const uint32_t* hash,
        const float* data, float* dists, uint32_t* labels);

    AscendIndexCagraImpl(const AscendIndexCagraImpl&) = delete;
    AscendIndexCagraImpl& operator=(const AscendIndexCagraImpl&) = delete;

protected:
    int ntotal{0};
    int degree{0};
    int hashLen{0};
    int dim{0};
    int topK{0};
    size_t dataNum{0};
    int hashBitlen{0};
    int pageSize{0};
    std::vector<int> deviceList;
    int64_t ascendResourceSize;
    std::map<int, std::unique_ptr<AscendOperator>> cagraSearchOp;
    std::vector<int> searchBatchSizes;
private:

    APP_ERROR SearchImpl(int n, const float* queryData, int topK, const uint32_t* graph, const uint32_t* hash,
        const float* data, uint32_t *labels, float *dists);

    APP_ERROR SearchPaged(AscendTensor<float, DIMS_2> &queries, AscendTensor<uint32_t, DIMS_2>& graphDevice,
        AscendTensor<uint32_t, DIMS_2>& hashDevice, AscendTensor<float, DIMS_2> &data,
        AscendTensor<uint32_t, DIMS_2> &outIndices, AscendTensor<float, DIMS_2> &outDistances);
    
    void ComputeBlockDist(AscendTensor<float, DIMS_2> &queryTensor,
        AscendTensor<uint32_t, DIMS_2>& graphDevice,
        AscendTensor<uint32_t, DIMS_2>& hashDevice, AscendTensor<float, DIMS_2> &data,
        AscendTensor<uint32_t, DIMS_2> &outIndices,
        AscendTensor<float, DIMS_2> &outDistances, aclrtStream stream);

    APP_ERROR ResetCagraSearchOp();
    std::unique_ptr<AscendResourcesProxy> pResources;
    
    IndexCagraInitParams initParams;
    IndexCagraSearchParams searchParams;
    
    // Graph data
    std::vector<uint32_t> graphData;
    std::string graphBinPath;
    
    // Device information
    int deviceId;
    bool isInitialized;
};

} // namespace ascend
} // namespace faiss

#endif // ASCEND_INDEX_CAGRA_IMPL_HOST