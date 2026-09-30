/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * -------------------------------------------------------------------------
 */

#include "index/AscendCagraGraphBuilder.h"

#include "common/ErrorCode.h"
#include "common/utils/LogUtils.h"
#include "impl/AscendCagraGraphBuilderImpl.h"

using namespace ::ascend;

namespace faiss
{
namespace ascend
{
namespace
{
constexpr int MAX_CAGRA_BUILD_DIM = 3072;
}

AscendCagraGraphBuilder::AscendCagraGraphBuilder() = default;

AscendCagraGraphBuilder::~AscendCagraGraphBuilder()
{
    std::lock_guard<std::mutex> lock(mutex);
    impl.reset();
}

APP_ERROR AscendCagraGraphBuilder::Init(int dim, int graphDegree, int dataNum, const std::vector<int> &deviceList)
{
    std::lock_guard<std::mutex> lock(mutex);
    APPERR_RETURN_IF_NOT_LOG(impl == nullptr, APP_ERR_INVALID_PARAM, "CAGRA graph builder is already initialized");
    APPERR_RETURN_IF_NOT_LOG(dim > 0 && dim <= MAX_CAGRA_BUILD_DIM, APP_ERR_INVALID_PARAM,
                             "dim must be in range [1, 3072]");
    APPERR_RETURN_IF_NOT_LOG(graphDegree > 0 && graphDegree < dataNum, APP_ERR_INVALID_PARAM,
                             "graphDegree must be in range (0, dataNum)");
    APPERR_RETURN_IF_NOT_LOG(dataNum > 1, APP_ERR_INVALID_PARAM, "dataNum must be greater than 1");
    APPERR_RETURN_IF_NOT_LOG(deviceList.size() == 1, APP_ERR_INVALID_PARAM,
                             "Only 1 chip is supported for CAGRA graph building");

    AscendCagraGraphBuilderImpl::Config implConfig;
    implConfig.deviceId = deviceList[0];
    implConfig.graphDegree = static_cast<uint32_t>(graphDegree);
    implConfig.dataSize = dataNum;
    impl = std::make_unique<AscendCagraGraphBuilderImpl>(dim, implConfig);
    const APP_ERROR ret = impl->Initialize();
    if (ret != APP_ERR_OK)
    {
        impl.reset();
    }
    return ret;
}

APP_ERROR AscendCagraGraphBuilder::Build(const float *data, uint32_t *graph, const AscendCagraGraphBuildConfig &config)
{
    std::lock_guard<std::mutex> lock(mutex);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INDEX_NOT_INIT, "CAGRA graph builder is not initialized");
    APPERR_RETURN_IF_NOT_LOG(data != nullptr, APP_ERR_INVALID_PARAM, "data cannot be nullptr");
    APPERR_RETURN_IF_NOT_LOG(graph != nullptr, APP_ERR_INVALID_PARAM, "graph cannot be nullptr");

    AscendCagraGraphBuilderImpl::BuildConfig implConfig;
    implConfig.intermediateDegree = config.intermediateDegree;
    implConfig.maxIterations = config.maxIterations;
    implConfig.terminationThreshold = config.terminationThreshold;
    implConfig.guaranteeConnectivity = config.guaranteeConnectivity;
    implConfig.verbose = config.verbose;
    return impl->BuildGraph(data, graph, implConfig);
}

APP_ERROR AscendCagraGraphBuilder::BuildToFile(const float *data, const std::string &graphFilePath,
                                               const AscendCagraGraphBuildConfig &config)
{
    std::lock_guard<std::mutex> lock(mutex);
    APPERR_RETURN_IF_NOT_LOG(impl != nullptr, APP_ERR_INDEX_NOT_INIT, "CAGRA graph builder is not initialized");
    APPERR_RETURN_IF_NOT_LOG(data != nullptr, APP_ERR_INVALID_PARAM, "data cannot be nullptr");
    APPERR_RETURN_IF_NOT_LOG(!graphFilePath.empty(), APP_ERR_INVALID_PARAM, "graphFilePath cannot be empty");

    AscendCagraGraphBuilderImpl::BuildConfig implConfig;
    implConfig.intermediateDegree = config.intermediateDegree;
    implConfig.maxIterations = config.maxIterations;
    implConfig.terminationThreshold = config.terminationThreshold;
    implConfig.guaranteeConnectivity = config.guaranteeConnectivity;
    implConfig.verbose = config.verbose;
    return impl->BuildGraphToFile(data, graphFilePath, implConfig);
}
}  // namespace ascend
}  // namespace faiss
