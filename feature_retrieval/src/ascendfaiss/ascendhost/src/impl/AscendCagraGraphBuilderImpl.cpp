/*
 * -------------------------------------------------------------------------
 * This file is part of the IndexSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * IndexSDK is licensed under Mulan PSL v2.
 * -------------------------------------------------------------------------
 */

#include "impl/AscendCagraGraphBuilderImpl.h"

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "ascenddaemon/utils/AscendOpDesc.h"
#include "ascenddaemon/utils/AscendTensor.h"
#include "ascenddaemon/utils/AscendUtils.h"
#include "common/utils/CommonUtils.h"
#include "common/utils/LogUtils.h"
#include "common/utils/SocUtils.h"

namespace faiss
{
namespace ascend
{

AscendCagraGraphBuilderImpl::AscendCagraGraphBuilderImpl(int dim, const Config &config) : dim(dim), config(config) {}

AscendCagraGraphBuilderImpl::~AscendCagraGraphBuilderImpl() { Reset(); }

APP_ERROR AscendCagraGraphBuilderImpl::BuildGraphToFile(const float *data, const std::string &graphFilePath,
                                                        const BuildConfig &newBuildConfig)
{
    APPERR_RETURN_IF_NOT(initialized, APP_ERR_INDEX_NOT_INIT);
    APPERR_RETURN_IF_NOT(data != nullptr, APP_ERR_INVALID_PARAM);
    APPERR_RETURN_IF_NOT(!graphFilePath.empty(), APP_ERR_INVALID_PARAM);
    const size_t dataSize = static_cast<size_t>(config.dataSize);
    const size_t maxGraphElements = std::numeric_limits<size_t>::max() / sizeof(uint32_t);
    APPERR_RETURN_IF_NOT(config.graphDegree > 0 && dataSize <= maxGraphElements / config.graphDegree,
                         APP_ERR_ACL_BAD_ALLOC);
    const size_t graphElements = dataSize * config.graphDegree;
    std::unique_ptr<uint32_t[]> graph(new (std::nothrow) uint32_t[graphElements]);
    APPERR_RETURN_IF_NOT(graph != nullptr, APP_ERR_ACL_BAD_ALLOC);
    APP_ERROR ret = BuildGraph(data, graph.get(), newBuildConfig);
    APPERR_RETURN_IF_NOT(ret == APP_ERR_OK, ret);
    return SaveGraphToFile(graphFilePath, graph.get(), graphElements);
}

APP_ERROR AscendCagraGraphBuilderImpl::SaveGraphToFile(const std::string &filePath, const uint32_t *graph,
                                                       size_t numElements)
{
    FILE *file = fopen(filePath.c_str(), "wb");
    APPERR_RETURN_IF_NOT_LOG(file != nullptr, APP_ERR_INNER_ERROR, "Failed to open graph output file");
    const size_t written = fwrite(graph, sizeof(uint32_t), numElements, file);
    const int closeRet = fclose(file);
    APPERR_RETURN_IF_NOT_LOG(written == numElements && closeRet == 0, APP_ERR_INNER_ERROR,
                             "Failed to write complete CAGRA graph");
    return APP_ERR_OK;
}

}  // namespace ascend
}  // namespace faiss
