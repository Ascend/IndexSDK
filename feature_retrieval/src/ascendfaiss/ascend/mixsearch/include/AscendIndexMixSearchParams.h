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


#ifndef ASCEND_INDEX_MIX_SEARCH_PARAMS_INCLUDED
#define ASCEND_INDEX_MIX_SEARCH_PARAMS_INCLUDED

#include <vector>
namespace faiss {
namespace ascend {
const int64_t VSTAR_DEFAULT_MEM = 0x8000000;  // 0x8000000 mean 128M(resource mem pool's size)
struct AscendIndexGreatInitParams {
    AscendIndexGreatInitParams() {}
    AscendIndexGreatInitParams(int dim, int degree, int convPQM, int evaluationType, int expandingFactor)
        : dim(dim), degree(degree), convPQM(convPQM),
        evaluationType(evaluationType), expandingFactor(expandingFactor) {}

    int dim = 256;
    int degree = 50;
    int convPQM = 128;
    int evaluationType = 0;
    int expandingFactor = 300;
};

struct AscendIndexVstarInitParams {
    AscendIndexVstarInitParams() {}
    AscendIndexVstarInitParams(int dim, int subSpaceDim, int nlist,
        const std::vector<int>& deviceList, bool verbose = false, int64_t resourceSize = VSTAR_DEFAULT_MEM)
        : dim(dim), subSpaceDim(subSpaceDim), nlist(nlist), deviceList(deviceList), verbose(verbose),
        resourceSize(resourceSize) {}

    int dim = 1024;
    int subSpaceDim = 128;
    int nlist = 1024;
    std::vector<int> deviceList = {0};
    bool verbose = false;
    int64_t resourceSize = VSTAR_DEFAULT_MEM;
};

struct AscendIndexVstarHyperParams {
    AscendIndexVstarHyperParams() {}
    AscendIndexVstarHyperParams(int nProbeL1, int nProbeL2, int l3SegmentNum)
        : nProbeL1(nProbeL1), nProbeL2(nProbeL2), l3SegmentNum(l3SegmentNum) {}

    int nProbeL1 = 72;
    int nProbeL2 = 64;
    int l3SegmentNum = 512;
};
struct AscendIndexHyperParams {
    AscendIndexHyperParams() {}
    AscendIndexHyperParams(const std::string& mode, const AscendIndexVstarHyperParams& vstarHyperParam,
        int expandingFactor)
        : mode(mode), vstarHyperParam(vstarHyperParam), expandingFactor(expandingFactor) {}

    AscendIndexHyperParams(const std::string& mode, int expandingFactor)
        : mode(mode), expandingFactor(expandingFactor) {}

    std::string mode = "AKMode";
    AscendIndexVstarHyperParams vstarHyperParam{72, 64, 512};
    int expandingFactor = 150;
};

struct AscendIndexSearchParams {
    AscendIndexSearchParams(size_t n, std::vector<float>& queryData, int topK,
        std::vector<float>& dists, std::vector<int64_t>& labels)
        : n(n), queryData(queryData), topK(topK), dists(dists), labels(labels) {}

    size_t n = 0;
    std::vector<float>& queryData;
    int topK = 100;
    std::vector<float>& dists;
    std::vector<int64_t>& labels;
};


};
};

#endif