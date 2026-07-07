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
#ifndef ASCEND_INDEX_CAGRA_H
#define ASCEND_INDEX_CAGRA_H

#include <memory>
#include <mutex>
#include <vector>

namespace faiss
{
namespace ascend
{
using APP_ERROR = int;
class AscendIndexCagraImpl;

class AscendIndexCagra
{
   public:
    AscendIndexCagra();
    AscendIndexCagra(const AscendIndexCagra&) = delete;
    AscendIndexCagra& operator=(const AscendIndexCagra&) = delete;
    virtual ~AscendIndexCagra();
    APP_ERROR Init(int dim, int graphDegree, int dataNum, int topK, const std::vector<int>& deviceList);
    APP_ERROR Add(const uint32_t* graph, const uint32_t* hash, const float* data);
    APP_ERROR QuantizeData(int n, const float* queryData, int ntotal, const float* baseData);
    APP_ERROR Search(int n, const float* queryData, int topK, float* dists, uint32_t* labels);

   private:
    std::shared_ptr<AscendIndexCagraImpl> pIndexCagraImpl;
    std::mutex mtx;
};
}  // namespace ascend
}  // namespace faiss
#endif  // ASCEND_INDEX_CAGRA_H
