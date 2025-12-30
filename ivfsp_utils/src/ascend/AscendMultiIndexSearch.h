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


#ifndef ASCEND_MULTI_INDEX_SEARCH_INCLUDED
#define ASCEND_MULTI_INDEX_SEARCH_INCLUDED

#include <vector>
#include <faiss/Index.h>
#include "ascendsearch/ascend/AscendIndex.h"

namespace faiss {
namespace ascendSearch {
// it used to search from mutiple AscendIndex in good performace
// the config parameters of indexes must be same except deviceList
// `x`, `distances` and `labels` need to be resident on the CPU
void Search(std::vector<Index *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, bool merged);

// it used to search from mutiple AscendIndexSQ in good performace
// the config parameters of indexes must be same except deviceList
// `x`, `distances` and `labels` need to be resident on the CPU
void Search(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, bool merged);

 
void SearchWithFilter(std::vector<AscendIndex *> indexes, idx_t n, const float *x, idx_t k,
    float *distances, idx_t *labels, const void *filters, bool merged);
} // ascend
} // faiss
#endif