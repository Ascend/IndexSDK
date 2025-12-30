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


#ifndef ASCEND_INDEX_QUANTIZER_IMPL_INCLUDED
#define ASCEND_INDEX_QUANTIZER_IMPL_INCLUDED

#include <faiss/IndexFlat.h>
#include "ascend/custom/AscendIndexFlatAT.h"
#include "ascend/custom/AscendIndexFlatATInt8.h"
#include "ascend/custom/AscendClustering.h"

namespace faiss {
namespace ascend {
class AscendIndexQuantizerImpl {
public:
    AscendIndexQuantizerImpl() {}
    ~AscendIndexQuantizerImpl() {}

public:
    // where the quantizer data stored
    std::shared_ptr<IndexFlat> cpuQuantizer = nullptr;
    std::shared_ptr<AscendIndexFlatAT> npuQuantizer = nullptr;
    std::shared_ptr<AscendIndexFlatATInt8> npuInt8Quantizer = nullptr;
    std::shared_ptr<AscendClustering> npuClus = nullptr;
};
}  // namespace ascend
}  // namespace faiss
#endif  // ASCEND_INDEX_QUANTIZER_IMPL_INCLUDED