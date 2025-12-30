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


#ifndef ASCEND_CLONER_INCLUDED
#define ASCEND_CLONER_INCLUDED

#include <faiss/Index.h>
#include <faiss/clone_index.h>
#include <faiss/MetaIndexes.h>
#include <initializer_list>
#include <vector>

#include "ascend/AscendClonerOptions.h"
#include "ascend/AscendIndexInt8.h"

namespace faiss {
namespace ascend {
faiss::Index *index_ascend_to_cpu(const faiss::Index *ascend_index);

faiss::Index *index_cpu_to_ascend(std::initializer_list<int> devices,
                                  const faiss::Index *index, const AscendClonerOptions *options = nullptr);
faiss::Index *index_cpu_to_ascend(std::vector<int> devices,
                                  const faiss::Index *index, const AscendClonerOptions *options = nullptr);

// handle index int8 which not inherited from faiss::Index
faiss::Index *index_int8_ascend_to_cpu(const AscendIndexInt8 *index);

AscendIndexInt8 *index_int8_cpu_to_ascend(std::initializer_list<int> devices, const faiss::Index *index,
                                          const AscendClonerOptions *options = nullptr);
AscendIndexInt8 *index_int8_cpu_to_ascend(std::vector<int> devices, const faiss::Index *index,
                                          const AscendClonerOptions *options = nullptr);
}  // namespace ascend
}  // namespace faiss
#endif  // ASCEND_CLONER_INCLUDED
