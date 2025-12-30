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


#ifndef ASCEND_CLONER_OPTIONS_INCLUDED
#define ASCEND_CLONER_OPTIONS_INCLUDED
#include <stdint.h>

namespace faiss {
namespace ascend {
struct AscendClonerOptions {
    AscendClonerOptions();

    long reserveVecs;
    bool verbose;
    int64_t resourceSize;

    // for sq
    bool slim = false;
    bool filterable = false;

    uint32_t indexMode;

    uint32_t blockSize;
};
}  // namespace ascend
}  // namespace faiss

#endif  // ASCEND_CLONER_OPTIONS_INCLUDED
