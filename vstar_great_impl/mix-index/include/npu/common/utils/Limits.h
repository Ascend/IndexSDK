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


#ifndef ASCEND_LIMITS_H
#define ASCEND_LIMITS_H

#include "npu/common/AscendFp16.h"

template <typename T>
struct Limits {
};

template <>
struct Limits<float16_t> {
    static inline float16_t getMin()
    {
        uint16_t val = 0xfbffU;
        return *((float16_t *)(&val));
    }
    static inline float16_t getMax()
    {
        uint16_t val = 0x7bffU;
        return *((float16_t *)(&val));
    }
};

#endif