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


#ifndef GRAPHRETRIEVAL_FASTBITMAP_H
#define GRAPHRETRIEVAL_FASTBITMAP_H

#include "stdint.h"
#include <iostream>
#include <memory>
#include <securec.h>

namespace ascendsearch {
namespace graph {

const int INT_BYTE_CNT = 32;

class FastBitMap {
    std::unique_ptr<uint32_t[]> data = nullptr;
    int byteCnt = 0;

public:
    explicit FastBitMap(int size)
    {
        byteCnt = size / INT_BYTE_CNT + 1;
        data = std::make_unique<uint32_t[]>(byteCnt);  // create an array that holds byteCnt data
    }

    ~FastBitMap()
    {
    }

    void set(int i)  // i is the number of bits
    {
        int g = i / INT_BYTE_CNT;  // if i == 5, g = 0
        int l = i % INT_BYTE_CNT;  // 5
        data[g] += (1 << l);       // += 1 * (2^5)
    }

    bool at(int i)
    {
        int g = i / INT_BYTE_CNT;
        int l = i % INT_BYTE_CNT;
        return data[g] & (1 << l);
    }

    void clear()
    {
        if (byteCnt > 0) {
            auto ret = memset_s(data.get(), byteCnt * sizeof(uint32_t), 0, byteCnt * sizeof(uint32_t));
            if (ret != 0) {
                std::cerr << "Error in FastBitMap:clear with memset; ret = " << ret << std::endl;
                return;
            }
        }
    }
};
}  // namespace graph
}  // namespace ascendsearch
#endif  // GRAPHRETRIEVAL_FASTBITMAP_H
