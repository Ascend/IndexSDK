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


#ifndef ASCEND_FP16_INCLUDED
#define ASCEND_FP16_INCLUDED

#include <string>

namespace faiss {
namespace ascendSearch {
/**
 * @ingroup fp16
 * @brief   Half precision float
 *         bit15:       1 bit SIGN      +---+-----+------------+
 *         bit14-10:    5 bit EXP       | S |EEEEE|MM MMMM MMMM|
 *         bit0-9:      10bit MAN       +---+-----+------------+
 */
struct fp16 {
    union {
        uint16_t data;  // All bits
        struct {
            uint16_t man : 10;  // mantissa
            uint16_t exp : 5;   // exponent
            uint16_t sign : 1;  // sign
        } Bits;
    };

public:
    fp16();

    explicit fp16(const uint16_t &val);

    explicit fp16(const int16_t &val);

    fp16(const fp16 &fp);

    explicit fp16(const int32_t &val);

    explicit fp16(const uint32_t &val);

    explicit fp16(const float &val);

    bool operator==(const fp16 &fp) const;

    bool operator!=(const fp16 &fp) const;

    bool operator>(const fp16 &fp) const;

    bool operator>=(const fp16 &fp) const;

    bool operator<(const fp16 &fp) const;

    bool operator<=(const fp16 &fp) const;

    fp16 &operator=(const fp16 &fp);

    fp16 &operator=(const float &val);

    operator float() const;

    static fp16 min();
    static fp16 max();
};
} // ascend
} // faiss

#endif