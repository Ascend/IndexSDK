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


#include <utils/FP16.h>

namespace ascendsearch {
namespace graph {
union Half {
    uint16_t data;  // All bits
    struct TagBits {
        uint16_t man : 10;  // mantissa
        uint16_t exp : 5;   // exponent
        uint16_t sign : 1;  // sign
    } Bits;
};

union Float {
    float data;  // All bits
    struct TagBits {
        uint32_t man : 23;  // mantissa
        uint32_t exp : 8;   // exponent
        uint32_t sign : 1;  // sign
    } Bits;
};

#define VALUE_CONSTRUCT(val, s, e, m) do { \
    (val).Bits.sign = s;          \
    (val).Bits.exp = e;           \
    (val).Bits.man = m;           \
} while (0)

const int HALF_EXP_BIAS = 15;          // fp16 exponent bias
const int HALF_MAX_EXP = 0x001F;       // maximum exponent value of fp16 is 31(11111)
const int HALF_MAX_MAN = 0x03FF;       // maximum mantissa value of fp16(11111 11111)
const int HALF_MAN_LEN = 10;           // the mantissa bit length of fp16 is 10
const int HALF_MAN_MASK = 0x03FF;      // mantissa mask of fp16(        11111 11111)
const int HALF_MAN_HIDE_BIT = 0x0400;  // hide bit of mantissa of fp16(   1 00000 00000)
#define HALF_SIGN_VALUE(x) ((x).Bits.sign)
#define HALF_EXP_VALUE(x) ((x).Bits.exp)
#define HALF_MAN_VALUE(x) ((x).Bits.man | (((x).Bits.exp > 0 ? 1 : 0) * HALF_MAN_HIDE_BIT))
#define HALF_IS_ZERO(x) (((x)&0x7FFF) == 0)
#define HALF_IS_NAN(x) (((x).Bits.exp == 0x1f) && ((x).Bits.man))
#define HALF_IS_INF(x) (((x).Bits.exp == 0x1f) && !((x).Bits.man))

const int FLOAT_EXP_BIAS = 127;  // fp32 exponent bias
const int FLOAT_MAN_LEN = 23;    // the mantissa bit length of float/fp32 is 23
const uint16_t SHIFT_LEN = FLOAT_MAN_LEN;
const uint32_t SHIFT_BIT = 15;
const uint32_t FLOAT_MAN_HIDE_BIT = 0x00800000u;  // hide bit of mantissa of fp32      (  1  0000 0000 0000 0000 0000
                                                  // 000)
#define FLOAT_SIGN_VALUE(x) ((x).Bits.sign)
#define FLOAT_EXP_VALUE(x) ((x).Bits.exp)
#define FLOAT_MAN_VALUE(x) ((x).Bits.man | (((x).Bits.exp > 0 ? 1 : 0) * FLOAT_MAN_HIDE_BIT))
#define FLOAT_IS_NAN(x) (((x).Bits.exp == 0xff) && ((x).Bits.man))
#define FLOAT_IS_INF(x) (((x).Bits.exp == 0xff) && !((x).Bits.man))

bool IsRoundOne(uint64_t man, uint16_t truncLen)
{
    const uint64_t maska = 0x4;
    const uint64_t maskb = 0x2;
    uint64_t mask0 = maska;
    uint64_t mask1 = maskb;
    uint64_t mask2;
    const uint16_t offset = 2;
    uint16_t shift = truncLen - offset;
    mask0 = mask0 << shift;
    mask1 = mask1 << shift;
    mask2 = mask1 - 1;

    bool lastBit = ((man & mask0) > 0);
    bool truncHigh = 0;
    bool truncLeft = 0;
    truncHigh = ((man & mask1) > 0);
    truncLeft = ((man & mask2) > 0);

    return (truncHigh && (truncLeft || lastBit));
}

void HalfNormalize(int16_t &exp, uint16_t &man)
{
    if (exp >= HALF_MAX_EXP) {
        exp = HALF_MAX_EXP - 1;
        man = HALF_MAX_MAN;
    } else if (exp == 0 && man == HALF_MAN_HIDE_BIT) {
        exp++;
        man = 0;
    }
}

static uint16_t FloatToFp16(const float &val)
{
    Float fval;
    fval.data = val;
    const uint16_t infs = 0x7c00;
    if (FLOAT_IS_INF(fval)) {
        return (infs | (FLOAT_SIGN_VALUE(fval) << SHIFT_BIT));
    }

    if (FLOAT_IS_NAN(fval)) {
        return 0x7fff;  // nan
    }

    uint32_t exp = fval.Bits.exp;
    uint32_t man = fval.Bits.man;  // 23 bit mantissa dont't need to care about denormal

    uint16_t retSign = FLOAT_SIGN_VALUE(fval);
    uint16_t retMan;
    int16_t retExp;

    // Exponent overflow/NaN converts to signed inf/NaN
    if (exp >= 0x8Fu) {  // 0x8Fu:142=127+15
        retExp = HALF_MAX_EXP - 1;
        retMan = HALF_MAX_MAN;
    } else if (exp <= 0x70u) {  // 0x70u:112=127-15 Exponent underflow converts to denormalized half or signed zero
        retExp = 0;
        if (exp >= 0x67) {  // 0x67:103=127-24 Denormal
            uint32_t fMan = (man | FLOAT_MAN_HIDE_BIT);
            uint64_t tmp = ((uint64_t)fMan) << (exp - 0x67);

            bool needRound = IsRoundOne(tmp, SHIFT_LEN);
            retMan = (uint16_t)(tmp >> SHIFT_LEN);
            if (needRound) {
                retMan++;
            }
        } else if (exp == 0x66 && man > 0) {  // 0x66:102 Denormal 0<f_v<min(Denormal)
            retMan = 1;
        } else {
            retMan = 0;
        }
    } else {  // Regular case with no overflow or underflow
        uint32_t shift = FLOAT_MAN_LEN - HALF_MAN_LEN;
        retExp = (int16_t)(exp - 0x70u);

        bool needRound = IsRoundOne(man, shift);
        retMan = (uint16_t)(man >> shift);
        if (needRound) {
            retMan++;
        }

        if (retMan & HALF_MAN_HIDE_BIT) {
            retExp++;
        }
    }

    HalfNormalize(retExp, retMan);

    Half hval;
    VALUE_CONSTRUCT(hval, retSign, retExp, retMan);
    return hval.data;
}

float Fp16ToFloat(const uint16_t &val)
{
    Float fval;
    Half hval;
    hval.data = val;

    if (HALF_IS_INF(hval)) {
        VALUE_CONSTRUCT(fval, HALF_SIGN_VALUE(hval), 0xff, 0);
        return fval.data;
    }

    if (HALF_IS_NAN(hval)) {
        VALUE_CONSTRUCT(fval, 0, 0xff, 0x7fffff);
        return fval.data;
    }

    uint16_t sign = HALF_SIGN_VALUE(hval);
    uint16_t man = HALF_MAN_VALUE(hval);
    int16_t exp = HALF_EXP_VALUE(hval);

    while (man && !(man & HALF_MAN_HIDE_BIT)) {
        man <<= 1;
        exp--;
    }

    uint32_t retExp = 0;
    uint32_t retMan = 0;
    if (!man) {
        retExp = 0;
        retMan = 0;
    } else {
        retExp = exp - HALF_EXP_BIAS + FLOAT_EXP_BIAS;
        retMan = man & HALF_MAN_MASK;
        retMan = retMan << (FLOAT_MAN_LEN - HALF_MAN_LEN);
    }

    VALUE_CONSTRUCT(fval, sign, retExp, retMan);
    return fval.data;
}

FP16::FP16() : data(0u)
{
}

FP16::FP16(const uint16_t &val) : data(val)
{
}

FP16::FP16(const int16_t &val) : data((uint16_t)val)
{
}

FP16::FP16(const FP16 &fp) : data(fp.data)
{
}

FP16::FP16(const int32_t &val) : data((uint16_t)val)
{
}

FP16::FP16(const uint32_t &val) : data((uint16_t)val)
{
}

FP16::FP16(const float &val) : data(FloatToFp16(val))
{
}

bool FP16::operator==(const FP16 &fp) const
{
    bool result = false;
    if (HALF_IS_ZERO(data) && HALF_IS_ZERO(fp.data)) {
        result = true;
    } else {
        result = (data == fp.data);  // bit compare
    }

    return result;
}

bool FP16::operator!=(const FP16 &fp) const
{
    bool result = false;
    if (HALF_IS_ZERO(data) && HALF_IS_ZERO(fp.data)) {
        result = false;
    } else {
        result = (data != fp.data);  // bit compare
    }

    return result;
}

bool FP16::operator>(const FP16 &fp) const
{
    bool result = false;

    // Compare
    if ((Bits.sign == 0) && (fp.Bits.sign > 0)) {  // +  -
        // -0=0
        result = !(HALF_IS_ZERO(data) && HALF_IS_ZERO(fp.data));
    } else if ((Bits.sign == 0) && (fp.Bits.sign == 0)) {  // + +
        if (Bits.exp > fp.Bits.exp) {                      // e_a - e_b >= 1; Va always larger than Vb
            result = true;
        } else if (Bits.exp == fp.Bits.exp) {
            result = Bits.man > fp.Bits.man;
        } else {
            result = false;
        }
    } else if ((Bits.sign > 0) && (fp.Bits.sign > 0)) {  // - -    opposite to  + +
        if (Bits.exp < fp.Bits.exp) {
            result = true;
        } else if (Bits.exp == fp.Bits.exp) {
            result = Bits.man < fp.Bits.man;
        } else {
            result = false;
        }
    } else {  // -  +
        result = false;
    }

    return result;
}

bool FP16::operator>=(const FP16 &fp) const
{
    bool result = false;
    if ((*this) > fp) {
        result = true;
    } else if ((*this) == fp) {
        result = true;
    } else {
        result = false;
    }

    return result;
}

bool FP16::operator<(const FP16 &fp) const
{
    bool result = true;
    if ((*this) >= fp) {
        result = false;
    } else {
        result = true;
    }

    return result;
}

bool FP16::operator<=(const FP16 &fp) const
{
    bool result = true;
    if ((*this) > fp) {
        result = false;
    } else {
        result = true;
    }

    return result;
}

FP16 &FP16::operator=(const FP16 &fp)
{
    if (this == &fp) {
        return *this;
    }

    data = fp.data;
    return *this;
}

FP16 &FP16::operator=(const float &val)
{
    data = FloatToFp16(val);
    return *this;
}

FP16::operator float() const
{
    return Fp16ToFloat(data);
}

}  // namespace graph
}  // namespace ascendsearch
