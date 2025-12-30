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

#include "ock/acladapter/param/OckMemoryCopyParam.h"
namespace ock {
namespace acladapter {
OckMemoryCopyParam::OckMemoryCopyParam(void *destination, size_t destinationMax, const void *source, size_t copySize,
    OckMemoryCopyKind copyKind)
    : dst(destination), destMax(destinationMax), src(source), count(copySize), kind(copyKind)
{}

void *OckMemoryCopyParam::DestAddr(void)
{
    return dst;
}
const void *OckMemoryCopyParam::DestAddr(void) const
{
    return dst;
}
size_t OckMemoryCopyParam::DestMax(void) const
{
    return destMax;
}
const void *OckMemoryCopyParam::SrcAddr(void) const
{
    return src;
}
size_t OckMemoryCopyParam::SrcCount(void) const
{
    return count;
}
OckMemoryCopyKind OckMemoryCopyParam::Kind(void) const
{
    return kind;
}
std::ostream &operator<<(std::ostream &os, const OckMemoryCopyParam &param)
{
    return os << "{'count':" << param.SrcCount() << ",'kind':" << param.Kind() << "}";
}
}  // namespace acladapter
}  // namespace ock