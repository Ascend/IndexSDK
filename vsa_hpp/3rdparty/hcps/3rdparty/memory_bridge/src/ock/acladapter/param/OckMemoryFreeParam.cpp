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

#include "ock/acladapter/param/OckMemoryFreeParam.h"
namespace ock {
namespace acladapter {

OckMemoryFreeParam::OckMemoryFreeParam(void *address, hmm::OckHmmHeteroMemoryLocation memLocation)
    : addr(address), location(memLocation)
{}
void *OckMemoryFreeParam::Addr(void)
{
    return addr;
}
hmm::OckHmmHeteroMemoryLocation OckMemoryFreeParam::Location(void) const
{
    return location;
}
const void *OckMemoryFreeParam::Addr(void) const
{
    return addr;
}

std::ostream &operator<<(std::ostream &os, const OckMemoryFreeParam &param)
{
    return os << "{'location':" << param.Location() << "}";
}
}  // namespace acladapter
}  // namespace ock