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

#include "ock/conf/OckSysConf.h"

namespace ock {
namespace conf {

const OckHmmConf &OckSysConf::HmmConf(void)
{
    static OckHmmConf conf;
    return conf;
}

const OckAclAdapterConf &OckSysConf::AclAdapterConf(void)
{
    static OckAclAdapterConf conf;
    return conf;
}

const OckHmmDeviceInfoConf &OckSysConf::DeviceInfoConf(void)
{
    static OckHmmDeviceInfoConf conf;
    return conf;
}

const OckTopoDetectConf &OckSysConf::ToolConf(void)
{
    static OckTopoDetectConf conf;
    return conf;
}

const OckHmmConf &g_OckSysConfOckHmmConf = OckSysConf::HmmConf();
const OckAclAdapterConf &g_OckSysConfOckAclAdapterConf = OckSysConf::AclAdapterConf();
const OckHmmDeviceInfoConf &g_OckSysConfOckHmmDeviceInfoConf = OckSysConf::DeviceInfoConf();
const OckTopoDetectConf &g_OckSysConfOckToolConf = OckSysConf::ToolConf();
}  // namespace conf
}  // namespace ock