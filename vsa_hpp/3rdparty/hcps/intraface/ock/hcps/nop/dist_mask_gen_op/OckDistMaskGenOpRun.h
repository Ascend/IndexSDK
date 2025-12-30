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


#ifndef HCPS_OCKDISTMASKGENOPRUN_H
#define HCPS_OCKDISTMASKGENOPRUN_H
#include <vector>
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/nop/dist_mask_gen_op/OckDistMaskGenMeta.h"
namespace ock {
namespace hcps {
namespace nop {
class OckDistMaskGenOpRun {
public:
    static OckHcpsErrorCode AddMaskOpsMultiBatches(std::shared_ptr<OckDistMaskGenOpHmoGroups> hmoGroups,
        std::shared_ptr<OckHeteroStreamBase> streamBase);
    static void AddMaskOpsSingleBatch(std::shared_ptr<OckDistMaskGenOpHmoGroups> hmoGroups,
        std::shared_ptr<OckHeteroStreamBase> streamBase);
};
}
}
}
#endif // HCPS_OCKDISTMASKGENOPRUN_H
