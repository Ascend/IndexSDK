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


#ifndef HCPS_PIER_OCKDISTMASKWITHEXTRAGENOPRUN_H
#define HCPS_PIER_OCKDISTMASKWITHEXTRAGENOPRUN_H
#include <vector>
#include "ock/hcps/stream/OckHeteroStreamBase.h"
#include "ock/hcps/nop/dist_mask_with_extra_gen_op/OckDistMaskWithExtraGenMeta.h"
namespace ock {
namespace hcps {
namespace nop {
class OckDistMaskWithExtraGenOpRun {
public:
    static OckHcpsErrorCode AddMaskWithExtraOpsMultiBatches(
        std::shared_ptr<OckDistMaskWithExtraGenOpHmoGroups> hmoGroups, std::shared_ptr<OckHeteroStreamBase> streamBase);
    static void AddMaskWithExtraOpsSingleBatch(std::shared_ptr<OckDistMaskWithExtraGenOpHmoGroups> hmoGroups,
        std::shared_ptr<OckHeteroStreamBase> streamBase);
};
}
}
}
#endif // HCPS_PIER_OCKDISTMASKWITHEXTRAGENOPRUN_H
