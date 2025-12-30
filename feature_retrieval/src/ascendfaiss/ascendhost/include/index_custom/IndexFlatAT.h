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

#ifndef ASCENDHOST_INDEXFLAT_AT_INCLUDED
#define ASCENDHOST_INDEXFLAT_AT_INCLUDED

#include "ascenddaemon/impl/Index.h"
#include "ascenddaemon/utils/DeviceVector.h"

namespace ascend {
class IndexFlatAT : public Index {
public:
    const int QUERY_ALIGN = 32;
    const int BURST_LEN = 64;
    const int TRANSFER_SIZE = 256;
    const int CODE_ALIGN = 512;
    const int SEARCH_PAGE = 32768;
    const int QUERY_BATCH = 1024;
    
    IndexFlatAT(int dim, int baseSize, int64_t resourceSize);

    APP_ERROR resetTopkCompOp();

protected:
    std::unique_ptr<::ascend::AscendOperator> topkComputeOps;
    int queryBatch;
    int searchPage;
    int baseSize;
    int bursts;
};
}  // namespace ascend

#endif  // ASCENDHOST_INDEXFLAT_AT_AICPU_INCLUDED