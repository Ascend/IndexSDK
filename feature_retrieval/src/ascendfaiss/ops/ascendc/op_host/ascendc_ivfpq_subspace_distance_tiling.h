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

#ifndef ASCENDC_OP_HOST_ASCENDC_IVFPQ_SUBSPACE_DISTANCE_TILING_H
#define ASCENDC_OP_HOST_ASCENDC_IVFPQ_SUBSPACE_DISTANCE_TILING_H

#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"
#include "tiling/matrix/matmul_tilingdata.h"
#include "tiling/matrix/matmul_tiling.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AscendcIvfpqSubspaceDistanceTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batch);
    TILING_DATA_FIELD_DEF(uint32_t, subSpaceNum);
    TILING_DATA_FIELD_DEF(uint32_t, dim);
    TILING_DATA_FIELD_DEF(uint32_t, kSub);
    TILING_DATA_FIELD_DEF(uint32_t, dSub);
    TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);

    TILING_DATA_FIELD_DEF(uint32_t, minValue);

    TILING_DATA_FIELD_DEF(uint32_t, codeBookMinValue);
    TILING_DATA_FIELD_DEF(uint32_t, nBlockTile);
    TILING_DATA_FIELD_DEF(uint32_t, nBlockNum);
    TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
    TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
    TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTiling);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AscendcIvfpqSubspaceDistance, AscendcIvfpqSubspaceDistanceTilingData)

class IvfpqSubspaceTiling {
public:
    ge::graphStatus ProcessTiling(gert::TilingContext* context, AscendcIvfpqSubspaceDistanceTilingData &tilingData);

protected:

private:
    ge::graphStatus GetNpuInfo();
    ge::graphStatus ProcessInput();
    ge::graphStatus Split();
    ge::graphStatus FillTilingData();
    ge::graphStatus SetExtraConfig();
    ge::graphStatus CubeTiling();


private:
    gert::TilingContext *context_ = nullptr;
    AscendcIvfpqSubspaceDistanceTilingData *tilingData_ = nullptr;

    uint32_t libapiSize_ = 0;
    uint32_t workSpaceSize_ = 0;
    uint64_t ubSize_ = 0;
    uint64_t tilingKey_ = 0;

    uint32_t batch_ = 0;
    uint32_t dim_ = 0;
    uint32_t subSpaceNum_ = 0;
    uint32_t kSub_ = 0;
    uint32_t dSub_ = 0;

    uint32_t minValue_ = 0;
    uint32_t maxValue_ = 0;

    uint32_t nBlockTile_ = 0;
    uint32_t nBlockNum_ = 0;
    uint32_t totalTaskNum_ = 0;

    uint32_t aicNum_ = 0;
    uint32_t usedAicNum_ = 0;

    uint32_t codeBookMinValue_ = 0;
    uint32_t codeBookMaxValue_ = 0;
};
}
#endif // ASCENDC_OP_HOST_ASCENDC_IVFPQ_SUBSPACE_DISTANCE_TILING_H