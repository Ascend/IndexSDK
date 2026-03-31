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
#include <cstdint>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "cagra_tiling.h"

namespace optiling{
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
	auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
	size_t* currentWorkspace = context->GetWorkspaceSizes(1);
	size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
	currentWorkspace[0] = systemWorkspacesSize;

	CagraTilingData tiling;
	//int32_t coreNum = ascendPlatform.GetCoreNumAiv();
	//context->SetBlockDim(coreNum)
	context->SetBlockDim(32);
	
	tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
	context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

	return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
	return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
	return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class Cagra : public OpDef {
public:
	explicit Cagra(const char* name) : OpDef(name)
	{
		this->Input("query")
			.ParamType(REQUIRED)
			.DataType({ge::DT_FLOAT})
			.Format({ge::FORMAT_ND})
			.UnknownShapeFormat({ge::FORMAT_ND});
		this->Input("knn_graph")
			.ParamType(REQUIRED)
			.DataType({ge::DT_UINT32})
			.Format({ge::FORMAT_ND})
			.UnknownShapeFormat({ge::FORMAT_ND});
		this->Input("visited_hashmap_ptr")
			.ParamType(REQUIRED)
			.DataType({ge::DT_UINT32})
			.Format({ge::FORMAT_ND})
			.UnknownShapeFormat({ge::FORMAT_ND});
		this->Input("ptr")
			.ParamType(REQUIRED)
			.DataType({ge::DT_FLOAT})
			.Format({ge::FORMAT_ND})
			.UnknownShapeFormat({ge:: FORMAT_ND});
		this->Output("result_distances_ptr")
			.ParamType(REQUIRED)
			.DataType({ge::DT_FLOAT})
			.Format({ge::FORMAT_ND})
			.UnknownShapeFormat({ge::FORMAT_ND});
		this->Output("result_indices_ptr")
			.ParamType(REQUIRED)
			.DataType({ge::DT_UINT32})
			.Format({ge::FORMAT_ND})
			.UnknownShapeFormat({ge::FORMAT_ND});

		this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

		this->AICore().SetTiling(optiling::TilingFunc);
		this->AICore().AddConfig("ascend950");
	}
};

OP_ADD(Cagra);
}