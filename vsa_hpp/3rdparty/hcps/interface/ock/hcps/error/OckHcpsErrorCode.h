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


#ifndef OCK_HCPS_ERROR_CODE_H
#define OCK_HCPS_ERROR_CODE_H
#include "ock/hmm/mgr/OckHmmErrorCode.h"
namespace ock {
namespace hcps {
/*
@brief: 0代表成功，0~255供可执行程序返回码使用， 10000~20000 HMM/AclAdapter 内部使用，20000~30000 HCPS 内部使用,
30000~40000 VSA 内部使用, 100000~1000000 ACL模块使用。VSA模块使用从 21000~22000间的错误码
*/
using OckHcpsErrorCode = int;
constexpr OckHcpsErrorCode HCPS_ERROR_NOT_SUPPORT_OPERATOR = 21200;  // 不支持的操作
constexpr OckHcpsErrorCode HCPS_ERROR_ACL_CREATE_DATA_BUFFER_FAILED = 20001;  // aclCreateDataBuffer返回值为nullptr
constexpr OckHcpsErrorCode HCPS_ERROR_ASCEND_OP_HANDLE_NULL = 20002;  // 昇腾算子的handle为空指针
constexpr OckHcpsErrorCode HCPS_ERROR_STREAM_NULL = 20003;  // stream为空指针
constexpr OckHcpsErrorCode HCPS_ERROR_GET_BUFFER_FAILED = 20004;  // 执行getBuffer操作失败
constexpr OckHcpsErrorCode HCPS_ERROR_NPU_BASE_EMPTY = 20005;  // npu侧底库为空
constexpr OckHcpsErrorCode HCPS_ERROR_NPU_GROUP_NOT_EXIST = 20006;  // npu侧group不存在
constexpr OckHcpsErrorCode HCPS_ERROR_BLOCK_POS_EXCEED_SCOPE = 20007;  // npu侧block不存在于指定的group内
constexpr OckHcpsErrorCode HCPS_ERROR_CUSTOM_ATTR_EMPTY = 20008;  // 传入的custom attr为空
constexpr OckHcpsErrorCode HCPS_ERROR_OFFSET_EXCEED_SCOPE = 20009;  // 传入的offset超出底库大小
constexpr OckHcpsErrorCode HCPS_ERROR_INVALID_OP_INPUT_PARAM = 20010;  // 算子计算入参错误
constexpr OckHcpsErrorCode HCPS_ERROR_INVALID_OP_HMO_BYTE_SIZE = 20011;  // 算子使用的hmo大小不符合要求
}  // namespace hcps
}  // namespace ock
#endif
