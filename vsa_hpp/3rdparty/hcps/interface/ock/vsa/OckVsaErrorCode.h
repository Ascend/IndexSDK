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


#ifndef OCK_VECTOR_SEARCH_ALGO_ERROR_CODE_H
#define OCK_VECTOR_SEARCH_ALGO_ERROR_CODE_H
#include "ock/hmm/mgr/OckHmmErrorCode.h"
#include "ock/hcps/error/OckHcpsErrorCode.h"
namespace ock {
namespace vsa {
/*
@brief: 0代表成功，0~255供可执行程序返回码使用， 10000~20000 HMM/AclAdapter 内部使用，20000~30000 HCPS 内部使用,
30000~40000 VSA 内部使用, 100000~1000000 ACL模块使用。VSA模块使用从 21000~22000间的错误码
*/
using OckVsaErrorCode = int;
constexpr OckVsaErrorCode VSA_SUCCESS = 0;                        // 成功
constexpr OckVsaErrorCode VSA_ERROR_INVALID_OUTTER_LABEL = 31100; // 错误的label，通常是超出范围或不存在
constexpr OckVsaErrorCode VSA_ERROR_INVALID_INPUT_PARAM = 31101;  // 不正确输入参数
constexpr OckVsaErrorCode VSA_ERROR_TOO_MANY_FEATURE_NUMBER = 31102;   // 特征数太多
constexpr OckVsaErrorCode VSA_ERROR_DEVICE_NOT_EXISTS = 31103;         // 输入的 DEVICE 不存在
constexpr OckVsaErrorCode VSA_ERROR_CPUID_NOT_EXISTS = 31104;          // 输入的 CpuId 不存在
constexpr OckVsaErrorCode VSA_ERROR_CPUID_OUT = 31105;                 // 输入的 cpuid 超出[4, uint32 Max]
constexpr OckVsaErrorCode VSA_ERROR_MAX_FEATURE_ROW_COUNT_OUT = 31106; // maxFeatureRowCount 超出[262144 * 64, 10亿]
constexpr OckVsaErrorCode VSA_ERROR_TOKEN_NUM_OUT = 31107;             // 输入的 tokenNum 超出[1, 3.0e5]
constexpr OckVsaErrorCode VSA_ERROR_EXTKEYATTR_BYTE_SIZE_OUT = 31108; // 输入的 extKeyAttrByteSize 超出[0, 22]
constexpr OckVsaErrorCode VSA_ERROR_EXTKEYATTR_BLOCK_SIZE_OUT = 31109; // extKeyAttrBlockSize超出[262144, GroupRowCount]
constexpr OckVsaErrorCode VSA_ERROR_GROUP_ROW_COUNT_OUT = 31110;   // GroupBlockCount * SliceRowCount 必须小于4亿
constexpr OckVsaErrorCode VSA_ERROR_MAX_GROUP_COUNT_OUT = 31111;   // ceil(MaxFeatureRowCount / GroupRowCount) 小于100

constexpr OckVsaErrorCode VSA_ERROR_MAX_FEATURE_ROW_COUNT_DIVISIBLE = 31201; // 输入的 maxFeatureRowCount 不是256的倍数
constexpr OckVsaErrorCode VSA_ERROR_BLOCK_ROW_COUNT_DIVISIBLE = 31202; // 输入的 blockRowCount 不是 SliceRowCount 的倍数
constexpr OckVsaErrorCode VSA_ERROR_GROUP_BLOCK_COUNT_DIVISIBLE = 31203; // 输入的 groupBlockCount 不是 16 的倍数
constexpr OckVsaErrorCode VSA_ERROR_SLICE_ROW_COUNT_DIVISIBLE = 31204;   // 输入的 sliceRowCount 不是 64 的倍数
constexpr OckVsaErrorCode VSA_ERROR_GROUP_ROW_COUNT_DIVISIBLE = 31205; // 每个 Group 的数据行数, 不是 ExtKeyAttrBlockSize 的倍数
constexpr OckVsaErrorCode VSA_ERROR_GROUP_SLICE_COUNT_DIVISIBLE = 31206; // 每个Group中的Slice数(65536), 不是64的倍数
constexpr OckVsaErrorCode VSA_ERROR_EXCEED_NPU_INDEX_MAX_FEATURE_NUMBER = 31210; // 超出device侧maxRowCount
constexpr OckVsaErrorCode VSA_ERROR_EXCEED_HPP_INDEX_MAX_FEATURE_NUMBER = 31211; // 超出hpp的maxRowCount
constexpr OckVsaErrorCode VSA_ERROR_LABEL_NOT_EXIST = 31220; // 输入label不存在
constexpr OckVsaErrorCode VSA_ERROR_INPUT_PARAM_WRONG = 31221; // 输入参数错误
constexpr OckVsaErrorCode VSA_ERROR_EMPTY_BASE = 31222; // 底库为空
constexpr OckVsaErrorCode VSA_ERROR_TOKEN_NOT_EXIST = 31223; // 输入token不存在
constexpr OckVsaErrorCode VSA_ERROR_BUILD_DELETE_ATTR_OP_FAILED = 31224; // 构建删除时空属性算子失败
constexpr OckVsaErrorCode VSA_ERROR_BUILD_DELETE_OP_FAILED = 31225; // 构建删除算子失败

constexpr OckVsaErrorCode VSA_ERROR_NOT_SUPPORT_OPERATOR = 31300; // 不支持的操作
} // namespace vsa
} // namespace ock
#endif
