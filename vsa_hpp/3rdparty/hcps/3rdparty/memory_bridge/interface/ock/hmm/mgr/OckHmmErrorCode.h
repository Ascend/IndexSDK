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


#ifndef OCK_MEMORY_BRIDGE_ERROR_CODE_H
#define OCK_MEMORY_BRIDGE_ERROR_CODE_H
namespace ock {
namespace hmm {
/*
@brief: 0代表成功，0~255供可执行程序返回码使用， 10000~20000 HMM/AclAdapter 内部使用，20000~30000 HCPS 内部使用,
100000~1000000 ACL模块使用
*/
using OckHmmErrorCode = int;
constexpr OckHmmErrorCode HMM_SUCCESS = 0;                             // 成功
constexpr OckHmmErrorCode HMM_ERROR_EXEC_INVALID_INPUT_PARAM = 1;      // 可执行程序入参错误
constexpr OckHmmErrorCode HMM_ERROR_EXEC_FAILED = 2;                   // 可执行程序运行错误
constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_EMPTY = 14000;         // 输入参数为空或输入参数列表为空
constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_OUT_OF_RANGE = 14001;  // 输入的参数超出规定的取值范围
constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_DEVICE_NOT_EXISTS = 14002;        // 输入的DEVICE不存在
constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_CPUID_NOT_EXISTS = 14004;         // cpuId不存在
constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_SRC_OFFSET_EXCEED_SCOPE = 14010;  // 源数据OFFSET超出范围
constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_DST_OFFSET_EXCEED_SCOPE = 14011;  // 目标数据OFFSET超出范围
constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_SRC_LENGTH_EXCEED_SCOPE = 14012;  // 源数据长度超出范围
constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_DST_LENGTH_EXCEED_SCOPE = 14013;  // 目标数据长度超出范围
constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_DEVICEID_NOT_EQUAL = 14014;       // 输入DeviceId不匹配
constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_NOT_SUPPORT_SUCH_OP = 14050;      // 不支持这样的操作
constexpr OckHmmErrorCode HMM_ERROR_INPUT_PARAM_ZERO_MALLOC = 14051;              // 分配0个字节
constexpr OckHmmErrorCode HMM_ERROR_WAIT_TIME_OUT = 14060;                        // 等待超时
constexpr OckHmmErrorCode HMM_ERROR_DEVICE_BUFFER_SPACE_NOT_ENOUGH = 14070;       // Device的BUFFER(Swap)空间不足
constexpr OckHmmErrorCode HMM_ERROR_DEVICE_DATA_SPACE_NOT_ENOUGH = 14071;         // Device的底库空间不足
constexpr OckHmmErrorCode HMM_ERROR_HOST_BUFFER_SPACE_NOT_ENOUGH = 14072;         // Host的BUFFER(Swap)空间不足
constexpr OckHmmErrorCode HMM_ERROR_HOST_DATA_SPACE_NOT_ENOUGH = 14073;           // Host的底库空间不足
constexpr OckHmmErrorCode HMM_ERROR_SPACE_NOT_ENOUGH = 14074;                     // 空间不足
constexpr OckHmmErrorCode HMM_ERROR_SWAP_SPACE_NOT_ENOUGH = 14075;                // swap空间不足
constexpr OckHmmErrorCode HMM_ERROR_STACK_MANAGE_SPACE_NOT_ENOUGH = 14076;        // 空间不足
constexpr OckHmmErrorCode HMM_ERROR_TASK_ALREADY_RUNNING = 14100;                 // 任务已经在运行
constexpr OckHmmErrorCode HMM_ERROR_HMO_NO_AVAIBLE_ID = 14200;                    // 无可用的HMOID
constexpr OckHmmErrorCode HMM_ERROR_HMO_OBJECT_NUM_EXCEED = 14201;                // HMO对象超过上限
constexpr OckHmmErrorCode HMM_ERROR_HMO_OBJECT_NOT_EXISTS = 14202;                // HMO对象不存在
constexpr OckHmmErrorCode HMM_ERROR_HMO_OBJECT_INVALID = 14203;                   // HMO对象不合法
constexpr OckHmmErrorCode HMM_ERROR_HMO_BUFFER_RELEASED = 14204;                  // buffer数据已经被释放
constexpr OckHmmErrorCode HMM_ERROR_HMO_BUFFER_NOT_ALLOCED = 14205;               // buffer数据未分配，不存在
constexpr OckHmmErrorCode HMM_ERROR_UNKOWN_INNER_ERROR = 19999;                   // 未知错误

}  // namespace hmm
}  // namespace ock
#endif  // OCK_MEMORY_BRIDGE_ERROR_CODE_H
