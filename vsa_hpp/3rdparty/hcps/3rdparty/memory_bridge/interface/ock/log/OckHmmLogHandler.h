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

#ifndef OCK_MEMORY_BRIDGE_OCK_HMM_LOG_HANDLER_H
#define OCK_MEMORY_BRIDGE_OCK_HMM_LOG_HANDLER_H
#include <cstdint>
#include <memory>
namespace ock {
class OckHmmLogHandler {
public:
    virtual ~OckHmmLogHandler() noexcept = default;
    /*
    @brief 日志记录句柄
    @param level： level主要方便用户定制level与字符串DEBUG INFO WARN ERROR FATAL的映射关系
    */
    virtual void Write(int32_t level, const char *levelStr, const char *fileName, uint64_t lineNo, const char *msg) = 0;

    /*
    @brief 刷写数据，当异常发生时，会先调用Flush接口，然后抛异常
    */
    virtual void Flush(void) = 0;
};
constexpr int32_t OCK_LOG_LEVEL_DEBUG = 0;  // DEBUG级别日志
constexpr int32_t OCK_LOG_LEVEL_INFO = 1;   // INFO级别日志
constexpr int32_t OCK_LOG_LEVEL_WARN = 2;   // WARN级别日志
constexpr int32_t OCK_LOG_LEVEL_ERROR = 3;  // ERROR级别日志
constexpr int32_t OCK_LOG_LEVEL_FATAL = 4;  // FATAL 级别日志

/*
@brief 设置日志处理handler，缺省情况下， 日志通过std::cout输出
*/
void OckHmmSetLogHandler(std::shared_ptr<OckHmmLogHandler> handler);
/*
@brief 只有当日志级别大于等于startLevel的日志才会调用 handler的Write，初始缺省值OCK_LOG_LEVEL_INFO
*/
void OckHmmSetLogLevel(int32_t level);

}  // namespace ock
#endif