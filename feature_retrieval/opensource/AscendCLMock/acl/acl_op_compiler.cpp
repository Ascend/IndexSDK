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

#include "acl_op_compiler.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup AscendCL
 * @brief compile op
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN]       pointer to array of output tensor descriptions
 * @param attr [IN]           pointer to instance of aclopAttr.
 * may pass nullptr if the op has no attribute
 * @param engineType [IN]       engine type
 * @param compileFlag [IN]      compile flag
 * @param opPath [IN]           path of op
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError aclopCompile(const char *opType, int numInputs, const aclTensorDesc * const inputDesc[], int numOutputs,
    const aclTensorDesc * const outputDesc[], const aclopAttr *attr, aclopEngineType engineType,
    aclopCompileType compileFlag, const char *opPath)
{
    return 0;
}

/**
 * @ingroup AscendCL
 * @brief compile and execute op
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param inputs [IN]           pointer to array of input buffers
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN]       pointer to array of output tensor descriptions
 * @param outputs [IN]          pointer to array of outputs buffers
 * @param attr [IN]             pointer to instance of aclopAttr.
 * may pass nullptr if the op has no attribute
 * @param engineType [IN]       engine type
 * @param compileFlag [IN]      compile flag
 * @param opPath [IN]           path of op
 * @param stream [IN]           stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError aclopCompileAndExecute(const char *opType, int numInputs, const aclTensorDesc * const inputDesc[],
    const aclDataBuffer * const inputs[], int numOutputs, const aclTensorDesc * const outputDesc[],
    aclDataBuffer * const outputs[], const aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag,
    const char *opPath, aclrtStream stream)
{
    return 0;
}

/**
 * @ingroup AscendCL
 * @brief compile and execute op
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param inputs [IN]           pointer to array of input buffers
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN|OUT]   pointer to array of output tensor descriptions
 * @param outputs [IN]          pointer to array of outputs buffers
 * @param attr [IN]             pointer to instance of aclopAttr.
 * may pass nullptr if the op has no attribute
 * @param engineType [IN]       engine type
 * @param compileFlag [IN]      compile flag
 * @param opPath [IN]           path of op
 * @param stream [IN]           stream handle
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError aclopCompileAndExecuteV2(const char *opType, int numInputs, aclTensorDesc *inputDesc[],
    aclDataBuffer *inputs[], int numOutputs, aclTensorDesc *outputDesc[], aclDataBuffer *outputs[], aclopAttr *attr,
    aclopEngineType engineType, aclopCompileType compileFlag, const char *opPath, aclrtStream stream)
{
    return 0;
}

/**
 * @ingroup AscendCL
 * @brief set compile option
 *
 * @param aclCompileOpt [IN]      compile option
 * @param value [IN]              pointer for the option value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError aclSetCompileopt(aclCompileOpt opt, const char *value)
{
    return 0;
}

/**
 * @ingroup AscendCL
 * @brief get compile option value size
 *
 * @param aclCompileOpt [IN]      compile option
 *
 * @retval size of compile option value
 */
size_t aclGetCompileoptSize(aclCompileOpt opt)
{
    return 0;
}

/**
 * @ingroup AscendCL
 * @brief get compile option
 *
 * @param aclCompileOpt [IN]      compile option
 * @param value [OUT]             pointer for the option value
 * @param length [IN]             length of value
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError aclGetCompileopt(aclCompileOpt opt, char *value, size_t length)
{
    return 0;
}

/**
 * @ingroup AscendCL
 * @brief set compile flag
 *
 * @param flag [IN]    compile flag, ACL_OP_COMPILE_DEFAULT means compile with default mode
 * ACL_OP_COMPILE_FUZZ means compile with fuzz mode
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError aclopSetCompileFlag(aclOpCompileFlag flag)
{
    return 0;
}

/**
 * @ingroup AscendCL
 * @brief generate graph and dump
 *
 * @param opType [IN]           op type
 * @param numInputs [IN]        number of inputs
 * @param inputDesc [IN]        pointer to array of input tensor descriptions
 * @param inputs [IN]           pointer to array of input buffers
 * @param numOutputs [IN]       number of outputs
 * @param outputDesc [IN]       pointer to array of output tensor descriptions
 * @param outputs [IN]          pointer to array of outputs buffers
 * @param attr [IN]             pointer to instance of aclopAttr.
 * may pass nullptr if the op has no attribute
 * @param engineType [IN]       engine type
 * @param graphDumpPath [IN]    dump path, if the suffix is ".txt", it means file path, else it means directory path
 * @param graphDumpOpt [IN]     dump option, nullptr is supported
 *
 * @retval ACL_SUCCESS The function is successfully executed.
 * @retval OtherValues Failure
 */
aclError aclGenGraphAndDumpForOp(const char *opType, int numInputs, const aclTensorDesc * const inputDesc[],
    const aclDataBuffer * const inputs[], int numOutputs, const aclTensorDesc * const outputDesc[],
    aclDataBuffer * const outputs[], const aclopAttr *attr, aclopEngineType engineType, const char *graphDumpPath,
    const aclGraphDumpOption *graphDumpOpt)
{
    return 0;
}

/**
 * @ingroup AscendCL
 * @brief Create the graph dump option
 *
 * @retval null for failed
 * @retval OtherValues success
 *
 * @see aclDestroyGraphDumpOpt
 */
aclGraphDumpOption *aclCreateGraphDumpOpt()
{
    return 0;
}

/**
 * @ingroup AscendCL
 * @brief Destroy graph dump option
 *
 * @param graphDumpOpt [IN]  pointer to the graph dump option
 *
 * @retval ACL_SUCCESS  The function is successfully executed.
 * @retval OtherValues Failure
 *
 * @see aclCreateGraphDumpOpt
 */
aclError aclDestroyGraphDumpOpt(const aclGraphDumpOption *graphDumpOpt)
{
    return 0;
}

aclError aclSetTensorShapeRange(aclTensorDesc* desc, size_t dimsCount, int64_t dimsRange[][ACL_TENSOR_SHAPE_RANGE_NUM])
{
    return 0;
}

#ifdef __cplusplus
}
#endif
