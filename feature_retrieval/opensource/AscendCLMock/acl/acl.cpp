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

#include <cstdlib>
#include <string>
#include <cstring>
#include <cstdarg>
#include <unistd.h>
#include "securec.h"
#include "simu/AscendSimu.h"
#include "acl.h"

struct streamInfo {
    uint32_t deviceId;
    bool isCreate;
    bool isDefault;
};

std::map<aclrtStream, streamInfo> g_streamInfo;

std::string formatArgs(const char *fmt, va_list args)
{
    char buffer[256];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    return std::string(buffer);
}

std::string formatArgs(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    auto string = formatArgs(fmt, args);
    va_end(args);

    return string;
}

// 以下定义是MOCK acl接口，需要定义成C接口形式

#ifdef __cplusplus
extern "C" {
#endif

// aclInit 会调用下面函数 配置模拟执行环境 ,这里声明为弱符号 如果外部依赖想重新定义 可以重载
void __attribute__((weak)) AscendEnvInit()
{
     // step1: 首先配置模拟DEVICE
    std::vector<AscendSimuDevice *> deviceList;
    for (int i = 0; i < MAX_DEVICE; i++) {
        deviceList.push_back(new AscendSimuDevice(i, 8)); // 310P has 8 ai core
    }
    // step2: 配置模拟执行运行环境 310P HOST duo卡
    ENV().construct("Ascend310P", ACL_HOST, deviceList);

    // step3: 配置日志目录调用app_acl_log将记录到该位置
    LOGGER().Reset();
    LOGGER().SetLogLevel(ACL_INFO); // 日志等级设为INFO

    char logName[64] = {0};
    sprintf(logName, "./log_pid_%d.txt", getpid());
    LOGGER().SetLogFile(logName); // 日志目录为log.txt
}

// aclFinalize 会调用下面函数
void __attribute__((weak)) AscendEnvFinalize()
{
    // step4: 日志复位不影响其他用例
    LOGGER().Reset();

    // step5: 模拟环境复位
    ENV().destruct();
}

void ShowSimuSystem()
{
    printf("------simu system info------\n");
    printf("------SoCName:%s------\n", aclrtGetSocName());

    aclrtRunMode runMode;
    aclrtGetRunMode(&runMode);
    printf("------runMode:%d------\n", runMode);

    uint32_t deviceCnt;
    aclrtGetDeviceCount(&deviceCnt);
    printf("------deviceCnt:%d------\n", deviceCnt);

    printf("------LogLevel:%s------\n", LOGGER().GetLogLevel());
    printf("------LogFilePath:%s------\n", LOGGER().GetLogFile());
    AscendSimuExecFlow::ShowOp();
}

aclError aclInit(const char *UNUSED(configPath))
{
    AscendEnvInit();
    simuOpInstall();
#ifdef BUILD_VISION
    simuOpTikInstall();
#endif
    ShowSimuSystem();
    aclrtSetDevice(0);
    return ACL_SUCCESS;
}

aclError aclFinalize()
{
    AscendEnvFinalize();
    simuOpUninstall();
#ifdef BUILD_VISION
    simuOpTikUninstall();
#endif
    return ACL_SUCCESS;
}

void setMaxSize(size_t size)
{
    g_maxSize = size;
}

aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy UNUSED(policy))
{
    // size = ((size + 31) / 32) + 32; // 按照手册会对用户的size按照32字节向上对齐，并额外添加32字节
    int ret = (*devPtr = malloc(size)) == nullptr ? ACL_ERROR_BAD_ALLOC : ACL_SUCCESS;
    return ret;
}

aclError aclrtMallocHost(void **hostPtr, size_t size)
{
    // size = ((size + 31) / 32) + 32; // 按照手册会对用户的size按照32字节向上对齐，并额外添加32字节
    return (*hostPtr = malloc(size)) == nullptr ? ACL_ERROR_BAD_ALLOC : ACL_SUCCESS;
}

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind UNUSED(kind))
{
    if (count > destMax) {
        return ACL_ERROR_STORAGE_OVER_LIMIT;
    }
    std::memcpy(dst, src, count);
    return ACL_SUCCESS;
}

aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count)
{
    if (count > maxCount) {
        return ACL_ERROR_STORAGE_OVER_LIMIT;
    }
    std::memset(devPtr, value, count);
    return ACL_SUCCESS;
}

aclError aclrtFree(void *devPtr)
{
    free(devPtr);
    return ACL_SUCCESS;
}

aclError aclrtFreeHost(void *hostPtr)
{
    free(hostPtr);
    return ACL_SUCCESS;
}

aclDataBuffer *aclCreateDataBuffer(void *data, size_t size)
{
    auto *buffer = new aclDataBuffer;
    buffer->data = data;
    buffer->size = size;
    return buffer;
}

aclError aclrtMemcpyAsync(void *dst,
                          size_t destMax,
                          const void *src,
                          size_t count,
                          aclrtMemcpyKind kind,
                          aclrtStream stream)
{
    if (stream == nullptr) {
        auto deviceId = ENV().getActiveDeviceId();
        stream = DEVICE(deviceId)->GetActiveContext()->m_defaultStream;
    }

    aclopHandle *opHandle = new aclopHandle;
    char *opType = "aclrtMemcpyAsync";
    auto len = strlen(opType) + 1; // 结束符多1个字节
    char *newOpName = new char[len];
    auto cpyRet = strcpy_s(newOpName, len, opType);
    if (cpyRet != EOK) {
        printf("strcpy_s error[%d]\r\n", cpyRet);
        return ACL_ERROR_FAILURE;
    }
    opHandle->opName = newOpName;
    opHandle->opAttr = nullptr;
    opHandle->numInputs = 3;
    opHandle->numOutPuts = 1;

    opHandle->inputData = new aclDataBuffer[3];
    opHandle->inputData[0].data = dst;
    opHandle->inputData[0].size = destMax;

    opHandle->inputData[1].data = const_cast<void *>(src);
    opHandle->inputData[1].size = count;

    opHandle->inputData[2].data = &kind;
    opHandle->inputData[2].size = sizeof(aclrtMemcpyKind);

    auto ret = DEVICE(stream->deviceId)->StreamExecFlowPushOpHandle(stream, opHandle);
    return ret ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer)
{
    delete dataBuffer;
    return ACL_SUCCESS;
}

aclError aclrtSetDevice(int32_t deviceId)
{
    ENV().SetDevice(deviceId);
    return ACL_SUCCESS;
}

aclError aclrtGetDevice(int32_t *deviceId)
{
    auto activeDeviceId = AscendSimuEnv::getActiveDeviceId();
    if (activeDeviceId == -1) {
        return ACL_ERROR_UNINITIALIZE;
    }
    *deviceId = activeDeviceId;
    return ACL_SUCCESS;
}

aclError aclrtResetDevice(int32_t deviceId)
{
    ENV().ReSetDevice(deviceId);
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceCount(uint32_t *count)
{
    *count = ENV().getDeviceCount();
    return ACL_SUCCESS;
}

aclError aclrtCreateStream(aclrtStream *stream)
{
    auto deviceId = ENV().getActiveDeviceId();
    DEVICE(deviceId)->CreateStream(DEVICE(deviceId)->GetActiveContext(), stream);
    g_streamInfo.insert({ *stream, streamInfo{ (uint32_t)deviceId, false, false } });
    return ACL_SUCCESS;
}

aclopAttr *aclopCreateAttr()
{
    return new aclopAttr;
}

void aclopDestroyAttr(const aclopAttr *attr)
{
    delete attr;
}

aclError aclrtSynchronizeStream(aclrtStream stream)
{
    if (stream != nullptr && g_streamInfo.find(stream) == g_streamInfo.end()) {
        return 1;
    }
    if (stream == nullptr) {
        auto deviceId = ENV().getActiveDeviceId();
        stream = DEVICE(deviceId)->GetActiveContext()->m_defaultStream;
    }

    auto deviceId = stream->deviceId;
    DEVICE(deviceId)->WaitStreamExecFlowIdle(stream);
    return ACL_SUCCESS;
}

aclError aclrtDestroyStream(aclrtStream stream)
{
    if (stream == nullptr) {
        // 不能删除默认stream
        return ACL_ERROR_INTERNAL_ERROR;
    }

    auto deviceId = stream->deviceId;
    DEVICE(deviceId)->DestoryStream(stream);
    g_streamInfo.erase(stream);
    delete stream;
    stream = nullptr;
    return ACL_SUCCESS;
}

void aclAppLog(aclLogLevel logLevel, const char *func, const char *file, uint32_t line,
               const char *fmt, ...)
{
    static const char *logLevelStr[] = {
        "ACL_DEBUG",
        "ACL_INFO",
        "ACL_WARNING",
        "ACL_ERROR"
    };
    std::string content;

    content = formatArgs("logLevel[%s] func[%s] file[%s] line[%d] msg:", logLevelStr[logLevel], func, file, line);

    va_list ap;
    va_start(ap, fmt);
    content += formatArgs(fmt, ap);
    va_end(ap);

    LOGGER().write(logLevel, content.data(), content.size());
}

aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId)
{
    // 如果在关联的device上没有调用aclrtSetDevice绑定，则在createcontext需要绑定
    if (DEVICE(deviceId)->GetRefCnt() == 0) {
        aclrtSetDevice(deviceId);
    }

    DEVICE(deviceId)->CreateContext(context);
    ENV().setActiveDeviceId(deviceId);
    return ACL_SUCCESS;
}

// 需要用户删除关联的stream
aclError aclrtDestroyContext(aclrtContext context)
{
    DEVICE(context->deviceId)->DestoryContext(context);
    if (context == DEVICE(context->deviceId)->GetActiveContext()) {
        DEVICE(context->deviceId)->SetActiveContext(nullptr);
    }

    return ACL_SUCCESS;
}

aclError aclrtGetCurrentContext(aclrtContext *context)
{
    auto deviceId = ENV().getActiveDeviceId();
    *context = DEVICE(deviceId)->GetActiveContext();
    return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext context)
{
    DEVICE(context->deviceId)->SetActiveContext(context);
    ENV().setActiveDeviceId(context->deviceId);
    return ACL_SUCCESS;
}

const char *aclrtGetSocName()
{
    return ENV().getSocName();
}

aclError aclopSetModelDir(const char *UNUSED(modelDir))
{
    return ACL_SUCCESS;
}

aclTensorDesc *aclCreateTensorDesc(aclDataType dataType,
                                   int numDims,
                                   const int64_t *dims,
                                   aclFormat format)
{
    auto tensorDesc = new aclTensorDesc;
    tensorDesc->dataType = dataType;
    tensorDesc->numDims = numDims;
    tensorDesc->dims = dims;
    tensorDesc->format = format;
    return tensorDesc;
}

void aclDestroyTensorDesc(const aclTensorDesc *desc)
{
    delete desc;
}

// 用户保证index不会超过限制
aclError aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize)
{
    *dimSize = desc->dims[index];
    return ACL_SUCCESS;
}

size_t aclGetTensorDescNumDims(const aclTensorDesc *desc)
{
    return desc->numDims;
}

static uint8_t getBytesByTensorType(aclDataType type)
{
    static std::map<aclDataType, uint8_t> typeMapBytes = {
        {ACL_INT8, sizeof(int8_t)},
        {ACL_INT16, sizeof(int16_t)},
        {ACL_INT32, sizeof(int32_t)},
        {ACL_INT64, sizeof(int64_t)},
        {ACL_UINT8, sizeof(uint8_t)},
        {ACL_UINT16, sizeof(uint16_t)},
        {ACL_UINT32, sizeof(uint32_t)},
        {ACL_UINT64, sizeof(uint64_t)},
        {ACL_FLOAT16, sizeof(int16_t)},
        {ACL_FLOAT, sizeof(float)},
        {ACL_DOUBLE, sizeof(double)},
        {ACL_BF16, sizeof(unsigned short)},
        {ACL_BOOL, sizeof(bool)},
    };

    return typeMapBytes[type];
}

size_t aclGetTensorDescElementCount(const aclTensorDesc *desc)
{
    size_t elementCount = 1;
    for (auto dimIndex = 0; dimIndex < desc->numDims; dimIndex++) {
        elementCount *= desc->dims[dimIndex];
    }
    return elementCount;
}

size_t aclGetTensorDescSize(const aclTensorDesc *desc)
{
    auto elmentCount = aclGetTensorDescElementCount(desc);
    auto elementBytes = getBytesByTensorType(desc->dataType);
    return elementBytes * elmentCount;
}

aclError aclopCreateHandle(const char *opType,
                           int numInputs,
                           const aclTensorDesc *const inputDesc[],
                           int numOutputs,
                           const aclTensorDesc *const outputDesc[],
                           const aclopAttr *opAttr,
                           aclopHandle **handle)
{
    *handle = new aclopHandle;
    (*handle)->opName = opType;
    (*handle)->numInputs = numInputs;
    (*handle)->numOutPuts = numOutputs;
    (*handle)->inputDesc = new aclTensorDesc[numInputs];
    (*handle)->outputDesc = new aclTensorDesc[numOutputs];
    (*handle)->opAttr = new aclopAttr;

    for (auto i = 0; i < numInputs; i++) {
        (*handle)->inputDesc[i] = *inputDesc[i];
    }

    for (auto i = 0; i < numOutputs; i++) {
        (*handle)->outputDesc[i] = *outputDesc[i];
    }

    if (opAttr != nullptr) {
        *((*handle)->opAttr) = *opAttr;
    }

    return ACL_SUCCESS;
}

aclError aclopExecWithHandle(aclopHandle *handle,
                             int numInputs,
                             const aclDataBuffer *const inputs[],
                             int numOutputs,
                             aclDataBuffer *const outputs[],
                             aclrtStream stream)
{
    if (!IS_OP_REG(handle->opName)) {
        aclAppLog(ACL_ERROR, __FUNCTION__, __FILE__, __LINE__,
                  "op:%s not register pls register first", handle->opName);
        return ACL_ERROR_UNINITIALIZE;
    }

    // stream为null 则表明使用defaultStream
    if (stream == nullptr) {
        auto deviceId = ENV().getActiveDeviceId();
        stream = DEVICE(deviceId)->GetActiveContext()->m_defaultStream;
    }

    aclopHandle *execHandle = new aclopHandle;
    *execHandle = *handle;

    execHandle->numInputs = numInputs;
    execHandle->numOutPuts = numOutputs;
    execHandle->inputData = new aclDataBuffer[numInputs];
    execHandle->outputData = new aclDataBuffer[numOutputs];
    auto len = strlen(handle->opName) + 1; // 结束符多1个字节
    char *newOpName = new char[len];
    execHandle->opName = newOpName;
    auto cpyRet = strcpy_s(newOpName, len, handle->opName);
    if (cpyRet != EOK) {
        printf("strcpy_s error[%d]\r\n", cpyRet);
        return ACL_ERROR_FAILURE;
    }

    for (auto i = 0; i < numInputs; i++) {
        execHandle->inputData[i].data = inputs[i]->data;
        execHandle->inputData[i].size = inputs[i]->size;
    }

    for (auto i = 0; i < numOutputs; i++) {
        execHandle->outputData[i].data = outputs[i]->data;
        execHandle->outputData[i].size = outputs[i]->size;
    }

    // put stream exec flow
    auto ret = DEVICE(stream->deviceId)->StreamExecFlowPushOpHandle(stream, execHandle);
    return ret ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

void aclopDestroyHandle(aclopHandle *handle)
{
    if (handle == nullptr) {
        return;
    }

    delete []handle->inputDesc;
    delete []handle->outputDesc;
    // delete []handle->inputData;
    // delete []handle->outputData;
    if (handle->opAttr != nullptr) {
        delete handle->opAttr;
    }
    delete handle;
}

aclError aclrtGetRunMode(aclrtRunMode *runMode)
{
    *runMode = ENV().getRunMode();
    return ACL_SUCCESS;
}

aclError aclrtLaunchCallback(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType, void *stream)
{
    tagAclrtStream *aclStream = (tagAclrtStream *)stream;
    if (aclStream == nullptr || g_streamInfo.find(aclStream) == g_streamInfo.end()) {
        return 1;
    }
    if (g_streamInfo[(aclrtStream)stream].isCreate == false) {
        return 1;
    }
    int deviceId = aclStream->deviceId;
    auto ret = DEVICE(deviceId)->StreamExecFlowPushOpHandle(aclStream, fn, userData);
    return !ret ? 1 : 0;
}

aclError aclrtProcessReport(int32_t timeout)
{
    return 0;
}

aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag)
{
    auto deviceId = ENV().getActiveDeviceId();
    DEVICE(deviceId)->CreateStream(DEVICE(deviceId)->GetActiveContext(), stream);
    g_streamInfo.insert({ *stream, streamInfo{ deviceId, false, false } });
    return ACL_SUCCESS;
}
aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    if (stream == nullptr) {
        return 0;
    }
    g_streamInfo[stream].isCreate = true;
    return 0;
}

aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    if (stream == nullptr) {
        return 0;
    }
    g_streamInfo[stream].isCreate = false;
    return 0;
}

aclError aclopExecuteV2(const char *opType, int numInputs, aclTensorDesc *inputDesc[], aclDataBuffer *inputs[],
    int numOutputs, aclTensorDesc *outputDesc[], aclDataBuffer *outputs[], aclopAttr *attr, void *stream)
{
    aclopHandle *opHandle = new aclopHandle;
    size_t strSize = strlen(opType);
    char *opName = new char[strSize + 1];
    memcpy(opName, opType, strSize);
    opName[strSize] = '\0';
    opHandle->opName = opName;
    opHandle->opAttr = new aclopAttr();
    opHandle->opAttr->attrs = attr->attrs;
    opHandle->numInputs = numInputs;
    opHandle->numOutPuts = numOutputs;

    opHandle->inputData = new aclDataBuffer[numInputs];
    opHandle->inputDesc = new aclTensorDesc[numInputs];
    opHandle->outputData = new aclDataBuffer[numOutputs];
    opHandle->outputDesc = new aclTensorDesc[numOutputs];

    for (auto i = 0; i < numInputs; i++) {
        opHandle->inputData[i].data = inputs[i]->data;
        opHandle->inputData[i].size = inputs[i]->size;
        opHandle->inputDesc[i].dataType = inputDesc[i]->dataType;
        opHandle->inputDesc[i].dims = inputDesc[i]->dims;
        opHandle->inputDesc[i].format = inputDesc[i]->format;
        opHandle->inputDesc[i].numDims = inputDesc[i]->numDims;
    }

    for (auto i = 0; i < numOutputs; i++) {
        opHandle->outputData[i].data = outputs[i]->data;
        opHandle->outputData[i].size = outputs[i]->size;
        opHandle->outputDesc[i].dataType = outputDesc[i]->dataType;
        opHandle->outputDesc[i].dims = outputDesc[i]->dims;
        opHandle->outputDesc[i].format = outputDesc[i]->format;
        opHandle->outputDesc[i].numDims = outputDesc[i]->numDims;
    }
    tagAclrtStream *aclStream = (tagAclrtStream *)stream;
    int deviceId = aclStream->deviceId;
    delete opHandle;
    delete[] opName;
    delete opHandle->opAttr;
    delete[] opHandle->inputData;
    delete[] opHandle->inputDesc;
    delete[] opHandle->outputData;
    delete[] opHandle->outputDesc;
    return ACL_SUCCESS;
}

aclError aclopSetAttrFloat(aclopAttr *attr, const char *attrName, float attrValue)
{
    attr->attrs.insert({ attrName, attrValue });
    return 0;
}
aclError aclopSetAttrInt(aclopAttr *attr, const char *attrName, int64_t attrValue)
{
    attr->attrs.insert({ attrName, attrValue });
    return 0;
}

aclFloat16 aclFloatToFloat16(float value)
{
    uint32_t bits = *((uint32_t *)&value);
    uint16_t sign = (bits >> 31) & 0x1;
    uint16_t exponent = ((bits >> 23) & 0xff) - 127 + 15;
    uint16_t mantissa = (bits & 0x7fffff) >> 13;
    uint16_t f16 = (sign << 15) | (exponent << 10) | mantissa;
    return f16;
}

float aclFloat16ToFloat(aclFloat16 f16)
{
    uint32_t sign = (f16 >> 15) & 0x1;
    uint32_t exponent = ((f16 >> 10) & 0x1f) - 15 + 127;
    uint32_t mantissa = (f16 & 0x3ff) << 13;
    uint32_t bits = (sign << 31) | (exponent << 23) | mantissa;
    float f = *((float *)&bits);
    return f;
}
aclDataType aclGetTensorDescType(const aclTensorDesc *desc)
{
    return (aclDataType)0;
}

aclError aclopCreateKernel(const char *opType, const char *kernelId, const char *kernelName, void *binData, int binSize,
    aclopEngineType enginetype, aclDataDeallocator deallocator)
{
    return 0;
}

aclError aclopSetAttrListFloat(aclopAttr *attr, const char *attrName, int numValues, const float *values)
{
    return 0;
}

aclError aclopSetAttrListBool(aclopAttr *attr, const char *attrName, int numValues, const uint8_t *values)
{
    return 0;
}

aclError aclopSetAttrListString(aclopAttr *attr, const char *attrName, int numValues, const char **values)
{
    return 0;
}

aclError aclopSetAttrBool(aclopAttr *attr, const char *attrName, uint8_t attrValue)
{
    return 0;
}

aclError aclopRegisterCompileFunc(const char *opType, aclopCompileFunc func)
{
    return 0;
}

aclError aclopUpdateParams(const char *opType, int numInputs, const aclTensorDesc * const inputDesc[], int numOutputs,
    const aclTensorDesc * const outputDesc[], const aclopAttr *attr)
{
    return 0;
}

aclError aclopSetAttrListInt(aclopAttr *attr, const char *attrName, int numValues, const int64_t *values)
{
    return 0;
}

aclError aclopSetAttrString(aclopAttr *attr, const char *attrName, const char *attrValue)
{
    return 0;
}

aclError aclrtMemsetAsync(void *devPtr, size_t maxCount, int32_t value, size_t count, aclrtStream stream)
{
    if (stream == nullptr) {
        auto deviceId = ENV().getActiveDeviceId();
        stream = DEVICE(deviceId)->GetActiveContext()->m_defaultStream;
    }

    aclopHandle *opHandle = new aclopHandle;
    char *opType = "aclrtMemsetAsync";
    auto len = strlen(opType) + 1; // 结束符多1个字节
    char *newOpName = new char[len];
    auto cpyRet = strcpy_s(newOpName, len, opType);
    if (cpyRet != EOK) {
        printf("strcpy_s error[%d]\r\n", cpyRet);
        return ACL_ERROR_FAILURE;
    }
    opHandle->opName = newOpName;
    opHandle->opAttr = nullptr;
    opHandle->numInputs = 3;
    opHandle->numOutPuts = 1;

    opHandle->inputData = new aclDataBuffer[3];
    opHandle->inputData[0].data = devPtr;
    opHandle->inputData[0].size = maxCount;

    opHandle->inputData[1].data = nullptr;
    opHandle->inputData[1].size = value;

    opHandle->inputData[2].data = nullptr;
    opHandle->inputData[2].size = count;
    auto ret = DEVICE(stream->deviceId)->StreamExecFlowPushOpHandle(stream, opHandle);
    return ret ? ACL_SUCCESS : ACL_ERROR_FAILURE;
}

aclError aclopSetKernelArgs(aclopKernelDesc *kernelDesc, const char *kernelId, uint32_t blockDim, const void *args,
    uint32_t argSize)
{
    return 0;
}

#ifdef __cplusplus
}
#endif