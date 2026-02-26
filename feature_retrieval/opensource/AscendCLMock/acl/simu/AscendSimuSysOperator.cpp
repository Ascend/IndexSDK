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

#include <cmath>
#include <iostream>
#include "securec.h"
#include "AscendSimuExecFlow.h"
#include "../acl_base.h"

using float16_t = uint16_t;

void resetFlat(aclopHandle &opHandle)
{
    uint16_t *flag = reinterpret_cast<uint16_t *>(opHandle.outputData[2].data);
    auto flag_Byte_size = opHandle.outputData[2].size;
    auto flat_num = flag_Byte_size / sizeof(*flag);  // 个数 = 总字节数 / 单个元素字节数

    for (uint32_t i = 0; i < flat_num; i++) {
        flag[i] = 1;
    }
}

void aclrtMemcpyAsyncOperator(aclopHandle &opHandle)
{
    void *dst = opHandle.inputData[0].data;
    int dstMax = opHandle.inputData[0].size;

    const void *src = opHandle.inputData[1].data;
    int count = opHandle.inputData[1].size;

    aclrtMemcpyKind kind = *reinterpret_cast<aclrtMemcpyKind *>(opHandle.inputData[2].data);

    aclrtMemcpy(dst, dstMax, src, count, kind);
}

void MrgbaOperator(aclopHandle &opHandle)
{
    uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
    size_t dataSize = opHandle.inputData[0].size;
    uint8_t *input2 = (uint8_t *)(opHandle.inputData[1].data);
    uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);
    for (size_t i = 0; i < dataSize / 3; i++) {
        output[i] = input1[i] * input2[i] / 255;
        output[i + 1] = input1[i + 1] * input2[i] / 255;
        output[i + 2] = input1[i + 2] * input2[i] / 255;
    }
}

void BitwiseXorOperator(aclopHandle &opHandle)
{
    uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
    size_t dataSize = opHandle.inputData[0].size;
    uint8_t *input2 = (uint8_t *)(opHandle.inputData[1].data);

    uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);

    for (size_t i = 0; i < dataSize; i++) {
        output[i] = input1[i] ^ input2[i];
    }
}

void BitwiseOrOperator(aclopHandle &opHandle)
{
    uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
    size_t dataSize = opHandle.inputData[0].size;
    uint8_t *input2 = (uint8_t *)(opHandle.inputData[1].data);

    uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);

    for (size_t i = 0; i < dataSize; i++) {
        output[i] = input1[i] | input2[i];
    }
}

void BitwiseAndOperator(aclopHandle &opHandle)
{
    uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
    size_t dataSize = opHandle.inputData[0].size;
    uint8_t *input2 = (uint8_t *)(opHandle.inputData[1].data);

    uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);

    for (size_t i = 0; i < dataSize; i++) {
        output[i] = input1[i] & input2[i];
    }
}

void BitwiseNotOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "BitwiseNot simu exec");
}

void AddOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "Add simu exec");
}

void SubtractOperator(aclopHandle &opHandle)
{
    if (opHandle.inputDesc[0].dataType == ACL_FLOAT16) {
        aclFloat16 *input1 = (aclFloat16 *)(opHandle.inputData[0].data);
        size_t dataSize = opHandle.inputData[0].size;
        aclFloat16 *input2 = (aclFloat16 *)(opHandle.inputData[1].data);

        aclFloat16 *output = (aclFloat16 *)(opHandle.outputData[0].data);

        for (size_t i = 0; i < dataSize / 2; i++) {
            output[i] = input1[i] - input2[i];
        }
    } else if (opHandle.inputDesc[0].dataType == ACL_UINT8) {
        uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
        size_t dataSize = opHandle.inputData[0].size;
        uint8_t *input2 = (uint8_t *)(opHandle.inputData[1].data);

        uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);

        for (size_t i = 0; i < dataSize; i++) {
            output[i] = input1[i] - input2[i];
        }
    } else {
        float *input1 = (float *)(opHandle.inputData[0].data);
        size_t dataSize = opHandle.inputData[0].size;
        float *input2 = (float *)(opHandle.inputData[1].data);

        float *output = (float *)(opHandle.outputData[0].data);

        for (size_t i = 0; i < dataSize / 4; i++) {
            output[i] = input1[i] - input2[i];
        }
    }
}

void PowOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "Pow simu exec");
}

void SqrtOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "Sqrt simu exec");
}

void ExpOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "Exp simu exec");
}

void LogOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "Log simu exec");
}

void SqrOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "Sqr simu exec");
}

void ThresholdBinaryOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "ThresholdBinary simu exec");
}

void AddWeightedOperator(aclopHandle &opHandle)
{
    uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
    size_t dataSize = opHandle.inputData[0].size;
    uint8_t *input2 = (uint8_t *)(opHandle.inputData[1].data);

    uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);
    float alpha = opHandle.opAttr->attrs["alpha"];
    float beta = opHandle.opAttr->attrs["beta"];
    float gamma = opHandle.opAttr->attrs["gamma"];
    for (size_t i = 0; i < dataSize; i++) {
        output[i] = input1[i] * alpha + input2[i] * beta + gamma;
    }
}

void AbsDiffOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "AbsDiff simu exec");
}

void MultiplyOperator(aclopHandle &opHandle)
{
    float scale = opHandle.opAttr->attrs["scale"];
    if (opHandle.inputDesc[0].dataType == ACL_FLOAT) {
        float *input1 = (float *)(opHandle.inputData[0].data);
        size_t dataSize = opHandle.inputData[0].size;
        float *input2 = (float *)(opHandle.inputData[1].data);

        float *output = (float *)(opHandle.outputData[0].data);

        for (size_t i = 0; i < dataSize / 4; i++) {
            output[i] = input1[i] * input2[i] * scale;
        }
    } else {
        uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
        size_t dataSize = opHandle.inputData[0].size;
        uint8_t *input2 = (uint8_t *)(opHandle.inputData[1].data);

        uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);

        for (size_t i = 0; i < dataSize; i++) {
            output[i] = input1[i] * input2[i] * scale;
        }
    }
}

void DivideOperator(aclopHandle &opHandle)
{
    if (opHandle.inputDesc[0].dataType == ACL_FLOAT) {
        float *input1 = (float *)(opHandle.inputData[0].data);
        size_t dataSize = opHandle.inputData[0].size;
        float *input2 = (float *)(opHandle.inputData[1].data);

        float *output = (float *)(opHandle.outputData[0].data);

        for (size_t i = 0; i < dataSize / 4; i++) {
            output[i] = input1[i] / input2[i];
        }
    } else {
        uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
        size_t dataSize = opHandle.inputData[0].size;
        uint8_t *input2 = (uint8_t *)(opHandle.inputData[1].data);

        uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);

        for (size_t i = 0; i < dataSize; i++) {
            output[i] = input1[i] / input2[i];
        }
    }
}

void AbsOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "Abs simu exec");
}

void ScaleAddOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "ScaleAdd simu exec");
}

void ConvertToOperator(aclopHandle &opHandle)
{
    int dst_type = (int)(opHandle.opAttr->attrs["dst_type"]);
    if (opHandle.inputDesc[0].dataType == ACL_FLOAT && dst_type == ACL_FLOAT16) {
        float *input1 = (float *)(opHandle.inputData[0].data);
        size_t dataSize = opHandle.inputData[0].size;
        aclFloat16 *output = (aclFloat16 *)(opHandle.outputData[0].data);

        for (size_t i = 0; i < dataSize / 4; i++) {
            output[i] = aclFloatToFloat16(input1[i]);
        }
    } else if (opHandle.inputDesc[0].dataType == ACL_FLOAT16 && dst_type == ACL_FLOAT) {
        aclFloat16 *input1 = (aclFloat16 *)(opHandle.inputData[0].data);
        size_t dataSize = opHandle.inputData[0].size;
        uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);

        for (size_t i = 0; i < dataSize / 2; i++) {
            output[i] = aclFloat16ToFloat(input1[i]);
        }
    } else if (opHandle.inputDesc[0].dataType == ACL_UINT8 && dst_type == ACL_FLOAT16) {
        uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
        size_t dataSize = opHandle.inputData[0].size;
        aclFloat16 *output = (aclFloat16 *)(opHandle.outputData[0].data);
        for (size_t i = 0; i < dataSize; i++) {
            output[i] = aclFloatToFloat16((float)(input1[i]));
        }
    } else {
        std::cout << "Not avaliable for " << opHandle.inputDesc[0].dataType << " to " << dst_type << std::endl;
    }
}

void ClipOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "Clip simu exec");
}

void MinOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "Min simu exec");
}

void MaxOperator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "Max simu exec");
}

void RescaleOperator(aclopHandle &opHandle)
{
    uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
    size_t dataSize = opHandle.inputData[0].size;
    uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);
    float scale = opHandle.opAttr->attrs["scale"];
    float bias = opHandle.opAttr->attrs["bias"];
    for (size_t i = 0; i < dataSize; i++) {
        output[i] = input1[i] * scale + bias;
    }
}

void BlendImageCaptionOperator(aclopHandle &opHandle)
{
    // frame, caption, captionAlpha, captionBg
    // caption*alpha/255 + (1-alpha/255)*(1-Opacity)*frame + captionBg*Opacity*(1-apha/255
    uint8_t *frame = (uint8_t *)(opHandle.inputData[0].data);
    uint8_t *caption = (uint8_t *)(opHandle.inputData[1].data);
    uint8_t *alpha = (uint8_t *)(opHandle.inputData[2].data);
    uint8_t *captionBg = (uint8_t *)(opHandle.inputData[3].data);
    size_t dataSize = opHandle.inputData[2].size;
    // 100*(100/255) + 200*(1-100/255) = 161
    int32_t *tilling = (int32_t *)(opHandle.inputDesc[4].dims);
    float Opacity = tilling[28] / 100.0;
    uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);
    for (size_t i = 0; i < dataSize; i++) {
        output[i] = caption[i] * (alpha[i] * 1.0 / 255) + (1 - alpha[i] * 1.0 / 255) * (1 - Opacity) * frame[i] +
            captionBg[i] * Opacity * (1 - alpha[i] * 1.0 / 255);
    }
}

void BackgroundReplaceOperator(aclopHandle &opHandle)
{
    uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
    uint8_t *input2 = (uint8_t *)(opHandle.inputData[1].data);
    aclFloat16 *mask = (aclFloat16 *)(opHandle.inputData[2].data);
    size_t dataSize = std::min(opHandle.inputData[0].size / 3, opHandle.inputData[2].size / 2);
    // 100*(100/255) + 200*(1-100/255) = 161

    uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);
    for (size_t i = 0; i < dataSize; i++) {
        output[i] = std::round(input2[i] * aclFloat16ToFloat(mask[i]) + input1[i] * \
                    (1 - aclFloat16ToFloat(mask[i])));
    }
}

void BlendImagesOperator(aclopHandle &opHandle)
{
    uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
    uint8_t *mask = (uint8_t *)(opHandle.inputData[1].data);
    uint8_t *input2 = (uint8_t *)(opHandle.inputData[2].data);
    size_t dataSize = opHandle.inputData[1].size;
    uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);
    for (size_t i = 0; i < dataSize; i++) {
        output[i] = input1[i] * (mask[i] * 1.0 / 255) + input2[i] * (1 - mask[i] * 1.0 / 255);
    }
}

void BackgroundReplaceNormOperator(aclopHandle &opHandle)
{
    // background, replace, mask, tilingTensor
    uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
    uint8_t *input2 = (uint8_t *)(opHandle.inputData[1].data);
    aclFloat16 *mask = (aclFloat16 *)(opHandle.inputData[2].data);
    size_t dataSize = std::min(opHandle.inputData[0].size / 3, opHandle.inputData[2].size / 2);
    uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);
    for (size_t i = 0; i < dataSize; i++) {
        output[i] = std::round(input2[i] * (aclFloat16ToFloat(mask[i]) / 255) + input1[i] * \
                    (1 - aclFloat16ToFloat(mask[i]) / 255));
    }
}

void SplitOperator(aclopHandle &opHandle)
{
    uint8_t *input1 = (uint8_t *)(opHandle.inputData[0].data);
    size_t dataSize = opHandle.inputData[0].size;
    for (size_t i = 0; i < dataSize; i++) {
        uint8_t *output = (uint8_t *)(opHandle.outputData[i % opHandle.numOutPuts].data);
        output[i / opHandle.numOutPuts] = input1[i];
    }
}

void MergeOperator(aclopHandle &opHandle)
{
    uint8_t *output = (uint8_t *)(opHandle.outputData[0].data);
    size_t dataSize = opHandle.outputData[0].size;
    for (size_t i = 0; i < dataSize; i++) {
        uint8_t *input = (uint8_t *)(opHandle.inputData[i % opHandle.numInputs].data);
        output[i] = input[i / opHandle.numInputs];
    }
}

void aclrtMemsetAsyncOperator(aclopHandle &opHandle)
{
    void *devPtr = opHandle.inputData[0].data;
    size_t maxCount = opHandle.inputData[0].size;
    int32_t value = opHandle.inputData[1].size;
    size_t count = opHandle.inputData[2].size;
    aclrtMemset(devPtr, maxCount, value, count);
}

void simuDistanceIVFSQ8IP8(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuDistanceIVFSQ8IP8 simu exec");
}
// 算子TransdataShaped 回调函数 这里可以实现CPU的算子算法 当前demo简单未实现 其实这里需要实现一个数据变形
void simuOpTransdataShaped(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "TransdataShaped simu exec");
}

void simuOpTransdataGet(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "TransdataGet simu exec");
}

void simuOpTransdataDist(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "TransdataDist simu exec");
}

void simuOpTransdataIdx(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "TransdataIdx simu exec");
}

void simuOpVecL2Sqr(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "VecL2Sqr simu exec");
}

void simuOpTransdataRaw(aclopHandle &opHandle)
{
    float16_t *outDist = static_cast<float16_t *>(opHandle.outputData[0].data);
    size_t outDistSizeInByte = opHandle.outputData[0].size;

    auto ret = memset_s(outDist, outDistSizeInByte, 0, outDistSizeInByte);
    if (ret != EOK) {
        ACL_APP_LOG(ACL_ERROR, "simuOpTransdataRaw memset failed !!!");
    }
    ACL_APP_LOG(ACL_INFO, "TransdataRaw simu exec");
}

void simuTopkFlat(aclopHandle &opHandle)
{
    int64_t *outLabel = static_cast<int64_t *>(opHandle.outputData[1].data);
    size_t outLabelSizeInByte = opHandle.outputData[1].size;

    float16_t *outDist = static_cast<float16_t *>(opHandle.outputData[0].data);
    size_t outDistSizeInByte = opHandle.outputData[0].size;

    memset_s(outLabel, outLabelSizeInByte, 0, outLabelSizeInByte);
    memset_s(outDist, outDistSizeInByte, 0, outDistSizeInByte);
    ACL_APP_LOG(ACL_INFO, "simuTopkFlat simu exec");
}

void simuDistanceComputeFlatMin64(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuDistanceComputeFlatMin64 simu exec");
    resetFlat(opHandle);
}

void simuL2Norm(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuL2Norm simu exec");
}

void simuDistanceFlatL2MinsAt(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatL2MinsAt simu exec");
}

void simuTopkIvf(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuTopkIvf simu exec");
}

void simuResidualIvf(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuResidualIvf simu exec");
}

void simuDistanceIVFSQ8L2(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuDistanceIVFSQ8L2 simu exec");
}

void simuDistanceBatchMaskGenerator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuDistanceBatchMaskGenerator simu exec");
}

void simuDistanceMaskGeneratorWithExtra(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuDistanceMaskGeneratorWithExtra simu exec");
}

void simuDistanceBatchMaskGeneratorWithExtra(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceBatchMaskGeneratorWithExtra simu exec");
}

void simuDistanceMaskGenerator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceMaskGenerator simu exec");
}
void simuDistanceFlatHamming(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatHamming simu exec");
}

void simuDistanceFlatHammingWithMask(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatHammingWithMask simu exec");
}

void simuL2NormFlatSub(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatL2MinsAt simu exec");
}

void simuDistanceFlatL2At(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatL2At simu exec");
}

void simuSubcentAccum(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "SubcentAccum simu exec");
}

void simuDistanceFlatL2Mins(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatL2Mins simu exec");
}

void simuTopkIvfsqtL1(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "TopkIvfsqtL1 simu exec");
}

void simuDistanceFlatSubcenters(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatSubcenters simu exec");
}

void simuTopkIvfsqtL2(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "TopkIvfsqtL2 simu exec");
}

void simuDistanceIVFSQ8IPX(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceIVFSQ8IPX simu exec");
}

void simuTopkIvfFuzzy(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "TopkIvfFuzzy simu exec");
}

void simuDistanceFlatIPMaxs(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatIPMaxs simu exec");
    resetFlat(opHandle);
    return;
}

void simuDistanceFlatIpByIdx(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatIpByIdx simu exec");
}

void simuDistanceFlatIpByIdxWithTable(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatIpByIdxWithTable simu exec");
}

void simuDistanceFlatIp(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatIp simu exec");
}

void simuDistanceFlatIpWithTable(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatIpWithTable simu exec");
}

void simuDistanceFilter(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuDistanceFilter simu exec");
}

void simuDistanceFlatIpByIdx2(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuDistanceFlatIpByIdx2 simu exec");
}

void simuRemovedataAttr(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuRemovedataAttr simu exec");
}

void simuTransdataCustomAttr(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuTransdataCustomAttr simu exec");
}

void simuRemovedataCustomAttr(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuRemovedataCustomAttr simu exec");
}

void simuRemovedataShaped(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "RemovedataShaped simu exec");
}

void simuInt8L2Norm(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "Int8L2Norm simu exec");
}

void simuDistanceInt8CosMaxs(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceInt8CosMaxs simu exec");
    resetFlat(opHandle);
}

void simuDistanceInt8CosMaxsWithMask(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceInt8CosMaxsWithMask simu exec");
    resetFlat(opHandle);
}

void simuDistanceFlatIPMaxsWithMask(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatIPMaxsWithMask simu exec");
}

void simuCorrCompute(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "CorrCompute simu exec");
}

void simuKmUpdateCentroids(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "KmUpdateCentroids simu exec");
}

void simuL2NormTypingInt8(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "L2NormTypingInt8 simu exec");
}

void simuDistanceL2MinsInt8At(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceL2MinsInt8At simu exec");
}

void simuCodesQuantify(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "CodesQuantify simu exec");
}

void simuDistanceSQ8L2Mins(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceSQ8L2Mins simu exec");
}

void simuDistanceInt8CosMaxsFilter(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceInt8CosMaxsFilter simu exec");
}

void simuDistanceInt8L2Mins(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceInt8L2Mins simu exec");
    resetFlat(opHandle);
}

void simuDistanceInt8L2MinsWithMask(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceInt8L2MinsWithMask simu exec");
    resetFlat(opHandle);
}

void simuDistanceSQ8IPMaxsDim64(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceSQ8IPMaxsDim64 simu exec");
}

void simuDistanceInt8L2FullMins(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceInt8L2FullMins simu exec");
    resetFlat(opHandle);
}

void simuDistanceInt8L2FullMinsWithMask(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceInt8L2FullMinsWithMask simu exec");
    resetFlat(opHandle);
}

void simuDistanceFlatIPMaxsBatch(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatIPMaxsBatch simu exec");
    resetFlat(opHandle);
}

void simuTopkMultisearch(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "TopkMultisearch simu exec");
}

void simuDistanceMaskedSQ8L2Mins(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceMaskedSQ8L2Mins simu exec");
}

void simuDistanceMaskedSQ8IPMaxsDim64(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuDistanceMaskedSQ8IPMaxsDim64 simu exec");
}

void simuCidFilter(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuCidFilter simu exec");
}

void simuDistanceInt8L2MinsWoQueryNorm(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuDistanceInt8L2MinsWoQueryNorm simu exec");
}

void simuDistanceFlatL2(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatL2 simu exec");
}

void simuDistanceFlatIP(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuDistanceFlatIP simu exec");
}

void simuDistanceBinaryFloat(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuDistanceBinaryFloat simu exec");
}

void simuDistanceBatchValMaskGenerator(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuDistanceBatchValMaskGenerator simu exec");
}

void simuAscendcL2Norm(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendcL2Norm simu exec");
}

void simuAscendcDistInt8FlatL2(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendcDistInt8FlatL2 simu exec");
}

void simuAscendcDistInt8FlatCos(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendcDistInt8FlatCos simu exec");
}

void SimuAscendDistanceComputeQC(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendDistanceComputeQC simu exec");
}

void SimuAscendTransdataShapedSp(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendTransdataShapedSp simu exec");
}

void SimuAscendVecL2SqrSp(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendVecL2SqrSp simu exec");
}

void SimuAscendFpToFp16(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendFpToFp16 simu exec");
}

void SimuAscendTopkIvfSpL1(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendTopkIvfSpL1 simu exec");
}

void SimuAscendTopkIvfSp(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendTopkIvfSp simu exec");
}

void SimuAscendDistanceIVFSpIntL2Mins(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendDistanceIVFSpIntL2Mins simu exec");
}

void SimuAscendTopkMultisearchIvfV2(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendTopkMultisearchIvfV2 simu exec");
}

void SimuAscendIvfCidFilter3(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendIvfCidFilter3 simu exec");
}

void SimuAscendDistanceMaskedIVFSpIntL2Mins(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "simuAscendDistanceMaskedIVFSpIntL2Mins simu exec");
}

void SimuDistanceFlatIpMaxsWithExtraScore(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatIPMaxsWithExtraScore simu exec");
}

void SimuDistanceInt8CosMaxsWithMaskExtraScore(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceInt8CosMaxsWithMaskExtraScore simu exec");
}

void SimuDistanceFlatIPMaxsWithScale(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatIPMaxsWithScale simu exec");
}

void SimuDistanceFlatIPMaxsNoScoreWithScale(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "DistanceFlatIPMaxsNoScoreWithScale simu exec");
}

void SimuVstarBaseAddMatMul(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "VstarBaseAddMatMul simu exec");
}

void SimuIvfMultiSpTopkL3(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "IvfMultiSpTopkL3 simu exec");
}

void SimuIvfSpTopkL2WithMask(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "IvfSpTopkL2WithMask simu exec");
}

void SimuIvfSpTopkL3(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "IvfSpTopkL3 simu exec");
}

void SimuIvfSpTopkL1(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "IvfSpTopkL1 simu exec");
}

void SimuIvfSpTopkL2(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "IvfSpTopkL2 simu exec");
}

void SimuIvfMultiSpTopkL2(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "IvfMultiSpTopkL2 simu exec");
}

void SimuVstarComputeL1(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "VstarComputeL1 simu exec");
}

void SimuVstarComputeL2(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "VstarComputeL2 simu exec");
}

void SimuVSC3(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "VSC3 simu exec");
}

void SimuVSM3(aclopHandle &opHandle)
{
    ACL_APP_LOG(ACL_INFO, "VSM3 simu exec");
}

void simuOpInstall()
{
    REG_OP("VSM3", SimuVSM3);
    REG_OP("VstarComputeL1", SimuVstarComputeL1);
    REG_OP("VstarComputeL2", SimuVstarComputeL2);
    REG_OP("VSC3", SimuVSC3);
    REG_OP("IvfSpTopkL1", SimuIvfSpTopkL1);
    REG_OP("IvfSpTopkL2", SimuIvfSpTopkL2);
    REG_OP("IvfSpTopkL3", SimuIvfSpTopkL3);
    REG_OP("IvfSpTopkL2WithMask", SimuIvfSpTopkL2WithMask);
    REG_OP("IvfMultiSpTopkL3", SimuIvfMultiSpTopkL3);
    REG_OP("IvfMultiSpTopkL2", SimuIvfMultiSpTopkL2);
    REG_OP("VstarBaseAddMatMul", SimuVstarBaseAddMatMul);
    REG_OP("DistanceComputeQC", SimuAscendDistanceComputeQC);
    REG_OP("TransdataShapedSp", SimuAscendTransdataShapedSp);
    REG_OP("VecL2SqrSp", SimuAscendVecL2SqrSp);
    REG_OP("FpToFp16", SimuAscendFpToFp16);
    REG_OP("TopkIvfSpL1", SimuAscendTopkIvfSpL1);
    REG_OP("TopkIvfSp", SimuAscendTopkIvfSp);
    REG_OP("DistanceIVFSpIntL2Mins", SimuAscendDistanceIVFSpIntL2Mins);
    REG_OP("TopkMultisearchIvfV2", SimuAscendTopkMultisearchIvfV2);
    REG_OP("IvfCidFilter3", SimuAscendIvfCidFilter3);
    REG_OP("DistanceMaskedIVFSpIntL2Mins", SimuAscendDistanceMaskedIVFSpIntL2Mins);
    REG_OP("TransdataShaped", simuOpTransdataShaped);
    REG_OP("VecL2Sqr", simuOpVecL2Sqr);
    REG_OP("TransdataRaw", simuOpTransdataRaw);
    REG_OP("TransdataGet", simuOpTransdataGet);
    REG_OP("TransdataDist", simuOpTransdataDist);
    REG_OP("TransdataIdx", simuOpTransdataIdx);
    REG_OP("TopkFlat", simuTopkFlat);
    REG_OP("TopkIvf", simuTopkIvf);
    REG_OP("DistanceComputeFlatMin64", simuDistanceComputeFlatMin64);
    REG_OP("aclrtMemcpyAsync", aclrtMemcpyAsyncOperator);
    REG_OP("L2Norm", simuL2Norm);
    REG_OP("DistanceFlatL2MinsAt", simuDistanceFlatL2MinsAt);
    REG_OP("DistanceMaskGenerator", simuDistanceMaskGenerator);
    REG_OP("DistanceFlatHamming", simuDistanceFlatHamming);
    REG_OP("DistanceFlatHammingWithMask", simuDistanceFlatHammingWithMask);
    REG_OP("L2NormFlatSub", simuL2NormFlatSub);
    REG_OP("DistanceFlatL2At", simuDistanceFlatL2At);
    REG_OP("SubcentAccum", simuSubcentAccum);
    REG_OP("DistanceFlatL2Mins", simuDistanceFlatL2Mins);
    REG_OP("TopkIvfsqtL1", simuTopkIvfsqtL1);
    REG_OP("DistanceFlatSubcenters", simuTopkIvfsqtL1);
    REG_OP("TopkIvfsqtL2", simuTopkIvfsqtL1);
    REG_OP("DistanceIVFSQ8IPX", simuDistanceIVFSQ8IPX);
    REG_OP("TopkIvfFuzzy", simuTopkIvfsqtL1);
    REG_OP("DistanceIVFSQ8IP8", simuDistanceIVFSQ8IP8);
    REG_OP("ResidualIvf", simuResidualIvf);
    REG_OP("DistanceIVFSQ8L2", simuDistanceIVFSQ8L2);
    REG_OP("DistanceFlatIPMaxs", simuDistanceFlatIPMaxs);
    REG_OP("DistanceFlatIpByIdx", simuDistanceFlatIpByIdx);
    REG_OP("DistanceFlatIpByIdxWithTable", simuDistanceFlatIpByIdxWithTable);
    REG_OP("DistanceFlatIp", simuDistanceFlatIp);
    REG_OP("DistanceFlatIpWithTable", simuDistanceFlatIpWithTable);
    REG_OP("DistanceFilter", simuDistanceFilter);
    REG_OP("DistanceFlatIpByIdx2", simuDistanceFlatIpByIdx2);
    REG_OP("RemovedataAttr", simuRemovedataAttr);
    REG_OP("TransdataCustomAttr", simuTransdataCustomAttr);
    REG_OP("RemovedataCustomAttr", simuRemovedataCustomAttr);
    REG_OP("RemovedataShaped", simuRemovedataShaped);
    REG_OP("Int8L2Norm", simuInt8L2Norm);
    REG_OP("DistanceInt8CosMaxs", simuDistanceInt8CosMaxs);
    REG_OP("DistanceFlatIPMaxsWithMask", simuDistanceFlatIPMaxsWithMask);
    REG_OP("CorrCompute", simuCorrCompute);
    REG_OP("KmUpdateCentroids", simuKmUpdateCentroids);
    REG_OP("L2NormTypingInt8", simuL2NormTypingInt8);
    REG_OP("DistanceL2MinsInt8At", simuDistanceL2MinsInt8At);
    REG_OP("CodesQuantify", simuCodesQuantify);
    REG_OP("DistanceSQ8L2Mins", simuDistanceSQ8L2Mins);
    REG_OP("DistanceInt8CosMaxsFilter", simuDistanceInt8CosMaxsFilter);
    REG_OP("DistanceInt8L2Mins", simuDistanceInt8L2Mins);
    REG_OP("DistanceSQ8IPMaxsDim64", simuDistanceSQ8IPMaxsDim64);
    REG_OP("DistanceComputeFlat", simuDistanceIVFSQ8IP8);
    REG_OP("DistanceIVFSQ8IP4", simuDistanceIVFSQ8IP8);
    REG_OP("DistanceInt8L2FullMins", simuDistanceInt8L2FullMins);
    REG_OP("DistanceFlatIPMaxsBatch", simuDistanceFlatIPMaxsBatch);
    REG_OP("TopkMultisearch", simuTopkMultisearch);
    REG_OP("DistanceMaskedSQ8L2Mins", simuDistanceMaskedSQ8L2Mins);
    REG_OP("DistanceMaskedSQ8IPMaxsDim64", simuDistanceMaskedSQ8IPMaxsDim64);
    REG_OP("CidFilter", simuCidFilter);
    REG_OP("DistanceInt8CosMaxsWithMask", simuDistanceInt8CosMaxsWithMask);
    REG_OP("DistanceInt8L2MinsWithMask", simuDistanceInt8L2MinsWithMask);
    REG_OP("DistanceInt8L2FullMinsWithMask", simuDistanceInt8L2FullMinsWithMask);
    REG_OP("DistanceBatchMaskGenerator", simuDistanceBatchMaskGenerator);
    REG_OP("DistanceMaskGeneratorWithExtra", simuDistanceMaskGeneratorWithExtra);
    REG_OP("DistanceBatchMaskGeneratorWithExtra", simuDistanceBatchMaskGeneratorWithExtra);
    REG_OP("DistanceInt8L2MinsWoQueryNorm", simuDistanceInt8L2MinsWoQueryNorm);
    REG_OP("DistanceFlatL2", simuDistanceFlatL2);
    REG_OP("DistanceFlatIP", simuDistanceFlatIP);
    REG_OP("DistanceBinaryFloat", simuDistanceBinaryFloat);
    REG_OP("DistanceBatchValMaskGenerator", simuDistanceBatchValMaskGenerator);
    REG_OP("AscendcL2Norm", simuAscendcL2Norm);
    REG_OP("AscendcDistInt8FlatL2", simuAscendcDistInt8FlatL2);
    REG_OP("AscendcDistInt8FlatCos", simuAscendcDistInt8FlatCos);
    REG_OP("BitwiseXor", BitwiseXorOperator);
    REG_OP("BitwiseOr", BitwiseOrOperator);
    REG_OP("BitwiseAnd", BitwiseAndOperator);
    REG_OP("BitwiseNot", BitwiseNotOperator);
    REG_OP("aclrtMemsetAsync", aclrtMemsetAsyncOperator);
    REG_OP("Add", AddOperator);
    REG_OP("Sub", SubtractOperator);
    REG_OP("Pow", PowOperator);
    REG_OP("Sqrt", SqrtOperator);
    REG_OP("Exp", ExpOperator);
    REG_OP("Log", LogOperator);
    REG_OP("Sqr", SqrOperator);
    REG_OP("ThresholdBinary", ThresholdBinaryOperator);
    REG_OP("AddWeighted", AddWeightedOperator);
    REG_OP("AbsDiff", AbsDiffOperator);
    REG_OP("Multiply", MultiplyOperator);
    REG_OP("Divide", DivideOperator);
    REG_OP("Abs", AbsOperator);
    REG_OP("ScaleAdd", ScaleAddOperator);
    REG_OP("Cast", ConvertToOperator);
    REG_OP("Clip", ClipOperator);
    REG_OP("Min", MinOperator);
    REG_OP("Max", MaxOperator);
    REG_OP("Rescale", RescaleOperator);
    REG_OP("BackgroundReplace", BackgroundReplaceOperator);
    REG_OP("background_replace", BackgroundReplaceOperator);
    REG_OP("background_replace_normalize", BackgroundReplaceNormOperator);
    REG_OP("blend_image_caption", BlendImageCaptionOperator);
    REG_OP("BlendImages", BlendImagesOperator);
    REG_OP("SplitD", SplitOperator);
    REG_OP("ConcatD", MergeOperator);
    REG_OP("MRGBA", MrgbaOperator);
    REG_OP("DistanceFlatIPMaxsWithExtraScore", SimuDistanceFlatIpMaxsWithExtraScore);
    REG_OP("DistanceInt8CosMaxsWithMaskExtraScore", SimuDistanceInt8CosMaxsWithMaskExtraScore);
    REG_OP("DistanceFlatIPMaxsWithScale", SimuDistanceFlatIPMaxsWithScale);
    REG_OP("DistanceFlatIPMaxsNoScoreWithScale", SimuDistanceFlatIPMaxsNoScoreWithScale);
}

void simuOpUninstall()
{
    UNREG_OP("VSM3");
    UNREG_OP("VstarComputeL1");
    UNREG_OP("VstarComputeL2");
    UNREG_OP("VSC3");
    UNREG_OP("IvfSpTopkL1");
    UNREG_OP("IvfSpTopkL2");
    UNREG_OP("IvfSpTopkL3");
    UNREG_OP("IvfSpTopkL2WithMask");
    UNREG_OP("IvfMultiSpTopkL3");
    UNREG_OP("IvfMultiSpTopkL2");
    UNREG_OP("VstarBaseAddMatMul");
    UNREG_OP("DistanceComputeQC");
    UNREG_OP("TransdataShapedSp");
    UNREG_OP("VecL2SqrSp");
    UNREG_OP("FpToFp16");
    UNREG_OP("TopkIvfSpL1");
    UNREG_OP("TopkIvfSp");
    UNREG_OP("DistanceIVFSpIntL2Mins");
    UNREG_OP("TopkMultisearchIvfV2");
    UNREG_OP("IvfCidFilter3");
    UNREG_OP("DistanceMaskedIVFSpIntL2Mins");
    UNREG_OP("TransdataShaped");
    UNREG_OP("VecL2Sqr");
    UNREG_OP("TransdataRaw");
    UNREG_OP("TopkFlat");
    UNREG_OP("TopkIvf");
    UNREG_OP("DistanceComputeFlatMin64");
    UNREG_OP("aclrtMemcpyAsync");
    UNREG_OP("L2Norm");
    UNREG_OP("DistanceFlatL2MinsAt");
    UNREG_OP("DistanceMaskGenerator");
    UNREG_OP("DistanceFlatHamming");
    UNREG_OP("DistanceFlatHammingWithMask");
    UNREG_OP("L2NormFlatSub");
    UNREG_OP("DistanceFlatL2At");
    UNREG_OP("simuSubcentAccum");
    UNREG_OP("simuDistanceFlatL2Mins");
    UNREG_OP("TopkIvfsqtL1");
    UNREG_OP("DistanceFlatSubcenters");
    UNREG_OP("TopkIvfsqtL2");
    UNREG_OP("DistanceIVFSQ8IPX");
    UNREG_OP("TopkIvfFuzzy");
    UNREG_OP("DistanceIVFSQ8IP8");
    UNREG_OP("DistanceFlatL2Mins");
    UNREG_OP("ResidualIvf");
    UNREG_OP("DistanceIVFSQ8L2");
    UNREG_OP("DistanceFlatIPMaxs");
    UNREG_OP("DistanceFlatIpByIdx");
    UNREG_OP("DistanceFlatIpByIdxWithTable");
    UNREG_OP("DistanceFlatIp");
    UNREG_OP("DistanceFlatIpWithTable");
    UNREG_OP("DistanceFilter");
    UNREG_OP("DistanceFlatIpByIdx2");
    UNREG_OP("RemovedataAttr");
    UNREG_OP("TransdataCustomAttr");
    UNREG_OP("RemovedataCustomAttr");
    UNREG_OP("RemovedataShaped");
    UNREG_OP("Int8L2Norm");
    UNREG_OP("DistanceInt8CosMaxs");
    UNREG_OP("DistanceFlatIPMaxsWithMask");
    UNREG_OP("CorrCompute");
    UNREG_OP("KmUpdateCentroids");
    UNREG_OP("L2NormTypingInt8");
    UNREG_OP("DistanceL2MinsInt8At");
    UNREG_OP("CodesQuantify");
    UNREG_OP("DistanceSQ8L2Mins");
    UNREG_OP("DistanceInt8CosMaxsFilter");
    UNREG_OP("DistanceInt8L2Mins");
    UNREG_OP("DistanceInt8L2MinsWithMask");
    UNREG_OP("DistanceSQ8IPMaxsDim64");
    UNREG_OP("DistanceComputeFlat");
    UNREG_OP("DistanceIVFSQ8IP4");
    UNREG_OP("DistanceInt8L2FullMins");
    UNREG_OP("DistanceInt8L2FullMinsWithMask");
    UNREG_OP("DistanceFlatIPMaxsBatch");
    UNREG_OP("TopkMultisearch");
    UNREG_OP("DistanceMaskedSQ8L2Mins");
    UNREG_OP("DistanceMaskedSQ8IPMaxsDim64");
    UNREG_OP("CidFilter");
    UNREG_OP("DistanceInt8CosMaxsWithMask");
    UNREG_OP("DistanceBatchMaskGenerator");
    UNREG_OP("DistanceBatchMaskGeneratorWithExtra");
    UNREG_OP("DistanceInt8L2MinsWoQueryNorm");
    UNREG_OP("DistanceFlatL2");
    UNREG_OP("DistanceFlatIP");
    UNREG_OP("DistanceBinaryFloat");
    UNREG_OP("DistanceBatchValMaskGenerator");
    UNREG_OP("AscendcL2Norm");
    UNREG_OP("AscendcDistInt8FlatL2");
    UNREG_OP("AscendcDistInt8FlatCos");
    UNREG_OP("BitwiseXor");
    UNREG_OP("BitwiseOr");
    UNREG_OP("BitwiseAnd");
    UNREG_OP("BitwiseNot");
    UNREG_OP("aclrtMemsetAsync");
    UNREG_OP("Add");
    UNREG_OP("Sub");
    UNREG_OP("Pow");
    UNREG_OP("Sqrt");
    UNREG_OP("Exp");
    UNREG_OP("Log");
    UNREG_OP("Sqr");
    UNREG_OP("ThresholdBinary");
    UNREG_OP("AddWeighted");
    UNREG_OP("AbsDiff");
    UNREG_OP("Multiply");
    UNREG_OP("Divide");
    UNREG_OP("Abs");
    UNREG_OP("ScaleAdd");
    UNREG_OP("Cast");
    UNREG_OP("Clip");
    UNREG_OP("Min");
    UNREG_OP("Max");
    UNREG_OP("Rescale");
    UNREG_OP("BackgroundReplace");
    UNREG_OP("background_replace");
    UNREG_OP("background_replace_normalize");
    UNREG_OP("blend_image_caption");
    UNREG_OP("BlendImages");
    UNREG_OP("SplitD");
    UNREG_OP("ConcatD");
    UNREG_OP("MRGBA");
    UNREG_OP("DistanceFlatIPMaxsWithExtraScore");
    UNREG_OP("DistanceInt8CosMaxsWithMaskExtraScore");
    UNREG_OP("DistanceFlatIPMaxsWithScale");
    UNREG_OP("DistanceFlatIPMaxsNoScoreWithScale");
}
