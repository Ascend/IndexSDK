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


#ifndef IVF_SP_CODEBOOK_TRAINER_INCLUDED
#define IVF_SP_CODEBOOK_TRAINER_INCLUDED

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include <stdio.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <ascend/utils/fp16.h>
#include <ascenddaemon/AscendResourcesProxy.h>
#include <ascenddaemon/utils/AscendOperator.h>
#include <ascenddaemon/utils/AscendTensor.h>
#include <common/ErrorCode.h>
#include <common/AscendFp16.h>
#include <common/utils/CommonUtils.h>
#include <common/utils/LogUtils.h>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <lapacke.h>
#include <omp.h>

#include "ascenddaemon/utils/IoUtil.h"
#include "common/utils/SocUtils.h"

namespace ascendSearch {

union FormatConvert {
    float16_t data;
    uint16_t value;
};

/**
 * @brief 用户使用时输入的参数结构体; IVFSPCodeBoookTrainer使用其中的参数进行初始化
 *
 */
struct IVFSPCodeBookTrainerInitParam {
    IVFSPCodeBookTrainerInitParam() {};

    int nlist = 256;
    int numIter = 1;
    float ratio = 1.0;
    int dim = 128;
    int nonzeroNum = 64;
    int device = 0;
    int batchSize = 32768;
    int codeNum = 32768;
    bool verbose = true;
    std::string learnDataPath = "";
    std::string codeBookOutputDir = "";
    const float *memLearnData = nullptr;
    size_t memLearnDataSize = 0;
    bool trainAndAdd = false;
};

class IVFSPCodeBookTrainer {
public:
    explicit IVFSPCodeBookTrainer(const IVFSPCodeBookTrainerInitParam &initParam);

    ~IVFSPCodeBookTrainer();

    void Train(int numIter);

    void ReadFile(const std::string &learnDataPath, float ratio = 1.0);

    void ReadMemLearnData(const float *memLearnData, size_t memLearnDataSize, float ratio = 1.0);

    std::vector<float> GetCodeBook();

private:
    int nlist = 1024;
    int dim = 256;
    int nonzeroNum = 128;
    int device = 0;
    int prevDevice = -1;
    int batchSize = 32768;
    int codeNum = 32768;
    bool verbose = true;
    bool trainAndAdd = false; // 布尔值去判断是否打印码本保存路径
    int nb = 0;
    std::string codeBookOutputPath;
    std::mt19937 randomNumberGenerator;

    std::vector<std::vector<float>> codeBookByNList; // 将codeBook数据按照nlist分片 (每个shape = dim * nonzeroNum)
    std::vector<float16_t> codeBookReshaped; // 码本进行Zz分形然后转成fp16形式
    std::vector<float> codeBookFp32; // Fp32格式存储的码本

    std::vector<float> learnDataFp32; // 被训练的底库数据(fp32格式)
    std::vector<float16_t> learnDataFp16; // 被训练的底库数据(fp16格式)
    std::vector<std::vector<float16_t>> learnDataByBatch; // 将learnData按照batchSize分片
    std::vector<std::vector<float16_t>> learnDataByNList; // 将learnData按照nonzeroIdx拆分到不同的桶内

    std::shared_ptr<AscendResourcesProxy> resources = nullptr;
    aclrtStream defaultStream = nullptr;

    std::unique_ptr<AscendOperator> distQCOp;
    std::unique_ptr<AscendOperator> matMulQCOp;
    std::unique_ptr<AscendOperator> matMulCBOp;
    std::unique_ptr<AscendOperator> corrOp;

    APP_ERROR ResetDistanceComputeQC();
    APP_ERROR ResetMatmulComputeQC();
    APP_ERROR ResetMatmulComputeCB();
    APP_ERROR ResetCorrCompute();

    /** 计算操作函数 **/
    void CalNonzeroIdx(std::vector<float16_t> &learnDataPerBatch);

    void OrthCodeBook(); // 对码本进行垂直化处理

    void UpdateCodeBookImpl();

    void UpdateCodeBookAcl(const std::vector<float16_t> &learnDataPerNList, int nlistId);

    int SingleFloatSVD(int m, int n, std::vector<float> &A, std::vector<float> &U, char jobu);

    // 使用KMeans算法对码本进行初始化
    void KMeansUpdateCodeBook();

    void PerformKMeansOnLearnData();

    /** 形状操作函数 **/
    void ReshapeCodeBook(); // 对fp32码本进行Zz分形后转成Fp16格式，存入codeBookReshaped中

    // 将一个fp32类型的码本 (dim * nlist * nonzeroNum) 拆分进入codeBookByNList内
    void SplitCodeBookByNList();

    // 将codeBookByNList中的码本数据合并为1个完整的码本数据 (dim * nlist * nonzeroNum), SplitCodeBookByNList的逆操作
    void MergeCodeBookByNList();

    // 将训练数据按照batch大小拆分，转成fp16后放入learnDataByBatch
    void SplitLearnDataByBatch();

    template <typename T>
    void Transpose(const std::vector<T> &src, std::vector<T> &dst, int srcRow, int srcCol);

    template <typename T>
    void ZzFormatReshape(const std::vector<T> &src, std::vector<T> &dst, int row, int col); // 对一个ND排布的矩阵进行Zz分形

    template <typename T>
    void NzFormatReshape(const std::vector<T> &src, std::vector<T> &dst, int row, int col);

    /** 工具类函数 **/

    void PreprocessLearnData(float ratio);

    void InitCodeBook(); // 通过底库数据对codeBook进行初始化

    void Fp32ToFp16(const std::vector<float> &src, std::vector<float16_t> &dst);

    void Fp16ToFp32(const std::vector<float16_t> &src, std::vector<float> &dst);

    void ArgMaxAlongNList(const std::vector<float16_t> &distFromEachNList, std::vector<float16_t> &labels);

    void CheckParams() const;

    void SampleLearnData(float ratio);

    void SetDeviceAndInitAscendResources();

    template <typename T>
    void PutDataInNList(const std::vector<float16_t> &learnDataPerBatch, const std::vector<T> &labels);

    void PadCurrNListDataWithZeros(std::vector<float16_t> &learnDataPerNList);

    void CopyFirstNColumns(const std::vector<float> &src, std::vector<float> &dst,
                           int srcCol, int dstCol, int copyColNum);
    
    template <typename T, int Dim>
    AscendTensor<T, Dim> CreateTensorAndAllocDeviceMem(AscendMemory &mem, const int (&sizes)[Dim], T *cpuDataPtr);

    void SaveCodeBook();

    inline void LogProgress(const std::string &message, int current, int total)
    {
        if (verbose) {
            printf("\r-----%s:[%d/%d]", message.c_str(), current + 1, total);
            fflush(stdout);
            if (current + 1 == total) {
                printf("\n");
            }
        }
    }

    inline void LogInfo()
    {
        if (verbose) {
            printf(" =========== Current CodeBook Trainer Info =========== \n");
            printf("dim = %d, nlist = %d, nonzeroNum = %d, nb = %d;\n", dim, nlist, nonzeroNum, nb);
            if (trainAndAdd) {
                printf("You enabled 'trainAndAdd' so your codebook will be added to the "
                    "index directly without saving to disk.\n");
            } else {
                printf("Output codebook dir = %s;\n", codeBookOutputPath.c_str());
            }
            printf(" ===================================================== \n");
        }
    }

    /* 执行算子 */
    void RunDistanceComputeQC(AscendTensor<float16_t, DIMS_2> &query,
                              AscendTensor<float16_t, DIMS_4> &codebookFormatted,
                              AscendTensor<float16_t, DIMS_1> &bucket,
                              AscendTensor<float16_t, DIMS_2> &dist,
                              AscendTensor<uint16_t, DIMS_2> &flag);

    void RunCorrCompute(AscendTensor<float16_t, DIMS_4> &queryFormattedShape,
                        AscendTensor<uint64_t, DIMS_1> &actualNumShape,
                        AscendTensor<float16_t, DIMS_2> &svdInputShape,
                        AscendTensor<uint16_t, DIMS_2> &flag);
};

} // namespace ascendSearch

#endif // IVF_SP_CODEBOOK_TRAINER_INCLUDED