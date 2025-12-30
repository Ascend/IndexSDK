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


#include "ascenddaemon/IVFSPCodeBookTrainer.h"

namespace ascendSearch {

namespace {
    constexpr int CORE_NUM = 8;
    constexpr int CUBE_ALIGN = 16;
    constexpr int FLAG_SIZE = 16; //  = 256 / (sizeof(uint16_t) * 8)
    constexpr uint64_t MAX_LEARN_DATA_SIZE = 100LLU * 1024 * 1024 * 1024; // 限制码本训练数据最大为100GB
    constexpr int RANDOM_SEED = 42;
    constexpr int KMEANS_ITER = 16;
    constexpr int HIGH_DIMS_NLIST_LIMIT = 2048; // 对于大维度数据，限制nlist的最大大小，防止码本大小过大
    constexpr int MAX_NONZERO_NUM = 128; // dim2 (nonzeroNum) 上限
    constexpr int MAX_TRAIN_ITER = 20;
    const std::vector<int> DIMS = {64, 128, 256, 512, 768};
    const std::vector<int> NLISTS = { 256, 512, 1024, 2048, 4096, 8192, 16384 };
    const std::vector<int> HIGH_DIMS = { 512, 768 };
    constexpr int BATCH_SIZE_LIMIT = 32768;
    constexpr size_t LOCAL_SECUREC_MEM_MAX_LEN = 2147483640; // max secure_c buffer size (2GB - 1) rounded down by 8
}

template<typename D, typename S>
void CopyDataForLoad(D *dest, const S *src, size_t sizeBytes, size_t &offset, size_t dataLen)
{
    ASCEND_THROW_IF_NOT_MSG(dataLen >= offset + sizeBytes, "memcpy error: insufficient data length.\n");
    int err = 0;
    size_t dstMoveCount = LOCAL_SECUREC_MEM_MAX_LEN / sizeof(D);
    size_t srcMoveCount = LOCAL_SECUREC_MEM_MAX_LEN / sizeof(S);
    size_t offsetMoveCount = offset / sizeof(S);
    size_t copyCounts = sizeBytes / LOCAL_SECUREC_MEM_MAX_LEN;
    for (size_t i = 0; i < copyCounts; ++i) {
        err = memcpy_s(dest + i * dstMoveCount,
                       std::min(LOCAL_SECUREC_MEM_MAX_LEN, dataLen - offset - i * LOCAL_SECUREC_MEM_MAX_LEN),
                       src + offsetMoveCount + i * srcMoveCount,
                       LOCAL_SECUREC_MEM_MAX_LEN);
        ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "memcpy (error %d)", err);
    }
    size_t remainBytes = sizeBytes - (copyCounts * LOCAL_SECUREC_MEM_MAX_LEN);
    err = memcpy_s(dest + copyCounts * dstMoveCount,
                   std::min(LOCAL_SECUREC_MEM_MAX_LEN, dataLen - offset - copyCounts * LOCAL_SECUREC_MEM_MAX_LEN),
                   src + offsetMoveCount + copyCounts * srcMoveCount,
                   remainBytes);
    ASCEND_THROW_IF_NOT_FMT(err == ACL_SUCCESS, "memcpy (error %d)", err);
    offset += sizeBytes;
}
    
IVFSPCodeBookTrainer::IVFSPCodeBookTrainer(const IVFSPCodeBookTrainerInitParam &initParam)
{
    this->nlist = initParam.nlist;
    this->dim = initParam.dim;
    this->nonzeroNum = initParam.nonzeroNum;
    this->device = initParam.device;
    this->batchSize = initParam.batchSize;
    this->codeNum = initParam.codeNum;
    this->trainAndAdd = initParam.trainAndAdd;
    this->verbose = initParam.verbose;

    CheckParams();
    SetDeviceAndInitAscendResources();
    
    defaultStream = resources->getDefaultStream();
    randomNumberGenerator.seed(RANDOM_SEED);
    learnDataByNList.resize(nlist);

    codeBookOutputPath = initParam.codeBookOutputDir + "/codebook_" +
                                     std::to_string(dim) + "_" +
                                     std::to_string(nonzeroNum) + "_" +
                                     std::to_string(nlist) + ".bin";

    ASCEND_THROW_IF_NOT_MSG(ResetDistanceComputeQC() == APP_ERR_OK,
        "[IVFSPCodeBookTrainer] reset DistanceComputeQC failed.\n");
    ASCEND_THROW_IF_NOT_MSG(ResetMatmulComputeQC() == APP_ERR_OK,
        "[IVFSPCodeBookTrainer] reset MatmulComputeQC failed.\n");
    ASCEND_THROW_IF_NOT_MSG(ResetMatmulComputeCB() == APP_ERR_OK,
        "[IVFSPCodeBookTrainer] reset MatmulComputeCB failed.\n");
    ASCEND_THROW_IF_NOT_MSG(ResetCorrCompute() == APP_ERR_OK,
        "[IVFSPCodeBookTrainer] reset CorrCompute failed.\n");
}

IVFSPCodeBookTrainer::~IVFSPCodeBookTrainer()
{
    if (prevDevice != -1) {
        aclrtSetDevice(prevDevice);
    }
}

void IVFSPCodeBookTrainer::Train(int numIter)
{
    ASCEND_THROW_IF_NOT_FMT(numIter > 0 && numIter <= MAX_TRAIN_ITER,
        "numIter[%d] should be in (0, %d].\n", numIter, MAX_TRAIN_ITER);
    OrthCodeBook();
    KMeansUpdateCodeBook();
    for (int i = 0; i < numIter; ++i) {
        LogProgress("Iteration", i, numIter);
        ReshapeCodeBook();
        for (size_t j = 0; j < learnDataByBatch.size(); ++j) {
            LogProgress("Train Batch", static_cast<int>(j), static_cast<int>(learnDataByBatch.size()));
            CalNonzeroIdx(learnDataByBatch[j]);
        }
        UpdateCodeBookImpl();
        if (!trainAndAdd) { // 仅在不直接添加码本的场景下保存码本
            SaveCodeBook();
        }
    }
}

void IVFSPCodeBookTrainer::SaveCodeBook()
{
    FSPIOWriter codeBookWriter(codeBookOutputPath);
    std::vector<char> magicNumber = {'C', 'D', 'B', 'K'};
    std::vector<uint8_t> version = {1, 0, 0};
    const int blankDataSize = 64;
    std::vector<uint8_t> blankData(blankDataSize, 0);

    codeBookWriter.WriteAndCheck(magicNumber.data(), magicNumber.size() * sizeof(magicNumber[0]));
    codeBookWriter.WriteAndCheck(version.data(), version.size() * sizeof(version[0]));

    codeBookWriter.WriteAndCheck(&dim, sizeof(dim));
    codeBookWriter.WriteAndCheck(&nonzeroNum, sizeof(nonzeroNum));
    codeBookWriter.WriteAndCheck(&nlist, sizeof(nlist));

    codeBookWriter.WriteAndCheck(blankData.data(), blankData.size() * sizeof(blankData[0]));

    std::vector<float> codeBookFp32Output;
    Transpose(codeBookFp32, codeBookFp32Output, dim, nlist * nonzeroNum);
    codeBookWriter.WriteAndCheck(codeBookFp32Output.data(),
                                 codeBookFp32Output.size() * sizeof(codeBookFp32Output[0]));
}

void IVFSPCodeBookTrainer::ReadFile(const std::string &learnDataPath, float ratio)
{
    FSPIOReader learnDataReader(learnDataPath);
    size_t fileSize = learnDataReader.GetFileSize();
    ASCEND_THROW_IF_NOT_FMT(fileSize <= MAX_LEARN_DATA_SIZE,
        "learnData filesize[%zu] should be <= %llu.\n", fileSize, MAX_LEARN_DATA_SIZE);
    ASCEND_THROW_IF_NOT_FMT(fileSize % (dim * sizeof(float)) == 0,
        "learnData filesize[%zu] should be a multiple of 4 * dim[%d].\n", fileSize, dim);
    ASCEND_THROW_IF_NOT_FMT(ratio > 0.0 && ratio <= 1.0, "ratio[%.2f] should be in (0.0, 1.0].\n", ratio);
    nb = fileSize / (dim * sizeof(float));
    learnDataFp32.resize(static_cast<size_t>(nb) * dim);
    learnDataReader.ReadAndCheck(learnDataFp32.data(), fileSize);
    PreprocessLearnData(ratio);
}

void IVFSPCodeBookTrainer::ReadMemLearnData(const float *memLearnData, size_t memLearnDataSize, float ratio)
{
    ASCEND_THROW_IF_NOT_MSG(memLearnData != nullptr, "Pointer to in-memory learnData cannot a nullptr.\n");
    ASCEND_THROW_IF_NOT_FMT(memLearnDataSize * sizeof(float) <= MAX_LEARN_DATA_SIZE,
        "In-memory learnData data element size [%zu] should be <= %llu.\n",
        memLearnDataSize, MAX_LEARN_DATA_SIZE / sizeof(float));
    ASCEND_THROW_IF_NOT_FMT(memLearnDataSize % dim == 0,
        "learnData filesize[%zu] should be a multiple of dim[%d].\n", memLearnDataSize, dim);
    ASCEND_THROW_IF_NOT_FMT(ratio > 0.0 && ratio <= 1.0, "ratio[%.2f] should be in (0.0, 1.0].\n", ratio);
    nb = memLearnDataSize / dim;
    learnDataFp32.resize(memLearnDataSize);
    size_t zeroOffset = 0;
    CopyDataForLoad(learnDataFp32.data(), memLearnData,
        memLearnDataSize * sizeof(float), zeroOffset, memLearnDataSize * sizeof(float));
    PreprocessLearnData(ratio);
}

void IVFSPCodeBookTrainer::PreprocessLearnData(float ratio)
{
    // 此时learnDataFp32内存放完整的float32数据，用这些数据初始化码本
    InitCodeBook();
    SampleLearnData(ratio);
    SplitLearnDataByBatch();
    LogInfo();
}

std::vector<float> IVFSPCodeBookTrainer::GetCodeBook()
{
    std::vector<float> codeBookFp32Transposed;
    Transpose(codeBookFp32, codeBookFp32Transposed, dim, nlist * nonzeroNum);
    return codeBookFp32Transposed;
}

void IVFSPCodeBookTrainer::OrthCodeBook()
{
    std::vector<float> U(dim * dim);
    for (size_t i = 0; i < codeBookByNList.size(); ++i) {
        LogProgress("Orth", static_cast<int>(i), static_cast<int>(codeBookByNList.size()));
        int ret = SingleFloatSVD(dim, nonzeroNum, codeBookByNList[i], U, 'O');
        if (ret != 0) {
            APP_LOG_ERROR("CodeBookTrainer Error: Convergence failed, skipping current nlist.");
        }
    }
    MergeCodeBookByNList();
}

void IVFSPCodeBookTrainer::ReshapeCodeBook()
{
    std::vector<float> codeBookFp32Transposed;
    Transpose(codeBookFp32, codeBookFp32Transposed, dim, nonzeroNum * nlist);

    std::vector<float> codeBookFp32Reshaped;
    ZzFormatReshape(codeBookFp32Transposed, codeBookFp32Reshaped, nonzeroNum * nlist, dim);

    Fp32ToFp16(codeBookFp32Reshaped, codeBookReshaped);
}

void IVFSPCodeBookTrainer::CalNonzeroIdx(std::vector<float16_t> &learnDataPerBatch)
{
    auto &mem = resources->getMemoryManager();
    int ret = 0;

    size_t actualBatchSize = learnDataPerBatch.size() / dim;
    if (actualBatchSize < static_cast<size_t>(batchSize)) {
        learnDataPerBatch.resize(static_cast<size_t>(batchSize) * dim, 0);
    }

    /* 输入准备 */
    auto learnDataPerBatchNPU = CreateTensorAndAllocDeviceMem(mem, { batchSize, dim }, learnDataPerBatch.data());
    auto codeBookReshapedNPU = CreateTensorAndAllocDeviceMem(
        mem,
        {utils::divUp(nonzeroNum * nlist, CUBE_ALIGN), utils::divUp(dim, CUBE_ALIGN), CUBE_ALIGN, CUBE_ALIGN },
        codeBookReshaped.data()
    );
    std::vector<float16_t> nlistOfZeros(nlist, 0);
    auto nlistOfZerosNPU = CreateTensorAndAllocDeviceMem(mem, { nlist }, nlistOfZeros.data());

    /* 输出准备 */
    AscendTensor<float16_t, DIMS_2> distNPU(mem, { batchSize, nlist }, defaultStream);
    AscendTensor<uint16_t, DIMS_2> flagNPU(mem, { CORE_NUM, FLAG_SIZE }, defaultStream);

    RunDistanceComputeQC(learnDataPerBatchNPU, codeBookReshapedNPU, nlistOfZerosNPU, distNPU, flagNPU);
    ret = aclrtSynchronizeStream(defaultStream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtSynchronizeStream aicore stream failed, error = %d.\n", ret);

    /* distNPU内容拷贝给dist */
    std::vector<float16_t> distFromEachNList(batchSize * nlist);
    ret = aclrtMemcpy(distFromEachNList.data(), distFromEachNList.size() * sizeof(float16_t),
                      distNPU.data(), distNPU.getSizeInBytes(), ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy failed, error = %d.\n", ret);
    distFromEachNList.resize(actualBatchSize * nlist);

    std::vector<float16_t> labels(actualBatchSize);
    ArgMaxAlongNList(distFromEachNList, labels);

    /* 按照分桶id将learnData放入对应的桶内，为之后的以桶为单位update_codebook做准备 */
    PutDataInNList(learnDataPerBatch, labels);

    // 还原learnDataPerBatch大小
    if (actualBatchSize < static_cast<size_t>(batchSize)) {
        learnDataPerBatch.resize(actualBatchSize * dim);
    }
}

void IVFSPCodeBookTrainer::Fp32ToFp16(const std::vector<float> &src, std::vector<float16_t> &dst)
{
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](float value) -> float16_t {
        FormatConvert convert;
        convert.value = faiss::ascendSearch::fp16(value).data;
        return convert.data;
    });
}

void IVFSPCodeBookTrainer::Fp16ToFp32(const std::vector<float16_t> &src, std::vector<float> &dst)
{
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](float16_t value) -> float {
        return static_cast<float>(faiss::ascendSearch::fp16(value));
    });
}

void IVFSPCodeBookTrainer::InitCodeBook()
{
    ASCEND_THROW_IF_NOT_FMT(nb >= nlist * nonzeroNum,
        "nb[%d] should be >= nlist[%d] * nonzeroNum[%d].\n", nb, nlist, nonzeroNum);
    std::vector<size_t> chooseFrom;
    chooseFrom.resize(static_cast<size_t>(nb), 0);
    std::iota(chooseFrom.begin(), chooseFrom.end(), 0);

    // shuffle选择的数组，将前nlist * nonzeroNum个插入codeBook
    std::shuffle(chooseFrom.begin(), chooseFrom.end(), randomNumberGenerator);

    std::vector<float> codeBookBeforeTranspose;
    codeBookBeforeTranspose.resize(static_cast<size_t>(nlist) * nonzeroNum * dim);
    for (int i = 0; i < nlist * nonzeroNum; ++i) {
        std::copy(learnDataFp32.begin() + chooseFrom[i] * dim,
                  learnDataFp32.begin() + (chooseFrom[i] + 1) * dim,
                  codeBookBeforeTranspose.begin() + i * dim);
    }

    Transpose(codeBookBeforeTranspose, codeBookFp32, nlist * nonzeroNum, dim);
    SplitCodeBookByNList();
}

void IVFSPCodeBookTrainer::SplitCodeBookByNList()
{
    // 将获得的codeBook按照nlist拆分入不同的桶内
    /*
    codeBookFp32内此时排布:
    | k_1 * nzNum | k_2 * nzNum | k_3 * nzNum ||k_1 * nzNum | k_2 * nzNum | k_3 * nzNum |
    |_____________|_____________|_____________||____________|_____________|_____________|...
                       dim_1                                      dim2
    */

    codeBookByNList.resize(static_cast<size_t>(nlist));
    for (size_t i = 0; i < static_cast<size_t>(nlist); ++i) {
        codeBookByNList[i].resize(static_cast<size_t>(dim) * nonzeroNum);
        for (size_t j = 0; j < static_cast<size_t>(dim); ++j) {
            std::copy(codeBookFp32.begin() + j * nlist * nonzeroNum + i * nonzeroNum,
                      codeBookFp32.begin() + j * nlist * nonzeroNum + (i + 1) * nonzeroNum,
                      codeBookByNList[i].begin() + j * nonzeroNum);
        }
    }
}

void IVFSPCodeBookTrainer::MergeCodeBookByNList()
{
    codeBookFp32.resize(static_cast<size_t>(dim) * nlist * nonzeroNum);
    for (size_t i = 0; i < static_cast<size_t>(dim); ++i) {
        for (size_t j = 0; j < static_cast<size_t>(nlist); ++j) {
            std::copy(codeBookByNList[j].begin() + i * nonzeroNum,
                      codeBookByNList[j].begin() + (i + 1) * nonzeroNum,
                      codeBookFp32.begin() + i * nlist * nonzeroNum + j * nonzeroNum);
        }
    }
}

void IVFSPCodeBookTrainer::ArgMaxAlongNList(const std::vector<float16_t> &distFromEachNList,
                                            std::vector<float16_t> &labels)
{
    for (size_t i = 0; i < labels.size(); ++i) {
        auto maxIter = std::max_element(distFromEachNList.begin() + i * nlist,
                                        distFromEachNList.begin() + (i + 1) * nlist);
        labels[i] = std::distance(distFromEachNList.begin() + i * nlist, maxIter);
    }
}

template <typename T>
void IVFSPCodeBookTrainer::PutDataInNList(const std::vector<float16_t> &learnDataPerBatch,
                                          const std::vector<T> &labels)
{
    for (size_t i = 0; i < labels.size(); ++i) {
        int nlistId = static_cast<int>(labels[i]);
        ASCEND_THROW_IF_NOT_FMT(nlistId <= nlist, "labels[%d] should be < nlist[%d].\n", nlistId, nlist);

        /* 一次只用最多codeNum个底库向量进行SVD分解和码本更新，因此只用保证每个桶内最多 codeNum个底库向量 */
        if (learnDataByNList[nlistId].size() < static_cast<size_t>(codeNum) * dim) {
            learnDataByNList[nlistId].insert(learnDataByNList[nlistId].end(),
                                             learnDataPerBatch.begin() + i * dim,
                                             learnDataPerBatch.begin() + (i + 1) * dim);
        }
    }
}

/* 将底库数据按桶拆分后，我们需要确保每个桶内的底库数据大小都是 codeNum * dim的，以符合算子输入需求 */
void IVFSPCodeBookTrainer::PadCurrNListDataWithZeros(std::vector<float16_t> &learnDataPerNList)
{
    if (learnDataPerNList.size() < static_cast<size_t>(codeNum) * dim) {
        learnDataPerNList.resize(static_cast<size_t>(codeNum) * dim, 0);
    }
}

void IVFSPCodeBookTrainer::UpdateCodeBookImpl()
{
    for (int k = 0; k < nlist; ++k) {
        LogProgress("Update Codebook", k, nlist);
        PadCurrNListDataWithZeros(learnDataByNList[k]); // 为当前nlist内的数据补0用于算子输入
        UpdateCodeBookAcl(learnDataByNList[k], k);
        std::vector<float16_t>().swap(learnDataByNList[k]); // 该nlist处理完后，清空按桶分布的底库数据并清空内存
    }
    MergeCodeBookByNList();
}

void IVFSPCodeBookTrainer::UpdateCodeBookAcl(const std::vector<float16_t> &learnDataPerNList, int nlistId)
{
    if (learnDataPerNList.size() == 0) {
        return;
    }

    auto &mem = resources->getMemoryManager();
    int ret = 0;

    std::vector<float16_t> learnDataPerNListNz;
    NzFormatReshape(learnDataPerNList, learnDataPerNListNz, codeNum, dim);

    /* 数据准备 */
    auto learnDataPerNListNzNPU = CreateTensorAndAllocDeviceMem(
        mem,
        { 1, utils::divUp(codeNum, CUBE_ALIGN), dim, CUBE_ALIGN },
        learnDataPerNListNz.data()
    );
    std::vector<uint64_t> actualNum(CORE_NUM, 0);
    actualNum[0] = codeNum;
    auto actualNumNPU = CreateTensorAndAllocDeviceMem(mem, { CORE_NUM }, actualNum.data());
    AscendTensor<float16_t, DIMS_2> svdInputNPU(mem, { dim, dim }, defaultStream);
    AscendTensor<uint16_t, DIMS_2> flagNPU(mem, { CORE_NUM, FLAG_SIZE }, defaultStream);

    RunCorrCompute(learnDataPerNListNzNPU, actualNumNPU, svdInputNPU, flagNPU);
    ret = aclrtSynchronizeStream(defaultStream);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtSynchronizeStream aicore stream failed: error = %d.\n", ret);

    std::vector<float16_t> svdInput(dim * dim);
    ret = aclrtMemcpy(svdInput.data(), svdInput.size() * sizeof(float16_t),
                      svdInputNPU.data(), svdInputNPU.getSizeInBytes(),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy failed; error = %d.\n", ret);

    /* 将svdInput转换为fp32 */
    std::vector<float> svdInputFp32(dim * dim);
    Fp16ToFp32(svdInput, svdInputFp32);
    std::vector<float> tempU(dim * dim);
    ret = SingleFloatSVD(dim, dim, svdInputFp32, tempU, 'O');
    if (ret != 0) {
        APP_LOG_ERROR("CodeBookTrainer Error: Convergence failed, skipping current nlist.");
    } else {
        int updateSize = std::min(nonzeroNum, dim);
        CopyFirstNColumns(svdInputFp32, codeBookByNList[nlistId], dim, nonzeroNum, updateSize);
    }
}

/**
 * @brief 单精度浮点(float32)SVD计算，使用LAPACKE_sgesdd；为了方便理解，参数名与LAPACKE库定义一致
 *
 * @param m A矩阵行数
 * @param n A矩阵列数
 * @param A 输入矩阵A
 * @param U 输出左矩阵U
 * @param jobu 对左矩阵处理方式；可能值为'O'或'S',功能见LAPACKE ?gesdd文档
 */
int IVFSPCodeBookTrainer::SingleFloatSVD(int m, int n, std::vector<float> &A, std::vector<float> &U, char jobu)
{
    int lda = std::max(1, n); // at least max(1, n) for row major layout
    int ldu = std::min(m, n); // ldu >= min(m, n) for row major layout
    size_t sVecSize = static_cast<size_t>(std::max(1, ldu));
    int ldvt = ldu; // 不适用VT输出，此处设置为ldu仅作参考
    std::vector<float> S(sVecSize);
    std::vector<float> VT; // VT使用'N'选项，该内存不会被使用
    int ret = LAPACKE_sgesdd(LAPACK_ROW_MAJOR, jobu, m, n, A.data(), lda,
                             S.data(), U.data(), ldu, VT.data(), ldvt);
    return ret;
}

/**
 * @brief 作为Python纵向切片的替代函数；将2维数据src的前copyColNum列拷贝进入dst的前copyColNum列
 *
 * @param src 输入矩阵
 * @param dst 输出矩阵
 * @param srcCol 输入矩阵列数
 * @param dstCol 输出矩阵列数
 * @param copyColNum 待拷贝的列数 （外部输入需确保 copyColNum <= srcCol 和 dstCol）
 */
void IVFSPCodeBookTrainer::CopyFirstNColumns(const std::vector<float> &src, std::vector<float> &dst,
                                             int srcCol, int dstCol, int copyColNum)
{
    size_t srcRow = src.size() / srcCol;
#pragma omp parallel for
    for (size_t i = 0; i < srcRow; ++i) {
        std::copy(src.begin() + i * srcCol, src.begin() + i * srcCol + copyColNum, dst.begin() + i * dstCol);
    }
}

template <typename T, int Dim>
AscendTensor<T, Dim> IVFSPCodeBookTrainer::CreateTensorAndAllocDeviceMem(AscendMemory &mem,
                                                                         const int (&sizes)[Dim],
                                                                         T *cpuDataPtr)
{
    AscendTensor<T, Dim> npuTensor(mem, sizes, defaultStream);
    int ret = aclrtMemcpy(npuTensor.data(), npuTensor.getSizeInBytes(),
                          cpuDataPtr, npuTensor.getSizeInBytes(),
                          ACL_MEMCPY_HOST_TO_DEVICE);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "aclrtMemcpy failed; error = %d.\n", ret);
    return npuTensor;
}

void IVFSPCodeBookTrainer::KMeansUpdateCodeBook()
{
    if (verbose) {
        printf("Initiating codebook; this may take a moment...\n");
    }
    PerformKMeansOnLearnData();
    UpdateCodeBookImpl();
}

void IVFSPCodeBookTrainer::PerformKMeansOnLearnData()
{
    faiss::Clustering clus(dim, nlist);
    clus.niter = KMEANS_ITER;
    faiss::IndexFlatL2 index(dim);
    clus.train(nb, learnDataFp32.data(), index);
    std::vector<int64_t> labels(nb);
    std::vector<float> distances(nb);
    index.search(nb, learnDataFp32.data(), 1, distances.data(), labels.data());

    // 清理learnDatFp32，避免内存占用过大
    std::vector<float>().swap(learnDataFp32);

    // 我们在SplitLearnDataByBatch阶段已经给learnDataFp16赋值，因此此时直接使用，将其分桶
    PutDataInNList(learnDataFp16, labels);
}

void IVFSPCodeBookTrainer::SplitLearnDataByBatch()
{
    // 将全量的fp32训练数据转成fp16
    Fp32ToFp16(learnDataFp32, learnDataFp16);

    // 将数据切片放入learnDataByBatch内
    int numBatch = (nb + (batchSize - 1)) / batchSize;
    learnDataByBatch.resize(static_cast<size_t>(numBatch));
    for (size_t i = 0; i < learnDataByBatch.size(); ++i) {
        size_t actualBatchSize = std::min(static_cast<size_t>(batchSize), static_cast<size_t>(nb) - i * batchSize);
        learnDataByBatch[i].resize(actualBatchSize * dim);
        std::copy(learnDataFp16.begin() + i * batchSize * dim,
                  learnDataFp16.begin() + i * batchSize * dim + actualBatchSize * dim,
                  learnDataByBatch[i].begin());
    }
}

template <typename T>
void IVFSPCodeBookTrainer::Transpose(const std::vector<T> &src, std::vector<T> &dst,
                                     int srcRow, int srcCol)
{
    ASCEND_THROW_IF_NOT_FMT(src.size() == static_cast<size_t>(srcRow) * srcCol,
        "src size[%zu] != row[%d] * col[%d].\n", src.size(), srcRow, srcCol);
    dst.resize(src.size());
#pragma omp parallel for collapse(2)
    for (int i = 0; i < srcRow; ++i) {
        for (int j = 0; j < srcCol; ++j) {
            dst[j * srcRow + i] = src[i * srcCol + j];
        }
    }
}

/**
 * @brief 对一个ND格式排布的矩阵进行Zz格式分形，用于输入算子
 *
 * @tparam T 数据类型
 * @param src 输入的ND排布原始矩阵(shape = row * col)
 * @param dst 输出的Zz格式的矩阵，其内部的数据排布使用4维张量可以表示为(shape = row // 16, col // 16, 16, 16)
 * @param row 输入矩阵的行数
 * @param col 输入矩阵的列数
 */
template <typename T>
void IVFSPCodeBookTrainer::ZzFormatReshape(const std::vector<T> &src, std::vector<T> &dst, int row, int col)
{
    ASCEND_THROW_IF_NOT_FMT(src.size() == static_cast<size_t>(row) * col,
        "src size[%zu] != row[%d] * col[%d].\n", src.size(), row, col);
    int rowMovCnt = utils::divUp(row, CUBE_ALIGN);
    int colMovCnt = utils::divUp(col, CUBE_ALIGN);
    dst.resize(static_cast<size_t>(rowMovCnt) * colMovCnt * CUBE_ALIGN * CUBE_ALIGN, 0);
#pragma omp parallel for
    for (int i = 0; i < rowMovCnt; ++i) {
        for (int k = 0; k < CUBE_ALIGN; ++k) {
            for (int j = 0; j < colMovCnt; ++j) {
                int ret = memcpy_s(dst.data() + i * CUBE_ALIGN * col + k * CUBE_ALIGN + j * CUBE_ALIGN * CUBE_ALIGN,
                                   CUBE_ALIGN * sizeof(T),
                                   src.data() + i * CUBE_ALIGN * col + k * col + j * CUBE_ALIGN,
                                   CUBE_ALIGN * sizeof(T));
                ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "ZzFormatReshape memory copy failed, err = %d.\n", ret);
            }
        }
    }
}

/* 注意：该函数与ZzFormatReshape不同，后者期待的src矩阵为已经转置后的矩阵，该函数期待的src矩阵是原始矩阵 */
template <typename T>
void IVFSPCodeBookTrainer::NzFormatReshape(const std::vector<T> &src, std::vector<T> &dst, int row, int col)
{
    ASCEND_THROW_IF_NOT_FMT(src.size() == static_cast<size_t>(row) * col,
        "src size[%zu] != row[%d] * col[%d].\n", src.size(), row, col);
    int rowMovCnt = utils::divUp(row, CUBE_ALIGN);
    int colMovCnt = col;
    int rowRemain = row - row / CUBE_ALIGN * CUBE_ALIGN;
    dst.resize(static_cast<size_t>(rowMovCnt) * CUBE_ALIGN * col, 0);
#pragma omp parallel for
    for (int i = 0; i < rowMovCnt; ++i) {
        for (int k = 0; k < colMovCnt; ++k) {
            auto loopJ = (i == rowMovCnt -1 && rowRemain != 0) ? rowRemain : CUBE_ALIGN;
            for (int j = 0; j < loopJ; ++j) {
                auto ret = memcpy_s(dst.data() + i * CUBE_ALIGN * col + k * CUBE_ALIGN + j, sizeof(T),
                                    src.data() + i * CUBE_ALIGN * col + j * col + k, sizeof(T));
                ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "NzFormatReshape memory copy failed, err = %d.\n", ret);
            }
        }
    }
}

void IVFSPCodeBookTrainer::RunDistanceComputeQC(AscendTensor<float16_t, DIMS_2> &query,
                                                AscendTensor<float16_t, DIMS_4> &codebookFormatted,
                                                AscendTensor<float16_t, DIMS_1> &bucket,
                                                AscendTensor<float16_t, DIMS_2> &dist,
                                                AscendTensor<uint16_t, DIMS_2> &flag)
{
    APP_LOG_INFO("[IVFSPCodeBookTrainer] RunDistanceComputeQC operation started.\n");
    ASCEND_THROW_IF_NOT(distQCOp);

    std::vector<const aclDataBuffer *> input;
    input.emplace_back(aclCreateDataBuffer(query.data(), query.getSizeInBytes()));
    input.emplace_back(aclCreateDataBuffer(codebookFormatted.data(), codebookFormatted.getSizeInBytes()));
    input.emplace_back(aclCreateDataBuffer(bucket.data(), bucket.getSizeInBytes()));

    std::vector<aclDataBuffer *> output;
    output.emplace_back(aclCreateDataBuffer(dist.data(), dist.getSizeInBytes()));
    output.emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));
    distQCOp->exec(input, output, defaultStream);
    for (auto &item : input) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    input.clear();
    for (auto &item : output) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    output.clear();
}

void IVFSPCodeBookTrainer::RunCorrCompute(AscendTensor<float16_t, DIMS_4> &queryFormattedShape,
                                          AscendTensor<uint64_t, DIMS_1> &actualNumShape,
                                          AscendTensor<float16_t, DIMS_2> &svdInputShape,
                                          AscendTensor<uint16_t, DIMS_2> &flag)
{
    APP_LOG_INFO("[IVFSPCodeBookTrainer] RunMatmulComputeCB operation started.\n");
    ASCEND_THROW_IF_NOT(corrOp);

    std::vector<const aclDataBuffer *> input;
    input.emplace_back(aclCreateDataBuffer(queryFormattedShape.data(), queryFormattedShape.getSizeInBytes()));
    input.emplace_back(aclCreateDataBuffer(actualNumShape.data(), actualNumShape.getSizeInBytes()));
    
    std::vector<aclDataBuffer *> output;
    output.emplace_back(aclCreateDataBuffer(svdInputShape.data(), svdInputShape.getSizeInBytes()));
    output.emplace_back(aclCreateDataBuffer(flag.data(), flag.getSizeInBytes()));

    corrOp->exec(input, output, defaultStream);

    for (auto &item : input) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    input.clear();
    for (auto &item : output) {
        ACL_REQUIRE_OK(aclDestroyDataBuffer(item));
    }
    output.clear();
}

APP_ERROR IVFSPCodeBookTrainer::ResetDistanceComputeQC()
{
    AscendOpDesc desc("DistanceComputeQC");
    APP_LOG_INFO("[IVFSPCodeBookTrainer] Resetting DistanceComputeQC...\n");
    std::vector<int64_t> queryShape({ batchSize, dim });
    std::vector<int64_t> codeBookFormattedShape({ utils::divUp(nonzeroNum * nlist, CUBE_ALIGN),
                                     utils::divUp(dim, CUBE_ALIGN),
                                     CUBE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> bucketShape({ nlist });
    std::vector<int64_t> distShape({ batchSize, nlist });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

    desc.addInputTensorDesc(ACL_FLOAT16, queryShape.size(), queryShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, codeBookFormattedShape.size(), codeBookFormattedShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, bucketShape.size(), bucketShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, distShape.size(), distShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    distQCOp.reset();
    distQCOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
    APPERR_RETURN_IF_NOT_LOG(distQCOp->init(), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
        "[IVFSPCodeBookTrainer] DistanceComputeQC op init failed!\n");
    return APP_ERR_OK;
}

APP_ERROR IVFSPCodeBookTrainer::ResetMatmulComputeQC()
{
    AscendOpDesc desc("MatmulCompute");
    APP_LOG_INFO("[IVFSPCodeBookTrainer] Resetting MatmulCompute(QC)...\n");
    std::vector<int64_t> codeBookShape({ nonzeroNum, dim });
    std::vector<int64_t> queryFormattedShape({ utils::divUp(codeNum, CUBE_ALIGN),
                                     utils::divUp(dim, CUBE_ALIGN),
                                     CUBE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> actualNumsShape({ CORE_NUM, CORE_NUM }); // 对应SP.py脚本中的actual_nums变量
    std::vector<int64_t> querySubdimShape({ nonzeroNum, codeNum });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

    desc.addInputTensorDesc(ACL_FLOAT16, codeBookShape.size(), codeBookShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, queryFormattedShape.size(), queryFormattedShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, actualNumsShape.size(), actualNumsShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, querySubdimShape.size(), querySubdimShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    matMulQCOp.reset();
    matMulQCOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
    APPERR_RETURN_IF_NOT_LOG(matMulQCOp->init(), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
        "[IVFSPCodeBookTrainer] MatmulCompute(QC) op init failed!\n");
    return APP_ERR_OK;
}

APP_ERROR IVFSPCodeBookTrainer::ResetMatmulComputeCB()
{
    AscendOpDesc desc("MatmulCompute");
    APP_LOG_INFO("[IVFSPCodeBookTrainer] Resetting MatmulCompute(CB)...\n");
    std::vector<int64_t> codeBookShape({ dim, nonzeroNum });
    std::vector<int64_t> querySubdimFormattedShape({ utils::divUp(codeNum, CUBE_ALIGN),
                                     utils::divUp(nonzeroNum, CUBE_ALIGN),
                                     CUBE_ALIGN, CUBE_ALIGN });
    std::vector<int64_t> actualNumsShape({ CORE_NUM, CORE_NUM }); // 对应SP.py脚本中的actual_nums变量
    std::vector<int64_t> queryRestoredShape({ dim, codeNum }); // 还原出的query, shape为: dim * queryNum
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

    desc.addInputTensorDesc(ACL_FLOAT16, codeBookShape.size(), codeBookShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_FLOAT16, querySubdimFormattedShape.size(),
                            querySubdimFormattedShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT32, actualNumsShape.size(), actualNumsShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, queryRestoredShape.size(), queryRestoredShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    matMulCBOp.reset();
    matMulCBOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
    APPERR_RETURN_IF_NOT_LOG(matMulCBOp->init(), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
        "[IVFSPCodeBookTrainer] MatmulCompute(CB) op init failed!\n");
    return APP_ERR_OK;
}

APP_ERROR IVFSPCodeBookTrainer::ResetCorrCompute()
{
    AscendOpDesc desc("CorrCompute");
    APP_LOG_INFO("[IVFSPCodeBookTrainer] Resetting CorrCompute...\n");
    std::vector<int64_t> queryFormattedShape({ 1, utils::divUp(codeNum, CUBE_ALIGN), dim, CUBE_ALIGN });
    std::vector<int64_t> actualNumShape({ CORE_NUM }); // 对应SP.py脚本中的actual_num变量
    std::vector<int64_t> SVDInputShape({ dim, dim });
    std::vector<int64_t> flagShape({ CORE_NUM, FLAG_SIZE });

    desc.addInputTensorDesc(ACL_FLOAT16, queryFormattedShape.size(), queryFormattedShape.data(), ACL_FORMAT_ND);
    desc.addInputTensorDesc(ACL_UINT64, actualNumShape.size(),
                            actualNumShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_FLOAT16, SVDInputShape.size(), SVDInputShape.data(), ACL_FORMAT_ND);
    desc.addOutputTensorDesc(ACL_UINT16, flagShape.size(), flagShape.data(), ACL_FORMAT_ND);

    corrOp.reset();
    corrOp = CREATE_UNIQUE_PTR(AscendOperator, desc);
    APPERR_RETURN_IF_NOT_LOG(corrOp->init(), APP_ERR_ACL_OP_LOAD_MODEL_FAILED,
        "[IVFSPCodeBookTrainer] CorrCompute op init failed!\n");
    return APP_ERR_OK;
}

void IVFSPCodeBookTrainer::CheckParams() const
{
    uint32_t devCount = faiss::ascendSearch::SocUtils::GetInstance().GetDeviceCount();
    ASCEND_THROW_IF_NOT_FMT(static_cast<uint32_t>(device) < devCount && device >= 0,
        "Device %d is invalid, total device %u.\n", device, devCount);

    ASCEND_THROW_IF_NOT_FMT(batchSize > 0 && batchSize <= BATCH_SIZE_LIMIT && batchSize % CUBE_ALIGN == 0,
        "batchSize[%d] should be > 0 and be a multiple of 16.\n", batchSize);
    ASCEND_THROW_IF_NOT_FMT(codeNum > 0 && codeNum <= BATCH_SIZE_LIMIT && codeNum % CUBE_ALIGN == 0,
        "codeNum[%d] should be > 0 and be a multiple of 16.\n", codeNum);

    ASCEND_THROW_IF_NOT_FMT(std::find(DIMS.begin(), DIMS.end(), dim) != DIMS.end(), "Unsupported dims %d.\n", dim);

    ASCEND_THROW_IF_NOT_FMT(std::find(NLISTS.begin(), NLISTS.end(), nlist) != NLISTS.end(),
        "Unsupported nlists %d.\n", nlist);
    if (std::find(HIGH_DIMS.begin(), HIGH_DIMS.end(), dim) != HIGH_DIMS.end()) {
        ASCEND_THROW_IF_NOT_FMT(nlist <= HIGH_DIMS_NLIST_LIMIT,
            "nlist should be <= %d if dims >= %d; yours is %d",
            HIGH_DIMS_NLIST_LIMIT, HIGH_DIMS[0], this->nlist);
    }

    ASCEND_THROW_IF_NOT_FMT(nonzeroNum > 0 && nonzeroNum <= dim &&
        nonzeroNum <= MAX_NONZERO_NUM && nonzeroNum % CUBE_ALIGN == 0,
        "Unsupported nonzeroNum %d.\n", nonzeroNum);
}

void IVFSPCodeBookTrainer::SampleLearnData(float ratio)
{
    if (ratio < 1.0) {
        nb = static_cast<int>(nb * ratio);
        std::shuffle(learnDataFp32.begin(), learnDataFp32.end(), randomNumberGenerator);
        learnDataFp32.resize(static_cast<size_t>(nb) * dim);
    }
}

void IVFSPCodeBookTrainer::SetDeviceAndInitAscendResources()
{
    int currentDevice = 0;
    int ret = 0;

    ret = aclrtGetDevice(&currentDevice);
    ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Get device id failed, error = %d.\n", ret);
    if (currentDevice != device) {
        ret = aclrtSetDevice(device);
        ASCEND_THROW_IF_NOT_FMT(ret == ACL_SUCCESS, "Set current device to %d failed, error = %d.\n", device, ret);
        prevDevice = currentDevice; // 此处设置prevDevice以确保接口独立性 (若抛异常，stack unwinding时将deviceId设置成之前的值)
    }
    resources = std::make_shared<AscendResourcesProxy>();
}

} // namespace ascendSearch