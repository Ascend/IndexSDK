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
// 需要生成 aicpu算子+ivfsqt算子(-d 256 -c 16384)+flat_at算子(-d 256 -c 16384)

#include <faiss/ascend/custom/AscendIndexIVFSQT.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <unordered_map>
#include <vector>

namespace
{
using recallMap = std::unordered_map<int, float>;
const int RECMAP_KEY_1 = 1;
const int RECMAP_KEY_10 = 10;
const int RECMAP_KEY_100 = 100;
const int MILLI_SECOND = 1000;
const int BITS = 8;
const int SEED = 1;
const int NPROBE = 64;
const int L2NPROBE = 360;
const int L3NPROBE = 360;
const int DIM_IN = 256;
const int DIM_OUT = 64;
void computeRecall(recallMap &recMap, int j)
{
    recMap[RECMAP_KEY_100]++;
    switch (j)
    {
        case 0:
            recMap[RECMAP_KEY_1]++;
            recMap[RECMAP_KEY_10]++;
            break;
        case 1 ... 9:  // case 1到9
            recMap[RECMAP_KEY_10]++;
            break;
        default:
            break;
    }
}

template <class T>
recallMap calRecallNomal(std::vector<T> label, int64_t *gt, int shape)
{
    recallMap recMap;
    recMap[RECMAP_KEY_1] = 0;
    recMap[RECMAP_KEY_10] = 0;
    recMap[RECMAP_KEY_100] = 0;
    if (shape <= 0)
    {
        std::cerr << "Error: Invalid shape value." << std::endl;
        return recMap;
    }
    int k = label.size() / shape;
    for (int i = 0; i < shape; i++)
    {
        for (int j = 0; j < k; j++)
        {
            if (gt[i * k] == label[i * k + j])
            {
                computeRecall(recMap, j);
                break;
            }
        }
    }
    recMap[RECMAP_KEY_1] = recMap[RECMAP_KEY_1] / shape * 100;  // recMap[1]的百分比 这里的100代表的是百分比的计算因子
    recMap[RECMAP_KEY_10] =
        recMap[RECMAP_KEY_10] / shape * 100;  // recMap[10]的百分比 这里的100代表的是百分比的计算因子
    recMap[RECMAP_KEY_100] =
        recMap[RECMAP_KEY_100] / shape * 100;  // recMap[100]的百分比 这里的100代表的是百分比的计算因子
    return recMap;
}

inline double GetMillisecs()
{
    struct timeval tv = {0, 0};
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}

std::independent_bits_engine<std::mt19937, BITS, uint8_t> engine(SEED);

int8_t RandomInt8()
{
    int offset = 128;
    int8_t i = engine() - offset;
    return i;
}

void GeneratorRandomIndex(size_t ntotal, size_t &idx)
{
    static thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_int_distribution<size_t> dis(0, ntotal - 1);
    idx = dis(generator);
}
struct dataFloat
{
    dataFloat(std::vector<float> &base, std::vector<float> &learn, std::vector<float> &query, std::vector<int64_t> &gt)
        : base(base), learn(learn), query(query), gt(gt) {};
    std::vector<float> &base;
    std::vector<float> &learn;
    std::vector<float> &query;
    std::vector<int64_t> &gt;
};

void DataProccess(int searchNum, int topk, dataFloat data, faiss::ascend::AscendIndexIVFSQT &index,
                  std::vector<faiss::idx_t> &labelTopk)
{
    int flatK = 1;
    double searchStart = GetMillisecs();

    std::vector<faiss::idx_t> labelFlat(searchNum * static_cast<size_t>(topk), 0);
    std::vector<float> distanceTopk(searchNum * static_cast<size_t>(topk), 0);
    std::vector<faiss::idx_t> labelRet(searchNum * static_cast<size_t>(topk), 0);
    std::vector<float> distanceFlat(searchNum * static_cast<size_t>(flatK), 0);

    int cpuNum = 48;
    auto cpuSearchFunctor = [searchNum, topk, flatK, &distanceTopk, &labelTopk, &distanceFlat, &labelRet, &data](
                                int cpuIdx, int start, int end)
    {
        for (int qId = start; qId < std::min(end, searchNum); qId++)
        {
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(cpuIdx, &cpuset);
            float max = 0;
            size_t offset = 0;
            for (size_t topkId = 0; topkId < static_cast<size_t>(topk); topkId++)
            {
                bool isValid = false;
                if (labelTopk[qId * topk + topkId] == -1)
                {
                    continue;
                }
                for (size_t dimId = 0; dimId < static_cast<size_t>(DIM_IN); dimId++)
                {
                    // 计算IP距离
                    distanceTopk[qId * topk + topkId] +=
                        data.query[qId * DIM_IN + dimId] * data.base[labelTopk[qId * topk + topkId] * DIM_IN + dimId];
                    isValid = true;
                }
                if (isValid && max < distanceTopk[qId * topk + topkId])
                {
                    max = distanceTopk[qId * topk + topkId];
                    offset = topkId;
                }
                labelRet[qId * topk] = labelTopk[qId * topk + offset];
                distanceFlat[qId * flatK] = distanceTopk[qId * topk + offset];
            }
        }
    };
    std::thread threads[cpuNum];
    int block = (searchNum + cpuNum - 1) / cpuNum;  // div up
    for (int i = 0; i < cpuNum; i++)
    {
        threads[i] = std::thread(cpuSearchFunctor, i, i * block, (i + 1) * block);
    }
    for (int i = 0; i < cpuNum; i++)
    {
        threads[i].join();
    }
    double cpuSearchEnd = GetMillisecs();
    recallMap recall = calRecallNomal(labelRet, data.gt.data(), searchNum);
    // QPS = 并发线程数 * (1000 / 平均耗时ms)
    printf("searchNum = %d, r1 = %.2f, r10 = %.2f, r100 = %.2f, qps = %lf\n", searchNum, recall[RECMAP_KEY_1],
           recall[RECMAP_KEY_10], recall[RECMAP_KEY_100], MILLI_SECOND * searchNum / (cpuSearchEnd - searchStart));
}

void SearchProccess(faiss::ascend::AscendIndexIVFSQT &index, size_t ntotal, std::vector<float> &base,
                    std::vector<float> &learn, dataFloat dataBaseFloat)
{
    int fuzzyK = 3;
    float threshold = 1.6;
    int trainSize = learn.size() / DIM_IN;
    int topk = 100;
    int searchNum = 10240;
    index.verbose = true;
    index.setFuzzyK(fuzzyK);
    index.setThreshold(threshold);

    double trainStart = GetMillisecs();
    index.train(trainSize, learn.data());
    double trainEnd = GetMillisecs();
    // 将毫秒转换为秒，故 / 1000
    printf("train time cost : %.2fs\n", (trainEnd - trainStart) / MILLI_SECOND);
    double addStart = GetMillisecs();
    index.add(ntotal, base.data());
    double addEnd = GetMillisecs();
    printf("add time cost : %.2fs\n", (addEnd - addStart) / MILLI_SECOND);
    double updateStart = GetMillisecs();
    index.update();
    double updateEnd = GetMillisecs();
    printf("update time cost : %.2fs\n", (updateEnd - updateStart) / MILLI_SECOND);
    printf("=> start Qps Test!\n");

    index.updateTParams(L2NPROBE, L3NPROBE);
    index.setNumProbes(NPROBE);
    std::vector<float> dist(searchNum * static_cast<size_t>(topk), 0);
    std::vector<faiss::idx_t> labelTopk(searchNum * static_cast<size_t>(topk), 0);

    double sqtSearchStart = GetMillisecs();
    index.search(searchNum, dataBaseFloat.query.data(), topk, dist.data(), labelTopk.data());
    double sqtSearchEnd = GetMillisecs();

    printf("searchNum = %d, sqt search cost: %lf\n", searchNum, sqtSearchEnd - sqtSearchStart);
    DataProccess(searchNum, topk, dataBaseFloat, index, labelTopk);
}

void TestIVFSQT(int niter, int ncentroids)
{
    size_t ntotal = 1000000;
    size_t queryNum = 1000000;
    size_t learnNum = ntotal / 10;
    int gtNum = 100;
    int centroid = 256;
    try
    {
        // resource size 1024 * 1024 * 1024 = 1GB
        faiss::ascend::AscendIndexIVFSQTConfig conf({0}, static_cast<int64_t>(1024 * 1024 * 1024));
        conf.cp.niter = niter;
        conf.useKmeansPP = true;
        conf.cp.max_points_per_centroid = centroid;
        faiss::ascend::AscendIndexIVFSQT index(DIM_IN, DIM_OUT, ncentroids,
                                               faiss::ScalarQuantizer::QuantizerType::QT_8bit,
                                               faiss::METRIC_INNER_PRODUCT, conf);
        std::vector<int8_t> baseInt8(ntotal * DIM_IN);
        for (size_t i = 0; i < ntotal * DIM_IN; i++)
        {
            baseInt8[i] = RandomInt8();
        }
        std::vector<int8_t> learnInt8(learnNum * DIM_IN);
        for (size_t i = 0; i < learnNum * DIM_IN; i++)
        {
            learnInt8[i] = RandomInt8();
        }
        std::vector<int8_t> queryInt8(queryNum * DIM_IN);
        std::vector<int64_t> gt(queryNum * gtNum, 0);
        for (size_t q = 0; q < queryNum; q++)
        {
            size_t idx;
            GeneratorRandomIndex(ntotal, idx);
            for (size_t d = 0; d < DIM_IN; d++)
            {
                queryInt8[q * DIM_IN + d] = baseInt8[idx * DIM_IN + d];
            }
            gt[q * gtNum] = static_cast<int64_t>(idx);
        }
        // int8 to float，除以128.0是为了将其映射到-1.0到1.0的区间内。
        float intToFloat = 128.0;
        std::vector<float> base(ntotal * DIM_IN);
        for (size_t i = 0; i < ntotal * DIM_IN; i++)
        {
            base[i] = static_cast<float>(baseInt8[i]) / intToFloat;
        }
        std::vector<float> query(queryNum * DIM_IN);
        for (size_t i = 0; i < queryNum * DIM_IN; i++)
        {
            query[i] = static_cast<float>(queryInt8[i]) / intToFloat;
        }
        std::vector<float> learn(learnNum * DIM_IN);
        for (size_t i = 0; i < learnNum * DIM_IN; i++)
        {
            learn[i] = static_cast<float>(learnInt8[i]) / intToFloat;
        }
        dataFloat dataBaseFloat(base, learn, query, gt);
        SearchProccess(index, ntotal, base, learn, dataBaseFloat);
    }
    catch (std::exception &e)
    {
        printf("%s\n", e.what());
    }
}

}  // namespace

int main(int argc, char **argv)
{
    int ncentroids = 16384;  // 分桶的数目
    int niter = 16;
    printf("Start Test\n");
    TestIVFSQT(niter, ncentroids);
}
