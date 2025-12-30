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


#include <sys/time.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <fstream>

#include <stdint.h>
#include <random>

namespace {
    typedef std::unordered_map<int, float> recallMap;
    typedef std::unordered_map<std::string, size_t> shapeMap;

    int deviceId = 3;

    // random generator that can be used in multithreaded contexts
    struct RandomGenerator {
        std::mt19937 mt;

        /// random positive integer
        int rand_int();

        /// random int64_t
        int64_t rand_int64();

        /// generate random integer between 0 and max-1
        int rand_int(int max);

        /// between 0 and 1
        float rand_float();

        double rand_double();

        explicit RandomGenerator(int64_t seed = 1234);
    };

    RandomGenerator::RandomGenerator(int64_t seed) : mt((unsigned int)seed) {}

    int RandomGenerator::rand_int()
    {
        return mt() & 0x7fffffff;
    }

    int64_t RandomGenerator::rand_int64()
    {
        return int64_t(rand_int()) | (int64_t(rand_int()) << 31);
    }

    int RandomGenerator::rand_int(int max)
    {
        return mt() % max;
    }

    float RandomGenerator::rand_float()
    {
        return mt() / float(mt.max());
    }

    double RandomGenerator::rand_double()
    {
        return mt() / double(mt.max());
    }

    void rand_perm(int* perm, size_t n, int64_t seed)
    {
        for (size_t i = 0; i < n; i++)
            perm[i] = i;

        RandomGenerator rng(seed);

        for (size_t i = 0; i + 1 < n; i++) {
            int i2 = i + rng.rand_int(n - i);
            std::swap(perm[i], perm[i2]);
        }
    }

    void Norm(float* data, int n, int dim)
    {
#pragma omp parallel for if(n > 1)
        for (size_t i = 0; i < n; ++i) {
            float l2norm = 0;
            for (int j = 0; j < dim; ++j) {
                l2norm += data[i * dim + j] * data[i * dim + j];
            }
            l2norm = std::sqrt(l2norm);

            for (int j = 0; j < dim; ++j) {
                data[i * dim + j] = data[i * dim + j] / l2norm;
            }
        }
    }

    void writeBinary(char* data, size_t size, std::string filename)
    {
        std::ofstream fout(filename, std::ios_base::out | std::ios_base::binary);
        fout.write(data, size);
        fout.close();
    }

    inline double GetMillisecs()
    {
        struct timeval tv = { 0, 0 };
        gettimeofday(&tv, nullptr);
        return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
    }

    const int64_t ID_START = 10000000000L;

/**
 * calculate Recall
 */
template<class T>
recallMap calRecall(std::vector<T> label, int64_t* gt, int shape)
{
    recallMap Map;
    Map[1] = 0;
    Map[10] = 0;
    Map[100] = 0;
    int k = label.size() / shape;

    for (int i = 0; i < shape; i++) {
        for (int j = 0; j < k; j++) {
            if (gt[i * k] == label[i * k + j]) {
                Map[100]++;
                switch (j) {
                    case 0:
                        Map[1]++;
                        Map[10]++;
                        break;
                    case 1 ... 9:
                        Map[10]++;
                        break;
                    default:
                        break;
                }
                break;
            }
        }
    }
    Map[1] = Map[1] / shape * 100;
    Map[10] = Map[10] / shape * 100;
    Map[100] = Map[100] / shape * 100;
    return Map;
}


    template<class T>
    recallMap calRecall_xdh(std::vector<T> label, int64_t* gt, int shape)
    {
        recallMap Map;
        Map[1] = 0;
        Map[10] = 0;
        Map[25] = 0;
        Map[50] = 0;
        Map[75] = 0;
        Map[100] = 0;
        Map[500] = 0;
        int k = label.size() / shape;

        for (int i = 0; i < shape; i++) {
            for (int j = 0; j < k; j++) {
                if (gt[i] == label[i * k + j]) {
                    switch (j) {
                        case 0:
                            Map[1]++;
                            Map[10]++;
                            Map[25]++;
                            Map[50]++;
                            Map[75]++;
                            Map[100]++;
                            Map[500]++;
                            break;
                        case 1 ... 9:
                            Map[10]++;
                            Map[25]++;
                            Map[50]++;
                            Map[75]++;
                            Map[100]++;
                            Map[500]++;
                            break;
                        case 10 ... 24:
                            Map[25]++;
                            Map[50]++;
                            Map[75]++;
                            Map[100]++;
                            Map[500]++;
                            break;
                        case 25 ... 49:
                            Map[50]++;
                            Map[75]++;
                            Map[100]++;
                            Map[500]++;
                            break;
                        case 50 ... 74:
                            Map[75]++;
                            Map[100]++;
                            Map[500]++;
                            break;
                        case 75 ... 99:
                            Map[100]++;
                            Map[500]++;
                            break;
                        case 100 ... 499:
                            Map[500]++;
                            break;
                        default:
                            break;
                    }
                    break;
                }
            }
        }
        Map[1] = Map[1] / shape * 100;
        Map[10] = Map[10] / shape * 100;
        Map[25] = Map[25] / shape * 100;
        Map[50] = Map[50] / shape * 100;
        Map[75] = Map[75] / shape * 100;
        Map[100] = Map[100] / shape * 100;
        Map[500] = Map[500] / shape * 100;
        return Map;
    }


    template<class T>
    recallMap calRecall_xdh_int(std::vector<T> label, int* gt, int shape)
    {
        recallMap Map;
        Map[1] = 0;
        Map[10] = 0;
        Map[25] = 0;
        Map[50] = 0;
        Map[75] = 0;
        Map[100] = 0;
        Map[500] = 0;
        int k = label.size() / shape;

        for (int i = 0; i < shape; i++) {
            for (int j = 0; j < k; j++) {
                if (gt[i] == label[i * k + j]) {
                    switch (j) {
                        case 0:
                            Map[1]++;
                            Map[10]++;
                            Map[25]++;
                            Map[50]++;
                            Map[75]++;
                            Map[100]++;
                            Map[500]++;
                            break;
                        case 1 ... 9:
                            Map[10]++;
                            Map[25]++;
                            Map[50]++;
                            Map[75]++;
                            Map[100]++;
                            Map[500]++;
                            break;
                        case 10 ... 24:
                            Map[25]++;
                            Map[50]++;
                            Map[75]++;
                            Map[100]++;
                            Map[500]++;
                            break;
                        case 25 ... 49:
                            Map[50]++;
                            Map[75]++;
                            Map[100]++;
                            Map[500]++;
                            break;
                        case 50 ... 74:
                            Map[75]++;
                            Map[100]++;
                            Map[500]++;
                            break;
                        case 75 ... 99:
                            Map[100]++;
                            Map[500]++;
                            break;
                        case 100 ... 499:
                            Map[500]++;
                            break;
                        default:
                            break;
                    }
                    break;
                }
            }
        }
        Map[1] = Map[1] / shape * 100;
        Map[10] = Map[10] / shape * 100;
        Map[25] = Map[25] / shape * 100;
        Map[50] = Map[50] / shape * 100;
        Map[75] = Map[75] / shape * 100;
        Map[100] = Map[100] / shape * 100;
        Map[500] = Map[500] / shape * 100;
        return Map;
    }

template<class T>
double calNewRecallHelper(std::vector<T> &label, int64_t* gt, int shape, int P)
{
    int k = label.size() / shape;
    double result = 0.0;
    for (int i = 0; i < shape; ++i) {
        int hit = 0;
        int valid_gt_num = 0;
        auto it = label.begin() + i * k;
        for (int j = 0; j < k; ++j) {
            int64_t one_gt = gt[i*k+j];
            if (one_gt == -1) {
                break;
            }
            valid_gt_num += 1;
            if (std::find(it, it+P, one_gt + ID_START) != it+P) {
                hit += 1;
            }
        }
        if (valid_gt_num > 0) {
            result += (double)hit / valid_gt_num;
        }
    }
    return result / shape * 100;
}

/**
 * calculate New Recall
 */
template<class T>
recallMap calNewRecall(std::vector<T> &label, int64_t* gt, int shape)
{
    recallMap Map;
    Map[1] = calNewRecallHelper(label, gt, shape, 1);
    Map[10] = calNewRecallHelper(label, gt, shape, 10);
    Map[32] = calNewRecallHelper(label, gt, shape, 32);
    Map[100] = calNewRecallHelper(label, gt, shape, 100);
    return Map;
}

/**
 * Top and Recall Log
 */
    template<class T>
    void resultLog(std::vector<T> &label, int64_t* gt, int shape, int nprobe, double timeCost)
    {
        recallMap Top = calRecall(label, gt, shape);
        printf("nprobe = %2d, t1 = %.2f, t10 = %.2f, t100 = %.2f | ",
               nprobe, Top[1], Top[10], Top[100]);

        recallMap Recall = calNewRecall(label, gt, shape);
        printf("nprobe = %2d, r1 = %.2f, r10 = %.2f, r32 = %.2f, r100 = %.2f, QPS = %.2f, duration = %.2fms\n",
               nprobe, Recall[1], Recall[10], Recall[32], Recall[100], 1000 * shape / timeCost, timeCost);
    }


    /**
    * readBin File
    */
    template <class T> static bool readBin(std::vector<T> &data, const std::string &fileName)
    {
        std::ifstream inFile(fileName, std::ios::binary | std::ios::in);
        if (!inFile.is_open()) {
            printf("Open file %s failed!\n", fileName.c_str());
            return false;
        }

        inFile.seekg(0, std::ios_base::end);
        size_t fileSize = inFile.tellg();

        inFile.seekg(0, std::ios_base::beg);
        data.resize(fileSize / sizeof(T));
        inFile.read((char *)data.data(), fileSize);

        return true;
    }

    class TestData {
    public:
        TestData(const std::string &baseName, const std::string &learnName,
                 const std::string &queryName, const std::string &gtName, int d)
        {
            this->baseName = baseName;
            this->learnName = learnName;
            this->queryName = queryName;
            this->gtName = gtName;
            this->dim = d;
            readData();
        }

        bool shapeAnalysis()
        {
            dataShape["learn"] = nLearn;
            dataShape["base"] = nBase;
            dataShape["query"] = nQuery;
            dataShape["dim"] = dim;

            for (auto i: dataShape) std::cout << i.first << " num is: " << i.second << "\n";
            return true;
        }

        bool readData()
        {
            readBin(gt, gtName);
            readBin(base, baseName);
            readBin(learn, learnName);
            readBin(query, queryName);
            // dim = query.size() / (gt.size() / topk);
            nBase = base.size() / dim;
            nLearn = learn.size() / dim;
            nQuery = query.size() / dim;
            shapeAnalysis();
            return true;
        }

    public:
        std::vector<float> base;
        std::vector<float> learn;
        std::vector<float> query;
        std::vector<int64_t> gt;

        int dim = 256;
        int topk = 100;
        size_t nBase = 0;
        size_t nLearn = 0;
        size_t nQuery = 0;
        shapeMap dataShape;

    private:
        std::string baseName;
        std::string learnName;
        std::string queryName;
        std::string gtName;
    };

    TestData readBinFeature(std::string dataset_dir, int d)
    {
        std::string learn_path = dataset_dir + "learn.bin";
        std::string base_path = dataset_dir + "base.bin";
        std::string query_path = dataset_dir + "query.bin";
        std::string gt_path = dataset_dir + "gt.bin";

        TestData dataSet = TestData(base_path, learn_path, query_path, gt_path, d);
        return dataSet;
    }
}
