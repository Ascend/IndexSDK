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


#ifndef TEST_DATA_H
#define TEST_DATA_H

#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <vector>
#include <string>
#include <memory>
#include <fstream>


class TestData {
public:
    TestData() {}

    static double GetMillisecs()
    {
        struct timeval tv = { 0, 0 };
        gettimeofday(&tv, nullptr);
        return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
    }

    int GetDim() const
    {
        return dim;
    }

    virtual void SetDim(int dim)
    {
        this->dim = dim;
    }

    virtual bool FetchData() = 0;

    virtual bool HasGroundTruth() const
    {
        return true;
    }

    void Norm()
    {
        printf("start norm base, learn, query\n");
        double t0 = GetMillisecs();
        Norm(base, nBase, dim);
        Norm(learn, nLearn, dim);
        Norm(query, nQuery, dim);
        double t1 = GetMillisecs();
        printf("end norm data, cost=%fms\n", (t1 - t0));
        normalized = true;
    }

    void Norm(std::vector<float> &data, size_t total, int d)
    {
#pragma omp parallel for
        for (size_t i = 0; i < total; ++i) {
            float mod = 0;
            for (int j = 0; j < d; ++j) {
                mod += data[i * d + j] * data[i * d + j];
            }

            mod = sqrt(mod);
            for (int j = 0; j < d; ++j) {
                data[i * d + j] = data[i * d + j] / mod;
            }
        }
    }

    size_t GetBaseSize() const
    {
        return nBase;
    }

    size_t GetQuerySize() const
    {
        return nQuery;
    }

    size_t GetLearnSize() const
    {
        return nLearn;
    }

    const std::vector<float> &GetBase() const
    {
        return base;
    }

    const std::vector<float> &GetLearn() const
    {
        return learn;
    }

    const std::vector<float> &GetQuery() const
    {
        return query;
    }

    const std::vector<int64_t> &GetGroudTruth() const
    {
        return groudTruth;
    }

    void Release()
    {
        std::vector<float>().swap(base);
        std::vector<float>().swap(learn);
        std::vector<float>().swap(query);
        std::vector<int64_t>().swap(groudTruth);
    }

    bool GetNormalized() const
    {
        return normalized;
    }

    void SetNormalized(bool normalized)
    {
        this->normalized = normalized;
    }

    int GetScale() const
    {
        return scale;
    }

    void SetScale(int scale)
    {
        printf("set scale %d\n", scale);
        this->scale = scale;
    }

    int GetGTK() const
    {
        return groundTruthTopK;
    }

    void EvaluateResultByCoverageRate(std::vector<int64_t> &label, size_t k)
    {
        int n1 = 0;
        int n10 = 0;
        int n100 = 0;
        int n500 = 0;
        int n1000 = 0;
        int n5000 = 0;

        for (size_t iq = 0; iq < nQuery; iq++) {
            std::vector<int64_t> target(groudTruth.begin() + iq * groundTruthTopK, groudTruth.begin() + iq * groundTruthTopK + k);
            for (auto tg : target) {
                auto it = std::find(label.begin() + iq * groundTruthTopK, label.begin() + (iq+1) * groundTruthTopK, tg);
                size_t pos = distance(label.begin() + iq * groundTruthTopK, it);
                if (pos < 5000) {
                    n5000++;
                }
                if (pos < 1000) {
                    n1000++;
                }
                if (pos < 500) {
                    n500++;
                }
                if (pos < 100) {
                    n100++;
                }
                if (pos < 10) {
                    n10++;
                }
                if (pos < 1) {
                    n1++;
                }
            }
        }

        printf("k[%lu] 1[%.4f] 10[%.4f] 100[%.4f] 500[%.4f] 1000[%.4f] 5000[%.4f]\r\n", k,
            n1 / static_cast<float>(nQuery) / static_cast<float>(k),
            n10 / static_cast<float>(nQuery) / static_cast<float>(k),
            n100 / static_cast<float>(nQuery) / static_cast<float>(k),
            n500 / static_cast<float>(nQuery) / static_cast<float>(k),
            n1000 / static_cast<float>(nQuery) / static_cast<float>(k),
            n5000 / static_cast<float>(nQuery) / static_cast<float>(k));
    }

    void EvaluateResult(std::vector<int64_t> &label)
    {
        int n1 = 0, n10 = 0, n100 = 0, n1000 = 0;
        for (size_t i = 0; i < nQuery; i++) {
            int64_t gtNn = groudTruth[i * groundTruthTopK];
            for (int j = 0; j < groundTruthTopK; j++) {
                if (label[i * groundTruthTopK + j] == gtNn) {
                    if (j < 1) {
                        n1++;
                    }
                    if (j < 10) {
                        n10++;
                    }
                    if (j < 100) {
                        n100++;
                    }
                    if (j < 1000) {
                        n1000++;
                    }
                }
            }
        }
        printf("R@1 = %.4f, R@10 = %.4f, R@100 = %.4f, R@1000 = %.4f\n", n1 / static_cast<float>(nQuery),
            n10 / static_cast<float>(nQuery), n100 / static_cast<float>(nQuery), n1000 / static_cast<float>(nQuery));

        EvaluateResultByCoverageRate(label, 1);
        EvaluateResultByCoverageRate(label, 10);
        EvaluateResultByCoverageRate(label, 100);
    }

protected:
    int dim = 128;

    size_t nBase = 0;
    size_t nLearn = 0;
    size_t nQuery = 0;

    int scale = 0;
    bool normalized = false;

    std::vector<float> base;
    std::vector<float> learn;
    std::vector<float> query;

    int groundTruthTopK = 1000;
    std::vector<int64_t> groudTruth;
};

class TestDataFile : public TestData {
public:
    TestDataFile(const std::string &baseName, const std::string &learnName, const std::string &queryName,
        const std::string &gtName)
        : baseFile(baseName), learnFile(learnName), queryFile(queryName), gtFile(gtName)
    {}

protected:
    std::string baseFile;
    std::string learnFile;
    std::string queryFile;
    std::string gtFile;
};

class TestDataSift1B : public TestDataFile {
public:
    TestDataSift1B(const std::string &baseName, const std::string &learnName, const std::string &queryName,
        const std::string &gtName)
        : TestDataFile(baseName, learnName, queryName, gtName) {}

    void LoadByteData(const std::string &fileName, int d, size_t &num, std::vector<float> &data)
    {
        std::vector<int8_t> xb;
        std::ifstream inFile(fileName, std::ios::binary | std::ios::in);
        if (!inFile.is_open()) {
            printf("Open file %s failed!\n", fileName.c_str());
        }

        inFile.seekg(0, std::ios_base::end);
        size_t fileSize = inFile.tellg();

        inFile.seekg(0, std::ios_base::beg);
        xb.resize(fileSize / sizeof(int8_t));
        inFile.read(reinterpret_cast<char *>(xb.data()), fileSize);

        data.resize(xb.size());
        std::transform(xb.begin(), xb.end(), data.begin(), [this] (int8_t x) {
            return (scale != 0) ? (static_cast<float>(x) / scale) : static_cast<float>(x);
        });

        num = xb.size() / d;
    }

    bool FetchData()
    {
        LoadByteData(baseFile, dim, nBase, base);
        LoadByteData(learnFile, dim, nLearn, learn);
        LoadByteData(queryFile, dim, nQuery, query);
        ReadBin(groudTruth, gtFile);
        groundTruthTopK = groudTruth.size() / nQuery;

        return true;
    }

    template <class T>
    bool ReadBin(std::vector<T> &data, const std::string &fileName)
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
};

#endif