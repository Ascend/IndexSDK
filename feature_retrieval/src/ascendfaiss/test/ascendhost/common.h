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


#include <unordered_map>
#include <fstream>

namespace {

typedef std::unordered_map<std::string, size_t> shapeMap;

size_t physical_memory_used_by_process()
{
    FILE* file = fopen("/proc/self/status", "r");
    int result = -1;
    char line[128];

    while (fgets(line, 128, file) != nullptr) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            int len = strlen(line);

            const char* p = line;
            for (; isdigit(*p) == false; ++p) {}

            line[len - 3] = 0;
            result = atoi(p);

            break;
        }
    }

    fclose(file);

    return result;
}

template<class T>
static bool readBin(std::vector<T> &data, const std::string &fileName)
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
             const std::string &queryName, const std::string &gtName) 
    {
        this->baseName = baseName;
        this->learnName = learnName;
        this->queryName = queryName;
        this->gtName = gtName;
        readData();
    }

    bool shapeAnalysis()
    {
        dataShape["learn"] = nLearn;
        dataShape["base"] = nBase;
        dataShape["query"] = nQuery;
        dataShape["dim"] = dim;

        for (auto i: dataShape) {
            std::cout << i.first << " num is: " << i.second << "\n";
        }
        return true;
    }

    bool readData()
    {   
        readBin(base, baseName);
        readBin(learn, learnName);
        readBin(query, queryName);
        readBin(gt, gtName);

        dim = query.size() / (gt.size() / topk);
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

    size_t dim = 256;
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

TestData readBinFeature(std::string dataset_dir)
{
    std::string learn_path = dataset_dir + "learn.bin";
    std::string base_path = dataset_dir + "base.bin";
    std::string query_path = dataset_dir + "query.bin";
    std::string gt_path = dataset_dir + "gt.bin";

    return TestData(base_path, learn_path, query_path, gt_path);
}
}

